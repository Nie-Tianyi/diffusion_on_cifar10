"""DiT (Diffusion Transformer) model for noise prediction.

Implements the architecture from Peebles & Xie (2023):
"Scalable Diffusion Models with Transformers"

Key design:
- Image → patch tokens via Conv2d stride
- Fixed 2D sinusoidal position embeddings (not learned)
- N transformer blocks with adaLN-Zero conditioning on timestep
- Final layerNorm + linear → unpatchify → pixel-space output

adaLN-Zero: the modulation MLP outputs (shift, scale, gate) for both
MSA and MLP sub-layers. Output linear is zero-initialized so the block
starts as identity, which greatly improves training stability.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# 2D sinusoidal position embedding (fixed, not learned)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal embedding à la "Attention Is All You Need".

    Args:
        embed_dim: output dimension (must be even).
        pos: (N,) positions (float tensor).

    Returns:
        (N, embed_dim) — sin/cos interleaved.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000.0 ** (2 * omega / embed_dim))
    out = pos.float().unsqueeze(-1) * omega.unsqueeze(0)  # (N, half)
    return torch.cat([out.sin(), out.cos()], dim=-1)       # (N, embed_dim)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sinusoidal position embedding for a square grid.

    Each patch position (row, col) gets an embedding formed by
    concatenating the 1D sin-cos embedding of its row and column.

    Args:
        embed_dim: output dimension (must be even — half for row, half for col).
        grid_size: height = width of the patch grid.

    Returns:
        (grid_size², embed_dim)
    """
    assert embed_dim % 2 == 0
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)  # (N, 2) — (row, col) for each patch

    emb_row = get_1d_sincos_pos_embed(embed_dim // 2, grid[:, 0])
    emb_col = get_1d_sincos_pos_embed(embed_dim // 2, grid[:, 1])
    return torch.cat([emb_row, emb_col], dim=-1)  # (N, embed_dim)


# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal → MLP)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embed integer diffusion timesteps into a continuous conditioning vector.

    Uses sinusoidal frequencies (like position encoding) followed by a
    2-layer MLP with SiLU activation.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: (B,) integer timesteps.
            dim: output dimensionality.
            max_period: controls the lowest frequency.

        Returns:
            (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)   # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)


# ---------------------------------------------------------------------------
# DiT building blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    The conditioning vector `c` (from timestep embedding) is projected
    through a small SiLU→Linear MLP to produce six modulation parameters:
      (shift_msa, scale_msa, gate_msa) for the attention sub-layer
      (shift_mlp, scale_mlp, gate_mlp) for the MLP sub-layer

    The output Linear is zero-initialized so the block is an identity
    function at the start of training — this is the "Zero" in adaLN-Zero
    and is critical for stable transformer diffusion training.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        # ---- Layer norms (no learnable affine — modulation handles that) ----
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # ---- Multi-head self-attention ----
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True,
        )

        # ---- MLP (GELU as in DiT paper) ----
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # ---- adaLN modulation: c → (shift, scale, gate) × 2 sub-layers ----
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        # Zero-init the output layer for identity-at-initialisation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) tokens,  c: (B, C) conditioning → (B, N, C)"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # ---- Self-attention with adaLN ----
        x_norm = self.norm1(x)
        x_mod = x_norm * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # ---- MLP with adaLN ----
        x_norm = self.norm2(x)
        x_mod = x_norm * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)

        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection → patch pixels.

    Uses the same modulation pattern (without gate — only shift & scale
    are needed for the output projection).
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),  # shift + scale only
        )

        # Zero-init modulation → identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # Output projection: Xavier uniform + zero bias
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C), c: (B, C) → (B, N, P²·out_ch)"""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer for ε-prediction on CIFAR-10 (32×32).

    Forward path
    ------------
    Image (B,3,32,32)
      → PatchEmbed Conv2d → (B, hidden, H', W') → flatten → (B, N, hidden)
      → + fixed 2D sin-cos position embedding
      → TimeEmbed(t) → conditioning vector c (B, hidden)
      → N × DiTBlock(x, c)
      → FinalLayer(x, c) → (B, N, P²·3)
      → unpatchify → (B, 3, 32, 32) predicted noise
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.patch_size = config.dit_patch_size
        self.hidden_size = config.dit_hidden_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.image_size = config.image_size

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # ---- Patch embedding (Conv2d stride = patch_size) ----
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dit_hidden_size,
            kernel_size=config.dit_patch_size,
            stride=config.dit_patch_size,
        )

        # ---- Fixed 2D sinusoidal position embedding ----
        pos_embed = get_2d_sincos_pos_embed(config.dit_hidden_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, C)

        # ---- Timestep embedding ----
        self.time_embed = TimestepEmbedder(config.dit_hidden_size)

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.dit_hidden_size,
                config.dit_num_heads,
                config.dit_mlp_ratio,
            )
            for _ in range(config.dit_depth)
        ])

        # ---- Final layer ----
        self.final_layer = FinalLayer(
            config.dit_hidden_size,
            config.dit_patch_size,
            config.out_channels,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for weights not already initialised.

        adaLN_modulation output layers and FinalLayer.linear are already
        initialised in their respective __init__ methods — we don't touch
        those here.
        """
        # Patch embed
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # Timestep embedder MLP
        for layer in self.time_embed.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Transformer block MLP sub-layers
        # (MHA and adaLN_modulation are handled by PyTorch / zero-init above)
        for block in self.blocks:
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to an image.

        (B, N, P²·C) → (B, C, H, W)
        """
        B = x.shape[0]
        P, G, C = self.patch_size, self.grid_size, self.out_channels
        x = x.reshape(B, G, G, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, G, P, G, P)
        x = x.reshape(B, C, G * P, G * P)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy image and timestep.

        Args:
            x: (B, C, H, W) noisy image in [-1, 1].
            t: (B,) integer diffusion timesteps.

        Returns:
            (B, C, H, W) predicted noise ε_θ(x_t, t).
        """
        B = x.shape[0]

        # ---- Patch embed ----
        x = self.patch_embed(x)             # (B, hidden, G, G)
        x = x.flatten(2).transpose(1, 2)    # (B, N, hidden)

        # ---- Add position embedding ----
        x = x + self.pos_embed

        # ---- Conditioning ----
        c = self.time_embed(t)              # (B, hidden)

        # ---- Transformer blocks ----
        for block in self.blocks:
            x = block(x, c)

        # ---- Final layer + unpatchify ----
        x = self.final_layer(x, c)          # (B, N, P²·C)
        x = self.unpatchify(x)              # (B, C, H, W)

        return x
