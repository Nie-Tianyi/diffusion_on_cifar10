"""UNet model for DDPM noise prediction.

Architecture follows the DDPM paper (Ho et al. 2020) with:
- Sinusoidal time-step embeddings
- ResNet blocks with group-normalisation
- Self-attention at selected resolutions
- Skip connections from encoder to decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Transformer-style sinusoidal position encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, dim)"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time-embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  t_emb: (B, time_emb_dim)"""
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Inject time embedding
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return h + x


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """One encoder stage: ResBlocks + optional attention + down-sample."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        num_heads: int,
        downsample: bool,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ch = in_ch
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
            ch = out_ch
            self.attentions.append(
                AttentionBlock(ch, num_heads) if use_attention else nn.Identity()
            )

        self.downsample = (
            nn.Conv2d(ch, ch, 3, stride=2, padding=1) if downsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, t_emb)
            x = attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """One decoder stage: ResBlocks + optional attention + up-sample."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        time_emb_dim: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        num_heads: int,
        upsample: bool,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ch = in_ch
        for i in range(num_res_blocks):
            if i == 0:
                self.res_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
            else:
                self.res_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
            ch = out_ch
            self.attentions.append(
                AttentionBlock(ch, num_heads) if use_attention else nn.Identity()
            )

        self.upsample = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            )
            if upsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, t_emb)
            x = attn(x)
        x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Full UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """DDPM U-Net for ε-prediction on 32×32 CIFAR-10."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # --- Time embedding ---
        time_emb_dim = config.base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.base_channels),
            nn.Linear(config.base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Initial convolution ---
        self.init_conv = nn.Conv2d(config.in_channels, config.base_channels, 3, padding=1)

        # --- Channel schedule ---
        chs = [config.base_channels * m for m in config.channel_multipliers]
        resolutions = [config.image_size // (2 ** i) for i in range(len(chs))]

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        in_ch = config.base_channels
        for i, (out_ch, res) in enumerate(zip(chs, resolutions)):
            use_attn = res in config.attention_resolutions
            is_last = i == len(chs) - 1
            self.down_blocks.append(
                DownBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    time_emb_dim=time_emb_dim,
                    num_res_blocks=config.num_res_blocks,
                    dropout=config.dropout,
                    use_attention=use_attn,
                    num_heads=config.num_heads,
                    downsample=not is_last,
                )
            )
            in_ch = out_ch

        # --- Bottleneck ---
        mid_ch = chs[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim, config.dropout)
        self.mid_attn = AttentionBlock(mid_ch, config.num_heads)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, config.dropout)

        # --- Decoder ---
        rev_chs = list(reversed(chs))
        skip_chs = list(reversed(chs))
        rev_resolutions = list(reversed(resolutions))

        self.up_blocks = nn.ModuleList()
        in_ch = mid_ch
        for i, (out_ch, skip_ch, res) in enumerate(zip(rev_chs, skip_chs, rev_resolutions)):
            use_attn = res in config.attention_resolutions
            is_last = i == len(rev_chs) - 1
            self.up_blocks.append(
                UpBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    skip_ch=skip_ch,
                    time_emb_dim=time_emb_dim,
                    num_res_blocks=config.num_res_blocks,
                    dropout=config.dropout,
                    use_attention=use_attn,
                    num_heads=config.num_heads,
                    upsample=not is_last,
                )
            )
            in_ch = out_ch

        # --- Output ---
        self.out_norm = nn.GroupNorm(min(32, rev_chs[-1]), rev_chs[-1])
        self.out_conv = nn.Conv2d(rev_chs[-1], config.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [-1,1]  t: (B,) int timesteps → predicted noise (B,3,H,W)"""
        t_emb = self.time_embed(t)

        h = self.init_conv(x)

        # Encoder
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            h, skip = down(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for up in self.up_blocks:
            skip = skips.pop()
            h = up(h, skip, t_emb)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h
