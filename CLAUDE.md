# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DDPM diffusion model training pipeline for CIFAR-10. Implements the Denoising Diffusion Probabilistic Models paper (Ho et al. 2020) with two backbone choices: a convolutional U-Net and a DiT (Diffusion Transformer, Peebles & Xie 2023). Target hardware is an RTX 5080 16 GB, but the code auto-scales to any GPU or CPU.

## Commands

```bash
# Install dependencies
uv sync

# Train with default config (RTX 5080 16GB optimised)
uv run python main.py

# Quick test run
uv run python main.py --epochs 50 --batch-size 128

# Resume from checkpoint
uv run python main.py --resume ./outputs/<run_id>/checkpoints/checkpoint_epoch_0050.pt

# Disable mixed-precision (debugging)
uv run python main.py --no-amp

# DDIM accelerated sampling (50 deterministic steps)
uv run python main.py --sampler ddim --ddim-steps 50

# DDIM with stochasticity (eta=1 recovers DDPM behaviour)
uv run python main.py --sampler ddim --ddim-steps 100 --ddim-eta 1.0

# Train with DiT-Small (~33M params, comparable to UNet)
uv run python main.py --model dit

# Train with DiT-Base (~131M params, higher quality)
uv run python main.py --model dit --batch-size 128

# Generate images from a DiT checkpoint
uv run python generate.py --checkpoint outputs/<run_id>/checkpoints/final.pt

# DiT smoke test
uv run python -c "
import torch
from config import dit_s_config
from dit import DiT
from diffusion import GaussianDiffusion
cfg = dit_s_config()
model = DiT(cfg.model).cuda()
diff = GaussianDiffusion(cfg.training.timesteps, cfg.training.beta_start, cfg.training.beta_end)
x = torch.randn(8, 3, 32, 32, device='cuda')
t = torch.randint(0, 1000, (8,), device='cuda')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
print(f'Loss: {diff.p_losses(model, x, t).item():.4f}')
with torch.no_grad():
    s = diff.ddim_sample_loop(model, (4, 3, 32, 32), 'cuda', ddim_steps=50)
    print(f'Sample shape: {s.shape}')
"

# Smoke test — verify model init, forward pass, and sampling all work
uv run python -c "
import torch
from config import cifar10_config
from unet import UNet
from diffusion import GaussianDiffusion
cfg = cifar10_config()
model = UNet(cfg.model).cuda()
diff = GaussianDiffusion(cfg.training.timesteps, cfg.training.beta_start, cfg.training.beta_end)
x = torch.randn(8, 3, 32, 32, device='cuda')
t = torch.randint(0, 1000, (8,), device='cuda')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
print(f'Loss: {diff.p_losses(model, x, t).item():.4f}')
with torch.no_grad():
    s = diff.p_sample_loop(model, (4, 3, 32, 32), 'cuda')
    print(f'Sample shape: {s.shape}')
"
```

## Architecture

### Data flow

```
CIFAR-10 image (3×32×32, [-1,1])
  → random timestep t ~ Uniform(0, T-1)
  → forward diffusion: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε  (no ε needed if predicting noise)
  → Model (UNet or DiT) predicts noise ε_θ(x_t, t)
  → MSE loss between ε_θ and true ε
  → backprop through model only
```

**Sampling (reverse process):** two samplers available:

- **DDPM** (`p_sample_loop`): start from pure Gaussian noise x_T, iterate t = T−1 → 0 using the DDPM ancestral sampler. 1000 steps, stochastic.
- **DDIM** (`ddim_sample_loop`): linearly-spaced subsequence of timesteps (default 50), deterministic by default (`eta=0`). Same trained model, much faster sampling. `eta=1` adds DDPM-level stochasticity back.

Both use EMA shadow weights for better quality. x₀ clipping to [-1, 1] is applied in both to prevent numerical explosion.

### Module dependency graph

```
config.py          ← dataclasses: ModelConfig, TrainingConfig, Config
unet.py           ← UNet (imports ModelConfig from config)
dit.py             ← DiT (imports ModelConfig from config)
diffusion.py       ← GaussianDiffusion (standalone, no project imports)
sampling.py        ← sample_and_save (imports GaussianDiffusion)
main.py            ← training loop (imports all above)
generate.py        ← standalone inference (imports model/dit + diffusion)
```

`diffusion.py` and `unet.py`/`dit.py` are independent; the training loop wires them together.

### Key design choices

- **ε-prediction, not x₀-prediction.** The UNet estimates the noise that was added, matching the DDPM "simple" objective. This works better in practice than predicting x₀ directly.
- **BF16 mixed precision via `torch.amp.autocast`.** RTX 5080 (Blackwell) runs BF16 natively. No `GradScaler` needed for BF16 (unlike FP16) — the scaler is included as a harmless safety net.
- **EMA with warmup + decay 0.9999.** During training, an exponential moving average of weights is maintained. Sampling always uses EMA weights; they are applied/restored around each sample step. A dynamic warmup schedule (`current_decay = min(0.9999, (1+step)/(10+step))`) prevents the shadow model from retaining too much random initial weight early in training — without this, early samples are pure-colour blocks.
- **Cosine β schedule (default).** Produces more even noise-level coverage than linear. A `linear_beta_schedule` is also available as a fallback. Configurable via `TrainingConfig.schedule`.
- **Self-attention only at resolution ≤ 16.** For 32×32 input this means attention at 16×16, 8×8, 4×4 layers and bottleneck — but not at 32×32 (saves parameters).
- **Channel multipliers [1, 2, 2, 2] with base 128.** Produces ~27M parameters. The up-block first ResBlock concatenates skip connection, so its input channels = in_ch + skip_ch.

### U-Net architecture

```
Init conv:       3 → 128, 32×32
Down[0]:  [2×ResBlock(128)]                     → skip@32×32 → downsample → 16×16
Down[1]:  [2×ResBlock(256) + SelfAttn]          → skip@16×16 → downsample → 8×8
Down[2]:  [2×ResBlock(256) + SelfAttn]          → skip@8×8  → downsample → 4×4
Down[3]:  [2×ResBlock(256) + SelfAttn]          → skip@4×4  (no downsample)
Mid:      ResBlock(256) + SelfAttn + ResBlock(256)            → 4×4
Up[0]:    concat(skip@4×4) → [2×ResBlock(256) + SelfAttn]     → upsample → 8×8
Up[1]:    concat(skip@8×8) → [2×ResBlock(256) + SelfAttn]     → upsample → 16×16
Up[2]:    concat(skip@16×16) → [2×ResBlock(256) + SelfAttn]   → upsample → 32×32
Up[3]:    concat(skip@32×32) → [2×ResBlock(128)]              → 32×32
Out:      GroupNorm + SiLU + Conv → 3, 32×32
```

Time embedding: sinusoidal encoding → Linear(128→512) → SiLU → Linear(512→512), injected into each ResBlock via `h + time_proj(silu(t_emb))`.

### DiT architecture

DiT (Diffusion Transformer, Peebles & Xie 2023) replaces the convolutional U-Net with a Vision Transformer that uses **adaLN-Zero** (adaptive layer norm with zero-initialized modulation) to condition on diffusion timesteps.

```
Image (3×32×32)
  → PatchEmbed: Conv2d(3→384, k=2, s=2)     → (B, 384, 16, 16)
  → flatten + transpose                       → (B, 256, 384) tokens
  → + fixed 2D sinusoidal position embed
  → TimeEmbed: sin-cos(256) → SiLU → Linear  → c (B, 384)
  → 12× DiTBlock(x, c):
      ┌─ LayerNorm (no affine)
      │  → modulate: x·(1+γ₁) + β₁  (γ₁,β₁ from adaLN_modulation(c))
      │  → MultiheadAttention(384, 6 heads)
      │  → scale by gate α₁  (from adaLN_modulation(c))
      │  → residual add
      ├─ LayerNorm (no affine)
      │  → modulate: x·(1+γ₂) + β₂  (γ₂,β₂ from adaLN_modulation(c))
      │  → MLP: Linear(384→1536) → GELU → Linear(1536→384)
      │  → scale by gate α₂  (from adaLN_modulation(c))
      └─ → residual add
  → FinalLayer: LayerNorm → modulate → Linear(384→12)
  → unpatchify                               → (B, 3, 32, 32) predicted noise
```

**adaLN-Zero:** Each `DiTBlock` has a small modulation MLP `SiLU → Linear(384→2304)` that projects the conditioning vector `c` into 6 parameters: (shift, scale, gate) × (MSA sub-layer, MLP sub-layer). The output Linear is **zero-initialized** so all modulation params start at 0 → each block is an identity function at the start of training. This is the key innovation that makes transformer diffusion training stable.

**Position embedding:** Fixed 2D sinusoidal (not learned). Row and column positions each get half the embedding dimension via sin/cos at logarithmically-spaced frequencies. Registered as a non-trainable buffer.

**DiT config presets (patch_size=2, 16×16 grid):**

| Preset | Hidden | Depth | Heads | Params | VRAM (bs=256) |
|--------|--------|-------|-------|--------|---------------|
| DiT-S  | 384    | 12    | 6     | ~32.5M | ~5-6 GB       |
| DiT-B  | 768    | 12    | 12    | ~130M  | ~10-12 GB     |

Use `dit_s_config()` or `dit_b_config()` to get a pre-built `Config`. Or use `--model dit` at the CLI (defaults to DiT-S with patch_size=2).

### Diffusion precomputed coefficients

All stored as 1-D tensors of length `T` on CPU (moved to correct device inside `_extract`). Key tensors:
- `alphas_cumprod` — ᾱ_t (stored as attribute for DDIM to index arbitrary timestep pairs)
- `sqrt_alphas_cumprod` — scale x₀ in forward diffusion
- `sqrt_one_minus_alphas_cumprod` — scale ε in forward diffusion
- `sqrt_recip_alphas_cumprod` / `sqrt_recipm1_alphas_cumprod` — recover predicted x₀ from ε
- `posterior_mean_coef1` / `posterior_mean_coef2` — posterior mean from clipped x₀ (Improved DDPM eq. 9)
- `posterior_variance` — `β̃_t` for sampling variance

**x₀ clipping in `p_sample` and `ddim_sample`:** Before computing the posterior mean, the predicted x₀ is recovered from ε and clipped to `[-1, 1]`. This prevents numerical explosion when the model's noise prediction is imperfect — essential for the cosine schedule where β at high t can exceed 0.8.

### DDIM sampling

DDIM (Song et al. 2021) reuses the same trained DDPM model but samples with a deterministic non-Markovian process on a subsequence of timesteps. The update rule (Eq. 12):

```
x_{prev} = √ᾱ_{prev} · x̂₀  +  √(1 − ᾱ_{prev} − σ²) · ε_θ  +  σ · z
```

where `x̂₀` is recovered from the predicted noise and clipped to [-1, 1], and:

```
σ = η · √((1−ᾱ_{prev})/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_{prev})
```

- `η = 0` → fully deterministic DDIM (same noise → same image every time)
- `η = 1` → recovers DDPM stochasticity
- `α_t` and `α_{prev}` are loaded directly from `self.alphas_cumprod` for arbitrary non-consecutive timesteps
- The `ddim_sample_loop` generates a linearly-spaced subsequence of `ddim_steps` timesteps, then iterates through them calling `ddim_sample` for each pair
- Final step uses `prev_t=0` (ṱ_0 = α_0) for the cleanest x₀ estimate

### Checkpoint format

```python
{
    "model_type": "unet" | "dit",     # model architecture
    "model": model.state_dict(),
    "ema": {"shadow": {...}, "decay": 0.9999},
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict() or None,
    "epoch": int,
    "step": int,
}
```

### Output directory layout

```
outputs/
└── <YYYYMMDD_HHMMSS>/
    ├── samples/
    │   ├── sample_0000001.png   # after first step
    │   ├── sample_0000500.png   # every 500 steps
    │   └── ...
    └── checkpoints/
        ├── checkpoint_epoch_0010.pt
        └── final.pt
```

### Where to extend

- **Improved DDPM (cosine schedule + learned variance):** already using cosine schedule by default; swap UNet `out_channels` from 3 to 6 (mean + variance per channel) and add a learned-variance loss term.
- **DDIM accelerated sampling:** ✅ already implemented (`ddim_sample` / `ddim_sample_loop`). Use `--sampler ddim --ddim-steps 50` at the CLI.
- **Larger resolution / Latent Diffusion:** increase `image_size` and adjust `channel_multipliers` for more down-sample stages. For 64×64 use multipliers `[1, 1, 2, 2, 2]`.
- **DiT (Diffusion Transformer):** ✅ already implemented. Use `--model dit` at the CLI or `dit_s_config()` / `dit_b_config()` in code. See DiT architecture section above.
- **Conditional generation:** add class embedding to the UNet or DiT (similar to time embedding) and train with labels.
