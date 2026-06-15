# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DDPM diffusion model training pipeline for CIFAR-10. Implements the Denoising Diffusion Probabilistic Models paper (Ho et al. 2020): a U-Net predicts noise added to images, trained with the simple MSE objective. Target hardware is an RTX 5080 16 GB, but the code auto-scales to any GPU or CPU.

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

# Smoke test — verify model init, forward pass, and sampling all work
uv run python -c "
import torch
from config import cifar10_config
from model import UNet
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
  → UNet predicts noise ε_θ(x_t, t)
  → MSE loss between ε_θ and true ε
  → backprop through UNet only
```

**Sampling (reverse process):** start from pure Gaussian noise x_T, iterate t = T−1 → 0 using the DDPM ancestral sampler: `x_{t-1} = 1/√α_t · (x_t − β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t·z`. Uses EMA shadow weights for better quality.

### Module dependency graph

```
config.py          ← dataclasses: ModelConfig, TrainingConfig, Config
model.py           ← UNet (imports ModelConfig from config)
diffusion.py       ← GaussianDiffusion (standalone, no project imports)
sampling.py        ← sample_and_save (imports GaussianDiffusion)
main.py            ← training loop (imports all above)
```

`diffusion.py` and `model.py` are the two core modules; neither imports the other — the training loop wires them together.

### Key design choices

- **ε-prediction, not x₀-prediction.** The UNet estimates the noise that was added, matching the DDPM "simple" objective. This works better in practice than predicting x₀ directly.
- **BF16 mixed precision via `torch.amp.autocast`.** RTX 5080 (Blackwell) runs BF16 natively. No `GradScaler` needed for BF16 (unlike FP16) — the scaler is included as a harmless safety net.
- **EMA with warmup + decay 0.9999.** During training, an exponential moving average of weights is maintained. Sampling always uses EMA weights; they are applied/restored around each sample step. A dynamic warmup schedule (`current_decay = min(0.9999, (1+step)/(10+step))`) prevents the shadow model from retaining too much random initial weight early in training — without this, early samples are pure-colour blocks.
- **Linear β schedule from 1e-4 to 0.02.** The simplest schedule that works. A `cosine_beta_schedule` is already implemented in `diffusion.py` for the next experiment (Improved DDPM).
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

### Diffusion precomputed coefficients

All stored as 1-D tensors of length `T` on CPU (moved to correct device inside `_extract`). Key tensors:
- `sqrt_alphas_cumprod` — scale x₀ in forward diffusion
- `sqrt_one_minus_alphas_cumprod` — scale ε in forward diffusion
- `sqrt_recip_alphas_cumprod` / `sqrt_recipm1_alphas_cumprod` — recover predicted x₀ from ε
- `posterior_mean_coef1` / `posterior_mean_coef2` — posterior mean from clipped x₀ (Improved DDPM eq. 9)
- `posterior_variance` — `β̃_t` for sampling variance

**x₀ clipping in `p_sample`:** Before computing the posterior mean, the predicted x₀ is recovered from ε and clipped to `[-1, 1]`. This prevents numerical explosion when the model's noise prediction is imperfect — essential for the cosine schedule where β at high t can exceed 0.8.

### Checkpoint format

```python
{
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

- **Improved DDPM (cosine schedule + learned variance):** swap `linear_beta_schedule` for `cosine_beta_schedule` in `diffusion.py.__init__`, and change UNet `out_channels` from 3 to 6 (mean + variance per channel).
- **DDIM accelerated sampling:** add a `ddim_sample_loop` method to `GaussianDiffusion` — only needs ~50 steps instead of 1000.
- **Larger resolution / Latent Diffusion:** increase `image_size` and adjust `channel_multipliers` for more down-sample stages. For 64×64 use multipliers `[1, 1, 2, 2, 2]`.
- **Conditional generation:** add class embedding to the UNet (similar to time embedding) and train with labels.
