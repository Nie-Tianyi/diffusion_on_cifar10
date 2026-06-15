"""Sampling utilities — generate and save image grids during training."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from diffusion import GaussianDiffusion


@torch.no_grad()
def sample_and_save(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    step: int,
    save_dir: str,
    device: torch.device,
    fixed_noise: torch.Tensor,
    ema_model: nn.Module | None = None,
) -> None:
    """Generate a grid of samples from fixed noise and save as PNG.

    Uses the same fixed noise at every call so you can watch the same
    "seeds" evolve as training progresses.
    """
    # Use EMA weights for inference when available
    m = ema_model if ema_model is not None else model
    m.eval()

    samples = diffusion.p_sample_loop(
        m,
        shape=fixed_noise.shape,
        device=device,
        noise=fixed_noise,
    )

    m.train()

    # Rescale from [-1, 1] → [0, 1]
    samples = (samples + 1.0) * 0.5
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=int(fixed_noise.shape[0] ** 0.5))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_{step:07d}.png")
    save_image(grid, save_path)
