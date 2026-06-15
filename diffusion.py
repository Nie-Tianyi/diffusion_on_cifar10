"""Gaussian diffusion process — forward (noising) and reverse (sampling).

Implements the DDPM formulation (Ho et al. 2020):
  - Linear β schedule
  - ε-prediction objective
  - DDPM ancestral sampling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 0.02) -> torch.Tensor:
    """Linear schedule: β_t grows linearly from `start` to `end`."""
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021).

    Produces ᾱ_t following a cosine curve — gives more even noise-level
    coverage than the linear schedule, which improves log-likelihood and
    sample quality by spending more steps in the mid-noise regime where
    the model learns the most about image structure.

    Beta values grow naturally at high t (up to ~0.8 for the final steps).
    This is by design: the corresponding sqrt_recip_alpha and coef_eps
    coefficients compensate, keeping the reverse process mathematically
    consistent.  Clamping betas would break this consistency.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-5, max=0.999)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather values from 1-D tensor `a` at indices `t` and broadcast to `x_shape`."""
    b = t.shape[0]
    out = a[t.cpu()]  # a is on CPU, t may be on CUDA — index on CPU
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)


# ---------------------------------------------------------------------------
# Gaussian Diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """DDPM forward/reverse process with pre-computed coefficients."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        schedule: str = "cosine",
    ):
        self.timesteps = timesteps

        # ── β schedule ──
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # ── Precompute coefficients ──
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Forward process
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

        # Reverse process — x₀ recovery (for clipping)
        self.sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0).sqrt()

        # Reverse process — posterior mean via x₀ (Improved DDPM eq. 9)
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # Loss
        self.loss_type = loss_type

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t given x_0:  x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε."""
        s1 = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s2 = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return s1 * x0 + s2 * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        denoise_fn: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDPM simple loss: MSE between true noise and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)
        predicted_noise = denoise_fn(xt, t)

        if self.loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # Reverse diffusion (sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self, denoise_fn: nn.Module, x: torch.Tensor, t: int, t_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_t → x_{t-1} with x₀ clipping.

        Instead of computing the posterior mean directly from ε, we first
        recover a point estimate of x₀ from the predicted noise, CLIP it to
        [-1, 1], then compute the posterior mean from the clipped x₀.

        This clipping prevents numerical explosion when the model's noise
        prediction is imperfect — critical for the cosine schedule where β
        values at high t can exceed 0.8 (vs ≤0.02 for linear).
        """
        eps = denoise_fn(x, t_tensor)

        # Recover predicted x₀ and clip
        sr_ac = _extract(self.sqrt_recip_alphas_cumprod, t_tensor, x.shape)
        sr_m1_ac = _extract(self.sqrt_recipm1_alphas_cumprod, t_tensor, x.shape)
        pred_x0 = sr_ac * x - sr_m1_ac * eps
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Posterior mean from clipped x₀
        coef1 = _extract(self.posterior_mean_coef1, t_tensor, x.shape)
        coef2 = _extract(self.posterior_mean_coef2, t_tensor, x.shape)
        mean = coef1 * pred_x0 + coef2 * x

        if t == 0:
            return mean

        var = _extract(self.posterior_variance, t_tensor, x.shape)
        noise = torch.randn_like(x)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoise_fn: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        progress: bool = False,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full reverse chain: x_T ∼ N(0,I) → … → x_0."""
        b = shape[0]
        if noise is not None:
            img = noise.to(device)
        else:
            img = torch.randn(shape, device=device)

        timestep_range = reversed(range(self.timesteps))
        if progress:
            from tqdm import tqdm
            timestep_range = tqdm(timestep_range, desc="Sampling", leave=False)

        for t in timestep_range:
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn, img, t, t_tensor)

        return img
