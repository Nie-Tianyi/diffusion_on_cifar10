"""Standalone image generation from a trained checkpoint.

Usage
-----
  # DDIM deterministic, 50 steps (fast, good quality)
  uv run python generate.py --checkpoint outputs/<run_id>/checkpoints/final.pt

  # DDIM 100 steps with slight stochasticity
  uv run python generate.py --checkpoint outputs/<run_id>/checkpoints/final.pt --ddim-steps 100 --ddim-eta 0.2

  # DDPM full 1000-step ancestral sampling
  uv run python generate.py --checkpoint outputs/<run_id>/checkpoints/final.pt --sampler ddpm

  # Generate a specific number of images
  uv run python generate.py --checkpoint outputs/<run_id>/checkpoints/final.pt --n-images 16
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torchvision.utils import save_image

from config import Config, cifar10_config, dit_s_config, dit_b_config
from unet import UNet
from dit import DiT
from diffusion import GaussianDiffusion
from main import EMA  # EMA lives in main.py


def generate(
    checkpoint_path: str,
    sampler: str = "ddim",
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    n_images: int = 64,
    seed: int | None = None,
    output: str | None = None,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint (read model_type before creating model) ──
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "unet")  # backward compat: old checkpoints are UNet
    print(f"Checkpoint model_type: {model_type}")

    # Choose config matching the checkpoint's architecture
    if model_type == "dit":
        # Try to infer DiT size from state dict; default to DiT-S
        cfg = dit_s_config()
    else:
        cfg = cifar10_config()

    # ── Model ──
    if model_type == "dit":
        model = DiT(cfg.model).to(device)
    else:
        model = UNet(cfg.model).to(device)
    ema = EMA(model, cfg.training.ema_decay)

    # ── Load weights ──
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  epoch: {ckpt['epoch']}, step: {ckpt['step']}")

    # ── Diffusion ──
    diffusion = GaussianDiffusion(
        timesteps=cfg.training.timesteps,
        beta_start=cfg.training.beta_start,
        beta_end=cfg.training.beta_end,
        schedule=cfg.training.schedule,
    )

    # ── Sample ──
    if seed is not None:
        torch.manual_seed(seed)

    shape = (n_images, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
    noise = torch.randn(shape)

    ema.apply_shadow()
    model.eval()

    with torch.no_grad():
        if sampler == "ddim":
            print(f"Sampling: DDIM  |  steps={ddim_steps}  |  eta={ddim_eta}  |  images={n_images}")
            samples = diffusion.ddim_sample_loop(
                model, shape, device,
                ddim_steps=ddim_steps, eta=ddim_eta,
                noise=noise, progress=True,
            )
        else:
            print(f"Sampling: DDPM  |  steps={cfg.training.timesteps}  |  images={n_images}")
            samples = diffusion.p_sample_loop(
                model, shape, device,
                noise=noise, progress=True,
            )

    model.train()
    ema.restore()

    # ── Rescale & save ──
    samples = (samples + 1.0) * 0.5
    samples = samples.clamp(0.0, 1.0)

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use checkpoint parent dir if it looks like an outputs/ run
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if "checkpoints" in os.path.basename(ckpt_dir):
            out_dir = os.path.join(os.path.dirname(ckpt_dir), "generated")
        else:
            out_dir = os.path.join(os.getcwd(), "generated")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{sampler}_{ddim_steps}steps_{ts}.png")

    nrow = int(n_images ** 0.5)
    save_image(samples, output, nrow=nrow)
    print(f"Saved → {output}")
    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images from a trained DDPM checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--ddim-steps", type=int, default=50, help="DDIM steps (default: 50)")
    p.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity (default: 0)")
    p.add_argument("--n-images", type=int, default=64, help="Number of images to generate")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--output", type=str, default=None, help="Output path (auto-generated if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(
        checkpoint_path=args.checkpoint,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        n_images=args.n_images,
        seed=args.seed,
        output=args.output,
    )
