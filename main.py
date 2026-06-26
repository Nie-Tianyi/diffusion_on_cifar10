"""DDPM training script for CIFAR-10.

Usage
-----
  python main.py                     # default config (RTX 5080 16 GB)
  python main.py --epochs 200        # shorter run
  python main.py --batch-size 128    # smaller batch
  python main.py --resume ./outputs/checkpoint_epoch_50.pt
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # type: ignore
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from config import Config, TrainingConfig, cifar10_config
from unet import UNet
from diffusion import GaussianDiffusion
from sampling import sample_and_save


def _create_model(config: "Config") -> "nn.Module":
    """Factory: return the right noise-prediction model based on config.model_type."""
    mc = config.model
    if config.model_type == "dit":
        from dit import DiT
        return DiT(mc)
    return UNet(mc)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Maintains an exponential moving average of model parameters.

    Inference is done with the EMA shadow weights, which gives noticeably
    better sample quality for essentially zero cost.

    Includes a warmup schedule: early in training the effective decay is
    lower so the shadow model can catch up to the current weights quickly,
    avoiding catastrophic "random weight residue" that ruins early samples.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._registered = False
        self._step = 0

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._registered = True

    def update(self) -> None:
        if not self._registered:
            self._register()

        self._step += 1
        # Warmup: ramp decay from ~0.09 (step 1) → 0.9999 over time.
        # After 7800 steps the residue of initial random weights is <1%
        # instead of ~46% with a fixed 0.9999 decay.
        current_decay = min(self.decay, (1 + self._step) / (10 + self._step))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = current_decay * self.shadow[name] + (1.0 - current_decay) * param.data

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].to(param.device)
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay, "step": self._step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
        self._step = state_dict.get("step", 0)
        self._registered = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataloader(config: TrainingConfig, train: bool = True) -> DataLoader:
    transform_list = [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if train:
        transform_list.insert(0, T.RandomHorizontalFlip())
    transform = T.Compose(transform_list)

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    epoch: int,
    step: int,
    model_type: str = "unet",
):
    checkpoint = {
        "model_type": model_type,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, resume: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    mc, tc = cfg.model, cfg.training

    # ── Model ──
    model = _create_model(cfg).to(device)
    ema = EMA(model, tc.ema_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f} M")

    # ── Diffusion ──
    diffusion = GaussianDiffusion(
        timesteps=tc.timesteps,
        beta_start=tc.beta_start,
        beta_end=tc.beta_end,
        schedule=tc.schedule,
    )

    # ── Optimiser ──
    optimizer = optim.Adam(model.parameters(), lr=tc.lr)

    # ── AMP dtype & scaler ──
    # BF16 has enough dynamic range that GradScaler is unnecessary (and can
    # occasionally cause skipped updates). Only use scaler for FP16.
    if tc.use_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16
        scaler = GradScaler("cuda") if amp_dtype == torch.float16 else None
    else:
        amp_dtype = None
        scaler = None

    # ── Data ──
    dataloader = make_dataloader(tc, train=True)

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    if resume is not None:
        start_epoch, global_step = load_checkpoint(
            resume, model, ema, optimizer, scaler
        )
        start_epoch += 1  # resume from next epoch
        print(f"Resumed from {resume} → epoch {start_epoch}, step {global_step}")

    # ── Fixed noise for progress tracking ──
    fixed_noise = torch.randn(tc.n_sample_images, mc.in_channels, mc.image_size, mc.image_size)

    # ── Output directories ──
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = os.path.join(tc.output_dir, run_id, "samples")
    ckpt_dir = os.path.join(tc.output_dir, run_id, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Samples → {sample_dir}")
    print(f"Checkpoints → {ckpt_dir}")

    # ── Training ──
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, tc.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{tc.epochs}", leave=False)

        epoch_loss = 0.0
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            # Sample random timesteps + noise
            t = torch.randint(0, tc.timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)

            # Forward
            with autocast(device.type, dtype=amp_dtype) if tc.use_amp else _null_context():
                loss = diffusion.p_losses(model, images, t, noise)
                loss = loss / tc.grad_accum_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % tc.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            ema.update()
            global_step += 1
            epoch_loss += loss.item() * tc.grad_accum_steps

            # Logging
            if global_step % tc.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item() * tc.grad_accum_steps:.4f}")

            # Sampling
            if global_step % tc.sample_interval == 0 or global_step == 1:
                ema.apply_shadow()
                sample_and_save(
                    model, diffusion, global_step, sample_dir, device, fixed_noise,
                    sampler=tc.sampler,
                    ddim_steps=tc.ddim_steps,
                    ddim_eta=tc.ddim_eta,
                )
                ema.restore()

        # ── End of epoch ──
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1:3d}/{tc.epochs} | avg loss: {avg_loss:.6f}")

        if (epoch + 1) % tc.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1:04d}.pt")
            save_checkpoint(ckpt_path, model, ema, optimizer, scaler, epoch, global_step, cfg.model_type)
            print(f"  → saved {ckpt_path}")

    # ── Final checkpoint ──
    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(final_path, model, ema, optimizer, scaler, tc.epochs - 1, global_step, cfg.model_type)
    print(f"Done! Final checkpoint → {final_path}")

    # ── Final sample with EMA ──
    ema.apply_shadow()
    sample_and_save(
        model, diffusion, global_step + 1, sample_dir, device, fixed_noise,
        sampler=tc.sampler,
        ddim_steps=tc.ddim_steps,
        ddim_eta=tc.ddim_eta,
    )
    ema.restore()
    print(f"Final samples saved in {sample_dir}")


class _null_context:
    """Minimal no-op context manager for when AMP is disabled."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM training on CIFAR-10")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"], help="Sampling method")
    p.add_argument("--ddim-steps", type=int, default=None, help="DDIM sampling steps")
    p.add_argument("--ddim-eta", type=float, default=None, help="DDIM stochasticity (0=deterministic)")
    p.add_argument("--model", type=str, default=None, choices=["unet", "dit"], help="Model architecture")
    args = p.parse_args()
    return args


def main():
    args = _parse_args()
    cfg = cifar10_config()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.no_amp:
        cfg.training.use_amp = False
    if args.sampler is not None:
        cfg.training.sampler = args.sampler
    if args.ddim_steps is not None:
        cfg.training.ddim_steps = args.ddim_steps
    if args.ddim_eta is not None:
        cfg.training.ddim_eta = args.ddim_eta
    if args.model is not None:
        cfg.model_type = args.model

    # Dump config
    print("=" * 50)
    print(f"Model type: {cfg.model_type}")
    print("=" * 50)
    print("Training config")
    print("=" * 50)
    for field_name in vars(cfg.training):
        print(f"  {field_name}: {getattr(cfg.training, field_name)}")
    print("Model config")
    for field_name in vars(cfg.model):
        print(f"  {field_name}: {getattr(cfg.model, field_name)}")
    print("=" * 50)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
