"""Hyperparameter configuration for DDPM training on CIFAR-10.

Target: RTX 5080 16GB — generous headroom for 32×32 image generation.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """UNet architecture hyperparameters."""
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    attention_resolutions: tuple[int, ...] = (16,)
    num_res_blocks: int = 2
    dropout: float = 0.1
    num_heads: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters — tuned for RTX 5080 16 GB."""
    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Optimisation
    batch_size: int = 256
    lr: float = 2e-4
    epochs: int = 500
    ema_decay: float = 0.9999
    use_amp: bool = True          # BF16 mixed-precision on RTX 5080
    grad_accum_steps: int = 1     # no accumulation needed at 32×32

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Logging & checkpointing
    log_interval: int = 100       # print loss every N steps
    sample_interval: int = 500    # save sample grid every N steps
    save_interval: int = 10       # save checkpoint every N epochs
    n_sample_images: int = 64     # generate 8×8 grid

    # Output
    output_dir: str = "./outputs"


@dataclass
class Config:
    """Aggregate config."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def cifar10_config() -> Config:
    """Return the default CIFAR-10 config tuned for RTX 5080 16 GB.

    Quick-reference numbers
    -----------------------
    - Model  : ~35 M parameters
    - VRAM   : ~4-6 GB peak (well within 16 GB)
    - Speed  : ~0.3–0.5 s/step → ~6-10 h for 500 epochs
    """
    return Config()
