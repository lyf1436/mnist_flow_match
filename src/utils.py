from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    # data
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    # model
    base_channels: int = 32
    time_embed_dim: int = 256
    # training
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    # checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 10
    resume_from: str = None
    # visualization
    viz_every: int = 5
    viz_n_samples: int = 64
    viz_n_pca_ref: int = 1000
    runs_dir: str = "./runs"
    # misc
    seed: int = 42
    device: str = "cuda"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    path: str,
    cfg: TrainConfig = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "cfg": cfg.__dict__ if cfg else None,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer,
    scheduler,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    ema_model.load_state_dict(ckpt["ema_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"]


def make_ema(model: nn.Module) -> nn.Module:
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
