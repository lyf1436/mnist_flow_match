"""
Training script for MNIST flow matching.

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 256 --viz_every 5
    python train.py --resume_from ./checkpoints/ckpt_0049.pt
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.utils import TrainConfig, set_seed, get_device, save_checkpoint, load_checkpoint, make_ema, update_ema
from src.data import get_mnist_dataloader, get_reference_batch
from src.model import UNet
from src.flow import flow_matching_loss
from src.ode import euler_solve, sample_trajectory
from src.viz import PCAProjector, tensor_to_flat_np, render_frame, frames_to_video, make_final_summary


def build_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    warmup_steps = cfg.lr_warmup_steps
    cos_steps = max(total_steps - warmup_steps, 1)

    warmup = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=cos_steps, eta_min=1e-5)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg)
    print(f"Training on {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.runs_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader = get_mnist_dataloader(cfg.data_root, cfg.batch_size, train=True, num_workers=cfg.num_workers)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs

    # Model + EMA
    model = UNet(cfg.base_channels, cfg.time_embed_dim).to(device)
    ema_model = make_ema(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # Optional resume
    start_epoch = 0
    if cfg.resume_from:
        start_epoch = load_checkpoint(cfg.resume_from, model, ema_model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # PCA reference (fit once)
    print("Fitting PCA on reference data...")
    ref_batch = get_reference_batch(train_loader, cfg.viz_n_pca_ref)
    ref_np = tensor_to_flat_np(ref_batch)
    projector = PCAProjector()
    projector.fit(ref_np)
    var = projector.explained_variance_ratio
    print(f"PCA explained variance: PC1={var[0]:.3f}, PC2={var[1]:.3f}")

    # Fixed noise for visualization consistency
    noise_ref = torch.randn(min(200, cfg.viz_n_pca_ref), 1, 28, 28)
    noise_ref_np = tensor_to_flat_np(noise_ref)

    # Training loop
    loss_history = []
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for x1, _ in pbar:
            x1 = x1.to(device)
            loss = flow_matching_loss(model, x1, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, cfg.ema_decay)
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Visualization frame
        if epoch % cfg.viz_every == 0 or epoch == cfg.epochs - 1:
            ema_model.eval()
            with torch.no_grad():
                x0_viz = torch.randn(cfg.viz_n_samples, 1, 28, 28, device=device)
                generated = euler_solve(ema_model, x0_viz, n_steps=50, device=device)

                x0_traj = torch.randn(200, 1, 28, 28, device=device)
                traj_states = sample_trajectory(ema_model, x0_traj, n_steps=10,
                                                return_all=True, device=device)
                mid_idx = len(traj_states) // 2
                mid_traj_np = tensor_to_flat_np(traj_states[mid_idx])

            render_frame(
                epoch=epoch,
                generated_samples=generated,
                real_np=ref_np[:200],
                noise_np=noise_ref_np,
                traj_np=mid_traj_np,
                projector=projector,
                model=ema_model,
                device=device,
                save_path=f"{cfg.runs_dir}/frame_{epoch:04d}.png",
            )

        # Checkpoint
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch,
                f"{cfg.checkpoint_dir}/ckpt_{epoch:04d}.pt",
                cfg=cfg,
            )

    # Save loss curve
    _save_loss_curve(loss_history, f"{cfg.runs_dir}/loss_curve.png")

    # Assemble video
    print("Assembling training video...")
    frames_to_video(cfg.runs_dir, f"{cfg.runs_dir}/training.mp4", fps=8)

    # Final summary
    print("Generating final summary figure...")
    make_final_summary(
        model=ema_model,
        real_np=ref_np,
        device=device,
        output_path=f"{cfg.runs_dir}/final_summary.png",
        use_umap=True,
    )

    print("Training complete.")


def _save_loss_curve(losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train MNIST Flow Matching")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--base_channels", type=int, default=cfg.base_channels)
    p.add_argument("--viz_every", type=int, default=cfg.viz_every)
    p.add_argument("--checkpoint_every", type=int, default=cfg.checkpoint_every)
    p.add_argument("--checkpoint_dir", type=str, default=cfg.checkpoint_dir)
    p.add_argument("--runs_dir", type=str, default=cfg.runs_dir)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
