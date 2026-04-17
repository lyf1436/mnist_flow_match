"""
Inference script for MNIST flow matching.

Usage:
    python inference.py --checkpoint ./checkpoints/ckpt_0099.pt
    python inference.py --checkpoint ./checkpoints/ckpt_0099.pt \\
        --n_samples 64 --solver rk4 --n_steps 100 \\
        --output ./outputs/samples.png --save_trajectory
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import UNet
from src.ode import euler_solve, rk4_solve, sample_trajectory
from src.viz import tensor_to_flat_np, samples_to_grid_np


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("cfg") or {}
    base_ch = cfg_dict.get("base_channels", 32)
    te_dim = cfg_dict.get("time_embed_dim", 256)
    model = UNet(base_ch=base_ch, time_embed_dim=te_dim).to(device)
    key = "ema_state" if use_ema and "ema_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[key])
    model.eval()
    print(f"Loaded {'EMA ' if use_ema else ''}model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def generate_samples(
    model: UNet,
    n_samples: int = 64,
    solver: str = "rk4",
    n_steps: int = 100,
    seed: int = 42,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(seed)
    x0 = torch.randn(n_samples, 1, 28, 28, device=device)
    if solver == "rk4":
        samples = rk4_solve(model, x0, n_steps=n_steps, device=device)
    else:
        samples = euler_solve(model, x0, n_steps=n_steps, device=device)
    return samples.clamp(-1, 1)


def save_sample_grid(
    samples: torch.Tensor,
    path: str,
    nrow: int = 8,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Denormalize [-1,1] -> [0,1]
    imgs = (samples.clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, path, nrow=nrow, padding=2, normalize=False)
    print(f"Saved sample grid to {path}")


def save_trajectory_gif(
    model: UNet,
    n_samples: int = 16,
    n_steps: int = 20,
    seed: int = 42,
    output_dir: str = "./outputs",
    device: torch.device = None,
) -> None:
    """Save each step of the ODE trajectory as a frame, assemble into a GIF."""
    import imageio.v2 as iio
    import numpy as np

    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(seed)
    x0 = torch.randn(n_samples, 1, 28, 28, device=device)
    states = sample_trajectory(model, x0, n_steps=n_steps, return_all=True, device=device)

    frames = []
    for state in states:
        grid = samples_to_grid_np(state[:16], nrow=4)
        frame_uint8 = (grid.squeeze() * 255).clip(0, 255).astype("uint8")
        # Convert grayscale to RGB for imageio
        frame_rgb = np.stack([frame_uint8] * 3, axis=-1)
        frames.append(frame_rgb)

    gif_path = str(Path(output_dir) / "trajectory.gif")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    iio.mimwrite(gif_path, frames, fps=8, loop=0)
    print(f"Trajectory GIF saved to {gif_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST Flow Matching Inference")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--solver", type=str, default="rk4", choices=["euler", "rk4"])
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--output", type=str, default="./outputs/samples.png")
    p.add_argument("--save_trajectory", action="store_true", help="Also save trajectory GIF")
    p.add_argument("--use_raw_model", action="store_true", help="Use raw weights instead of EMA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(
        args.checkpoint, device, use_ema=not args.use_raw_model
    )

    print(f"Generating {args.n_samples} samples with {args.solver} ({args.n_steps} steps)...")
    samples = generate_samples(
        model, n_samples=args.n_samples, solver=args.solver,
        n_steps=args.n_steps, seed=args.seed, device=device,
    )
    save_sample_grid(samples, args.output)

    if args.save_trajectory:
        output_dir = str(Path(args.output).parent)
        save_trajectory_gif(model, n_samples=16, n_steps=20, seed=args.seed,
                            output_dir=output_dir, device=device)


if __name__ == "__main__":
    main()
