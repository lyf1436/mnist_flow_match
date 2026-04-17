from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

class PCAProjector:
    """Fits PCA once on reference real data; reused every frame."""

    def __init__(self, n_components: int = 2):
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, x: np.ndarray) -> "PCAProjector":
        # x: (N, 784), already flattened and normalized
        self.pca.fit(x)
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(x)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(z)

    @property
    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_


def tensor_to_flat_np(x: Tensor) -> np.ndarray:
    """(N,1,28,28) tensor -> (N,784) numpy array on CPU."""
    return x.detach().cpu().float().view(x.shape[0], -1).numpy()


def fit_umap(
    x: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """UMAP embedding (N,784) -> (N,2). Slow; only for final summary."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        return reducer.fit_transform(x)
    except ImportError:
        print("umap-learn not installed, falling back to PCA for final summary.")
        pca = PCA(n_components=2)
        return pca.fit_transform(x)


# ---------------------------------------------------------------------------
# Vector Field in 2D PCA Space
# ---------------------------------------------------------------------------

def compute_2d_vector_field(
    model: nn.Module,
    projector: PCAProjector,
    grid_size: int = 15,
    t_val: float = 0.5,
    device: torch.device = None,
):
    """
    Build a quiver plot of v_theta in the 2D PCA subspace.
    Returns X_grid, Y_grid, U, V arrays for matplotlib quiver.
    """
    if device is None:
        device = next(model.parameters()).device

    # Get extent from PCA training data mean ± 3*std
    mean = projector.pca.mean_
    components = projector.pca.components_  # (2, 784)
    proj_std = np.sqrt(projector.pca.explained_variance_)  # (2,)

    r0 = 3.5 * proj_std[0]
    r1 = 3.5 * proj_std[1]
    ax0 = np.linspace(-r0, r0, grid_size)
    ax1 = np.linspace(-r1, r1, grid_size)
    Z0, Z1 = np.meshgrid(ax0, ax1)  # each (grid_size, grid_size)
    Z_flat = np.stack([Z0.ravel(), Z1.ravel()], axis=1)  # (G*G, 2)

    # Lift to 784D
    x_784 = projector.inverse_transform(Z_flat)  # (G*G, 784)
    x_tensor = torch.tensor(x_784, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    t_tensor = torch.full((x_tensor.shape[0],), t_val, device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        v_field = model(x_tensor, t_tensor)  # (G*G, 1, 28, 28)

    v_flat = v_field.detach().cpu().float().view(-1, 784).numpy()
    # Project vector field directions into PCA space
    v_2d = v_flat @ components.T  # (G*G, 2)

    U = v_2d[:, 0].reshape(grid_size, grid_size)
    V = v_2d[:, 1].reshape(grid_size, grid_size)
    return Z0, Z1, U, V


# ---------------------------------------------------------------------------
# Sample Grid Helper
# ---------------------------------------------------------------------------

def samples_to_grid_np(samples: Tensor, nrow: int = 8) -> np.ndarray:
    """
    Convert (N,1,28,28) tensor in [-1,1] to a single HWC numpy image in [0,1].
    """
    imgs = samples.detach().cpu().float().clamp(-1, 1)
    imgs = (imgs + 1) / 2  # -> [0,1]
    n = imgs.shape[0]
    ncol = (n + nrow - 1) // nrow
    # pad to nrow*ncol
    pad = nrow * ncol - n
    if pad > 0:
        imgs = torch.cat([imgs, torch.zeros(pad, 1, 28, 28)], dim=0)
    grid = imgs.view(ncol, nrow, 1, 28, 28)
    grid = grid.permute(0, 3, 1, 4, 2)  # (ncol, 28, nrow, 28, 1)
    grid = grid.reshape(ncol * 28, nrow * 28, 1)
    return grid.numpy()


# ---------------------------------------------------------------------------
# Frame Rendering
# ---------------------------------------------------------------------------

def render_frame(
    epoch: int,
    generated_samples: Tensor,
    real_np: np.ndarray,
    noise_np: np.ndarray,
    traj_np: np.ndarray,
    projector: PCAProjector,
    model: nn.Module,
    device: torch.device,
    save_path: str,
    grid_size: int = 15,
) -> None:
    """
    3-panel figure:
      Left  : 8x8 grid of generated samples
      Center: PCA scatter (real=orange, noise=blue, mid-traj=green)
      Right : Vector field quiver in PCA space at t=0.5
    """
    # Project points
    real_2d = projector.transform(real_np)
    noise_2d = projector.transform(noise_np)
    traj_2d = projector.transform(traj_np)

    # Vector field
    Z0, Z1, U, V = compute_2d_vector_field(model, projector, grid_size=grid_size, device=device)

    # Sample grid image
    grid_img = samples_to_grid_np(generated_samples[:64], nrow=8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Epoch {epoch}", fontsize=14)

    # Panel 1: generated samples
    axes[0].imshow(grid_img.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].axis("off")
    axes[0].set_title("Generated Samples")

    # Panel 2: PCA scatter
    ax = axes[1]
    ax.scatter(noise_2d[:, 0], noise_2d[:, 1], s=4, alpha=0.3, c="steelblue", label="Noise")
    ax.scatter(real_2d[:, 0], real_2d[:, 1], s=4, alpha=0.4, c="darkorange", label="Real data")
    ax.scatter(traj_2d[:, 0], traj_2d[:, 1], s=4, alpha=0.5, c="green", label="Mid-traj (t=0.5)")
    ax.set_title("PCA Scatter")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Panel 3: vector field quiver
    ax2 = axes[2]
    ax2.scatter(real_2d[:, 0], real_2d[:, 1], s=2, alpha=0.15, c="darkorange")
    magnitude = np.sqrt(U ** 2 + V ** 2) + 1e-8
    ax2.quiver(Z0, Z1, U / magnitude, V / magnitude, magnitude,
               cmap="plasma", alpha=0.85, scale=25, width=0.004)
    ax2.set_title("Vector Field (t=0.5)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video Assembly
# ---------------------------------------------------------------------------

def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: int = 8,
    pattern: str = "frame_*.png",
) -> None:
    """Sort PNG frames by epoch number and assemble into MP4."""
    import imageio.v2 as iio
    frame_paths = sorted(Path(frame_dir).glob(pattern),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if not frame_paths:
        print(f"No frames found in {frame_dir} matching {pattern}")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(output_path, fps=fps, format="FFMPEG",
                            codec="libx264", quality=8)
    for fp in frame_paths:
        frame = iio.imread(str(fp))
        writer.append_data(frame)
    writer.close()
    print(f"Video saved to {output_path} ({len(frame_paths)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Final Summary Figure
# ---------------------------------------------------------------------------

def make_final_summary(
    model: nn.Module,
    real_np: np.ndarray,
    device: torch.device,
    output_path: str,
    n_traj: int = 10,
    traj_steps: int = 20,
    use_umap: bool = True,
) -> None:
    """
    4-panel high-quality summary:
      1. Embedding scatter (UMAP or PCA): real / noise / generated
      2. Generated samples grid
      3. Trajectory streamlines in 2D
      4. Vector field quiver
    """
    from .ode import sample_trajectory, euler_solve

    model.eval()

    # --- Generate samples ---
    n_gen = 64
    x0_gen = torch.randn(n_gen, 1, 28, 28, device=device)
    gen_samples = euler_solve(model, x0_gen, n_steps=100, device=device)
    gen_np = tensor_to_flat_np(gen_samples)

    # --- Noise reference ---
    noise_np = np.random.randn(len(real_np), 784).astype(np.float32)

    # --- Embed with UMAP or PCA ---
    all_points = np.concatenate([real_np, gen_np, noise_np[:len(real_np)]], axis=0)
    n_real, n_gen_pts = len(real_np), len(gen_np)

    print("Computing 2D embedding for final summary (may take ~10s for UMAP)...")
    if use_umap:
        emb = fit_umap(all_points)
    else:
        projector_final = PCAProjector()
        projector_final.fit(all_points)
        emb = projector_final.transform(all_points)

    real_emb = emb[:n_real]
    gen_emb = emb[n_real:n_real + n_gen_pts]
    noise_emb = emb[n_real + n_gen_pts:]

    # --- Trajectories ---
    x0_traj = torch.randn(n_traj, 1, 28, 28, device=device)
    traj_states = sample_trajectory(model, x0_traj, n_steps=traj_steps,
                                    return_all=True, device=device)

    # PCA for trajectory projection (fast)
    pca_traj = PCA(n_components=2)
    pca_traj.fit(real_np)
    traj_lines = [pca_traj.transform(tensor_to_flat_np(s)) for s in traj_states]

    # PCA projector for vector field
    projector_vf = PCAProjector()
    projector_vf.fit(real_np)
    Z0, Z1, U, V = compute_2d_vector_field(model, projector_vf, grid_size=12, device=device)

    # --- Plot ---
    grid_img = samples_to_grid_np(gen_samples[:64], nrow=8)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Flow Matching – Final Summary", fontsize=16)

    # 1. Embedding scatter
    ax = axes[0]
    ax.scatter(noise_emb[:, 0], noise_emb[:, 1], s=3, alpha=0.25, c="steelblue", label="Noise")
    ax.scatter(real_emb[:, 0], real_emb[:, 1], s=3, alpha=0.35, c="darkorange", label="Real")
    ax.scatter(gen_emb[:, 0], gen_emb[:, 1], s=5, alpha=0.7, c="green", label="Generated")
    ax.legend(markerscale=3, fontsize=8)
    method = "UMAP" if use_umap else "PCA"
    ax.set_title(f"{method} Embedding")

    # 2. Generated samples grid
    axes[1].imshow(grid_img.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].set_title("Generated Samples")

    # 3. Trajectory streamlines (in PCA space)
    ax3 = axes[2]
    for i in range(n_traj):
        xs = [traj_lines[step][i, 0] for step in range(len(traj_lines))]
        ys = [traj_lines[step][i, 1] for step in range(len(traj_lines))]
        ax3.plot(xs, ys, "-o", markersize=2, linewidth=1, alpha=0.7)
        ax3.plot(xs[0], ys[0], "o", color="steelblue", markersize=5)
        ax3.plot(xs[-1], ys[-1], "*", color="red", markersize=6)
    ax3.set_title("Flow Trajectories (PCA)")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    # 4. Vector field
    ax4 = axes[3]
    real_2d_vf = projector_vf.transform(real_np)
    ax4.scatter(real_2d_vf[:, 0], real_2d_vf[:, 1], s=2, alpha=0.15, c="darkorange")
    magnitude = np.sqrt(U ** 2 + V ** 2) + 1e-8
    ax4.quiver(Z0, Z1, U / magnitude, V / magnitude, magnitude,
               cmap="plasma", alpha=0.85, scale=20, width=0.005)
    ax4.set_title("Vector Field (t=0.5)")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Final summary saved to {output_path}")
