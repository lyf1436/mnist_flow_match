import torch
import torch.nn as nn
from torch import Tensor


def sample_t(batch_size: int, device: torch.device) -> Tensor:
    t = torch.rand(batch_size, device=device)
    # avoid exact 0 and 1 for numerical stability
    return t * (1 - 2e-4) + 1e-4


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """x_t = (1-t)*x0 + t*x1, t broadcast over spatial dims."""
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x0 + t * x1


def target_vector_field(x0: Tensor, x1: Tensor) -> Tensor:
    """Straight-path CFM target: constant u = x1 - x0."""
    return x1 - x0


def flow_matching_loss(
    model: nn.Module,
    x1: Tensor,
    device: torch.device,
) -> Tensor:
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_t(B, device)
    x_t = interpolate(x0, x1, t)
    target = target_vector_field(x0, x1)
    pred = model(x_t, t)
    return ((pred - target) ** 2).mean()
