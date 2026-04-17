import torch
import torch.nn as nn
from torch import Tensor
from typing import Union


@torch.no_grad()
def euler_solve(
    model: nn.Module,
    x0: Tensor,
    n_steps: int = 100,
    device: torch.device = None,
) -> Tensor:
    """Euler integration of dx/dt = v_theta(x_t, t) from t=0 to t=1."""
    model.eval()
    x = x0.clone()
    if device is not None:
        x = x.to(device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x = x + dt * v
    return x


@torch.no_grad()
def rk4_solve(
    model: nn.Module,
    x0: Tensor,
    n_steps: int = 50,
    device: torch.device = None,
) -> Tensor:
    """4th-order Runge-Kutta integration from t=0 to t=1."""
    model.eval()
    x = x0.clone()
    if device is not None:
        x = x.to(device)
    dt = 1.0 / n_steps

    def v(xt, t_val):
        t = torch.full((xt.shape[0],), t_val, device=xt.device, dtype=xt.dtype)
        return model(xt, t)

    for i in range(n_steps):
        t_val = i * dt
        k1 = v(x, t_val)
        k2 = v(x + 0.5 * dt * k1, t_val + 0.5 * dt)
        k3 = v(x + 0.5 * dt * k2, t_val + 0.5 * dt)
        k4 = v(x + dt * k3, t_val + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def sample_trajectory(
    model: nn.Module,
    x0: Tensor,
    n_steps: int = 20,
    solver: str = "euler",
    return_all: bool = False,
    device: torch.device = None,
) -> Union[Tensor, list]:
    """
    Integrate ODE, optionally collecting all intermediate states.
    Returns final state, or list of states at each step if return_all=True.
    """
    model.eval()
    x = x0.clone()
    if device is not None:
        x = x.to(device)
    dt = 1.0 / n_steps
    states = [x.clone()] if return_all else None

    def v(xt, t_val):
        t = torch.full((xt.shape[0],), t_val, device=xt.device, dtype=xt.dtype)
        return model(xt, t)

    for i in range(n_steps):
        t_val = i * dt
        if solver == "rk4":
            k1 = v(x, t_val)
            k2 = v(x + 0.5 * dt * k1, t_val + 0.5 * dt)
            k3 = v(x + 0.5 * dt * k2, t_val + 0.5 * dt)
            k4 = v(x + dt * k3, t_val + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            x = x + dt * v(x, t_val)
        if return_all:
            states.append(x.clone())

    return states if return_all else x
