from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import Tensor


def get_mnist_dataloader(
    root: str = "./data",
    batch_size: int = 256,
    train: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_reference_batch(
    dataloader: DataLoader,
    n: int = 1000,
    device: torch.device = None,
) -> Tensor:
    collected = []
    total = 0
    for x, _ in dataloader:
        needed = n - total
        collected.append(x[:needed])
        total += x[:needed].shape[0]
        if total >= n:
            break
    out = torch.cat(collected, dim=0)[:n]
    if device is not None:
        out = out.to(device)
    return out
