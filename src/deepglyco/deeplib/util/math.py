__all__ = [
    "normalize_by_max",
    "normalize_by_sum",
    "dot_product",
    "spectral_angle",
]

import numpy as np
import torch
import torch.nn.functional as F


def normalize_by_max(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max()
    if x_max > 0:
        x = x / x_max
    return x


def normalize_by_sum(x: torch.Tensor) -> torch.Tensor:
    x_sum = x.sum()
    if x_sum > 0:
        x = x / x_sum
    return x


def dot_product(x: torch.Tensor, y: torch.Tensor, batched=True) -> torch.Tensor:
    if batched:
        x_norm = F.normalize(x.flatten(start_dim=1), p=2)
        y_norm = F.normalize(y.flatten(start_dim=1), p=2)
        return torch.sum(x_norm * y_norm, dim=1)
    else:
        x_norm = F.normalize(x.flatten(), dim=0, p=2)
        y_norm = F.normalize(y.flatten(), dim=0, p=2)
        return torch.sum(x_norm * y_norm)


def spectral_angle(x: torch.Tensor, y: torch.Tensor, batched=True) -> torch.Tensor:
    dot_prod = dot_product(x, y, batched=batched)
    return dot_prod.acos() / np.pi * 2
