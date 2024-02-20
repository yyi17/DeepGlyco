__all__ = ["SpectralAngleLoss"]

import torch
from torch.nn.modules.loss import _Loss

from ...util.math import spectral_angle
from .data import PeptideMS2DataBatch


class SpectralAngleLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, batch: PeptideMS2DataBatch, pred: torch.Tensor) -> torch.Tensor:
        angle = spectral_angle(pred, batch.fragment_intensity)

        if self.reduction == "mean":
            return angle.mean()
        elif self.reduction == "sum":
            return angle.sum()
        else:
            return angle
