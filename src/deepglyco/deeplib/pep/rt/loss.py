__all__ = ["RTMSELoss"]

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import mse_loss, l1_loss

from .data import PeptideRTDataBatch


class RTMSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, batch: PeptideRTDataBatch, pred: torch.Tensor) -> torch.Tensor:
        return mse_loss(pred, batch.retention_time, reduction=self.reduction)


class RTMAELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, batch: PeptideRTDataBatch, pred: torch.Tensor) -> torch.Tensor:
        return l1_loss(pred, batch.retention_time, reduction=self.reduction)

