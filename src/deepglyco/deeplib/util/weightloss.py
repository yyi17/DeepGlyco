__all__ = ["AutomaticWeightedLoss"]

from typing import Any, Mapping
import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)

    # https://github.com/Mikoto10032/AutomaticWeightedLoss
    # A PyTorch implementation of Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]. arXiv preprint arXiv:1805.06334, 2018.
    # The above paper improves the paper "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" to avoid the loss of becoming negative during training.
    # Apache License 2.0
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.parameter.Parameter(params)

    def forward(self, *losses: torch.Tensor):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum


class DynamicWeightAverageLoss(nn.Module):
    r"""Dynamic Weight Average (DWA).

    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.
    https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/DWA.py
    Args:
        T (float, default=2.0): The softmax temperature.
    """

    def __init__(self, num=2, T=2.0):
        super().__init__()
        self.num = num
        self.T = T
        self.register_buffer("train_loss_buffer", torch.full((2, num), float("inf")))

    def forward(self, *losses: torch.Tensor):
        loss = torch.stack(losses)
        if torch.isfinite(self.train_loss_buffer).all().item():
            w_i = torch.as_tensor(
                self.train_loss_buffer[-1, :]
                / self.train_loss_buffer[0, :],
                device=loss.device,
            )
            batch_weight = self.num * nn.functional.softmax(w_i / self.T, dim=-1)
        else:
            batch_weight = torch.ones(self.num, device=loss.device)

        loss_sum = loss.mul(batch_weight).sum()
        return loss_sum

    def step(self, *losses):
        losses = torch.tensor(losses)
        self.train_loss_buffer = torch.concat(
            (self.train_loss_buffer[1:], losses.unsqueeze(0))
        )


class ScaledDetachedLoss(nn.Module):
    def forward(self, *losses: torch.Tensor):
        loss0 = losses[0]
        loss = loss0
        for l in losses[1:]:
            loss = loss + l / (l / loss0).detach()
        return loss


class GeometricLoss(nn.Module):
    r"""Geometric Loss Strategy (GLS).

    This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
    and implemented by us.

    https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/GLS.py
    """

    def forward(self, *losses: torch.Tensor):
        loss = torch.stack(losses)
        return torch.pow(loss.prod(), 1.0 / loss.size(0))
