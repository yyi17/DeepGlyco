import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ....util.collections.dict import chain_item_typed
from ...common.training import TrainerBase
from ...util.weightloss import (
    AutomaticWeightedLoss,
    DynamicWeightAverageLoss,
    ScaledDetachedLoss,
)
from .data import GlycoPeptideMS2Data, GlycoPeptideMS2DataBatch, GlycoPeptideMS2Output
from .loss import GlycoSpectralAngleLoss
from .model import GlycoPeptideMS2TreeLSTM


class GlycoPeptideMS2Trainer(TrainerBase[GlycoPeptideMS2Data]):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
        progress_factory=None,
        summary_writer=None,
        logger=None,
    ):
        self.config: dict
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "gpms2model.yaml"
            )

        super().__init__(
            configs=configs,  # type: ignore
            progress_factory=progress_factory,
            summary_writer=summary_writer,
            logger=logger,
        )

    def _collate_data(self, batch_data: List[GlycoPeptideMS2Data]):
        return GlycoPeptideMS2DataBatch.collate(batch_data)

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "GlycoPeptideMS2TreeLSTM":
            model_factory = GlycoPeptideMS2TreeLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _build_loss_fn(self):
        self.spec_loss_fn = GlycoSpectralAngleLoss(reduction="none")
        self.ratio_loss_fn = torch.nn.MSELoss(reduction="none")

        # total_loss_fn = AutomaticWeightedLoss(num=4)
        total_loss_fn = DynamicWeightAverageLoss(num=4)
        # total_loss_fn = ScaledDetachedLoss()
        return total_loss_fn

    def train(self, num_epochs: int, **kwargs) -> dict:
        if isinstance(self.loss_fn, DynamicWeightAverageLoss):
            total_loss_fn = self.loss_fn
            callback = kwargs.pop("callback", None)

            def loss_callback(summary: dict):
                total_loss_fn.step(*summary["metrics"]["training"].values())
                if callback:
                    callback(summary)

            kwargs["callback"] = loss_callback

        return super().train(num_epochs, **kwargs)

    @property
    def default_metrics(self):
        return "sa_total"

    def _calculate_loss_metrics(
        self,
        batch: GlycoPeptideMS2DataBatch,
        pred: GlycoPeptideMS2Output,
        keep_all_loss_metrics: bool = False,
    ) -> dict:
        spec_loss = self.spec_loss_fn(batch, pred)
        ratio_loss = self.ratio_loss_fn(
            pred.fragment_intensity_ratio, batch.fragment_intensity_ratio
        )

        metrics_dict: dict[str, torch.Tensor] = dict(**spec_loss, mse_ratio=ratio_loss)
        losses = []
        for loss in metrics_dict.values():
            if len(loss.shape) > 0:
                loss = loss.nanmean()
            losses.append(loss)

        total_loss = self.loss_fn(*losses)

        if self.metrics_fn:
            metrics_dict.update({k: v(batch, pred) for k, v in self.metrics_fn.items()})

        metrics = {}
        metrics_all = {}
        for k, m in metrics_dict.items():
            if len(m.shape) > 0:
                if keep_all_loss_metrics:
                    m_all = torch.empty_like(m)
                    for i in range(m.size(0)):
                        m_all[batch.indices[i]] = m[i]
                    metrics_all[k] = m_all.detach().cpu()
                m = m.nanmean()
            metrics[k] = m

        r = {}
        r["loss"] = total_loss
        if metrics:
            r["metrics"] = metrics
        if metrics_all:
            r["metrics_all"] = metrics_all
        return r
