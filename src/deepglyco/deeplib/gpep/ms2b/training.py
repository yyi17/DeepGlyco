import os
from typing import Any, List, Union, cast

import torch

from ..ms2.training import GlycoPeptideMS2Trainer
from ....util.collections.dict import chain_item_typed
from ...common.training import TrainerBase
from ...util.weightloss import (
    AutomaticWeightedLoss,
    DynamicWeightAverageLoss,
    ScaledDetachedLoss,
)
from .data import GlycoPeptideBranchMS2Data, GlycoPeptideBranchMS2DataBatch, GlycoPeptideBranchMS2Output
from .loss import GlycoBranchSpectralAngleLoss
from .model import GlycoPeptideBranchMS2TreeLSTM


class GlycoPeptideBranchMS2Trainer(TrainerBase[GlycoPeptideBranchMS2Data]):
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
                os.path.dirname(os.path.abspath(__file__)), "gpms2_branch_model.yaml"
            )

        super().__init__(
            configs=configs,  # type: ignore
            progress_factory=progress_factory,
            summary_writer=summary_writer,
            logger=logger,
        )

    def _collate_data(self, batch_data: List[GlycoPeptideBranchMS2Data]):
        return GlycoPeptideBranchMS2DataBatch.collate(batch_data)

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "GlycoPeptideBranchMS2TreeLSTM":
            model_factory = GlycoPeptideBranchMS2TreeLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _build_loss_fn(self):
        self.spec_loss_fn = GlycoBranchSpectralAngleLoss(reduction="none")
        self.ratio_loss_fn = torch.nn.MSELoss(reduction="none")

        # total_loss_fn = AutomaticWeightedLoss(num=5)
        total_loss_fn = DynamicWeightAverageLoss(num=5)
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
        self, batch: GlycoPeptideBranchMS2DataBatch, pred: GlycoPeptideBranchMS2Output,
        keep_all_loss_metrics: bool = False
    ) -> dict:
        return GlycoPeptideMS2Trainer._calculate_loss_metrics(
            cast(Any, self),
            batch,
            cast(Any, pred),
            keep_all_loss_metrics=keep_all_loss_metrics,
        )
