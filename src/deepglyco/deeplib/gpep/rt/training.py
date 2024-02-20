__all__ = ["GlycoPeptideRTTrainer"]

import os
from logging import Logger
from typing import List, Mapping, Optional, Union

import torch

from ...util.weightloss import AutomaticWeightedLoss

from ....util.progress import ProgressFactoryProto
from ...common.training import TrainerBase
from ...pep.rt.loss import RTMAELoss
from ...util.writer import SummaryWriterProto
from .data import GlycoPeptideRTData, GlycoPeptideRTDataBatch
from .model import GlycoPeptideRTTreeLSTM


class GlycoPeptideRTTrainer(TrainerBase[GlycoPeptideRTData]):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        summary_writer: Optional[SummaryWriterProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "gprtmodel.yaml"
            )

        super().__init__(
            configs=configs,
            progress_factory=progress_factory,
            summary_writer=summary_writer,
            logger=logger,
        )

    def _collate_data(self, batch_data: List[GlycoPeptideRTData]):
        return GlycoPeptideRTDataBatch.collate(batch_data)

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "GlycoPeptideRTTreeLSTM":
            model_factory = GlycoPeptideRTTreeLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _build_loss_fn(self):
        return RTMAELoss()

    # def _build_metrics_fn(self):
    #     return { "mse": RTMSELoss() }

    # def _build_loss_fn(self):
    #     self.rt_loss_fn = RTMAELoss()
    #     total_loss_fn = AutomaticWeightedLoss(num=4)
    #     return total_loss_fn

    # def _calculate_loss_metrics(
    #     self, batch: GlycoPeptideRTDataBatch, pred: tuple[torch.Tensor, torch.Tensor]
    # ) -> dict:
    #     pep_loss = self.rt_loss_fn(batch, pred[0])
    #     gly_loss = self.rt_loss_fn(batch, pred[1])

    #     metrics_dict: dict[str, torch.Tensor] = dict(mae_pep=pep_loss, mae_gly=gly_loss)
    #     losses = []
    #     for loss in metrics_dict.values():
    #         if len(loss.shape) > 0:
    #             loss = loss.mean()
    #         losses.append(loss)

    #     total_loss = self.loss_fn(*losses)

    #     if self.metrics_fn:
    #         metrics_dict.update({k: v(batch, pred) for k, v in self.metrics_fn.items()})

    #     metrics = {}
    #     metrics_all = {}
    #     for k, m in metrics_dict.items():
    #         if len(m.shape) > 0:
    #             m_all = torch.empty_like(m)
    #             for i in range(m.size(0)):
    #                 m_all[batch.indices[i]] = m[i]
    #             metrics_all[k] = m_all.detach().cpu()
    #             m = m.mean()
    #         metrics[k] = m

    #     r = {}
    #     r["loss"] = total_loss
    #     if metrics:
    #         r["metrics"] = metrics
    #     if metrics_all:
    #         r["metrics_all"] = metrics_all
    #     return r
