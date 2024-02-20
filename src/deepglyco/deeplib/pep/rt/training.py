__all__ = ["PeptideRTTrainer"]

import os
from logging import Logger
from typing import List, Optional, Union

from ....util.progress import ProgressFactoryProto
from ...common.training import TrainerBase
from ...util.writer import SummaryWriterProto
from .data import PeptideRTData, PeptideRTDataBatch
from .loss import RTMAELoss
from .model import PeptideRTBiLSTM


class PeptideRTTrainer(TrainerBase[PeptideRTData]):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        summary_writer: Optional[SummaryWriterProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "rtmodel.yaml"
            )

        super().__init__(
            configs=configs,
            progress_factory=progress_factory,
            summary_writer=summary_writer,
            logger=logger,
        )

    def _collate_data(self, batch_data: List[PeptideRTData]):
        return PeptideRTDataBatch.collate(batch_data)

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "PeptideRTBiLSTM":
            model_factory = PeptideRTBiLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _build_loss_fn(self):
        return RTMAELoss()
