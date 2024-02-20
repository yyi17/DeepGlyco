__all__ = ["PeptideMS2Trainer"]

import os
from logging import Logger
from typing import List, Optional, Union

from ....util.progress import ProgressFactoryProto
from ...common.training import TrainerBase
from ...util.writer import SummaryWriterProto
from .data import PeptideMS2Data, PeptideMS2DataBatch
from .loss import SpectralAngleLoss
from .model import PeptideMS2BiLSTM, PeptideMS2CNNLSTM


class PeptideMS2Trainer(TrainerBase[PeptideMS2Data]):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        summary_writer: Optional[SummaryWriterProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ms2model.yaml"
            )

        super().__init__(
            configs=configs,
            progress_factory=progress_factory,
            summary_writer=summary_writer,
            logger=logger,
        )

    def _collate_data(self, batch_data: List[PeptideMS2Data]):
        return PeptideMS2DataBatch.collate(batch_data)

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "PeptideMS2BiLSTM":
            model_factory = PeptideMS2BiLSTM
        elif model_type == "PeptideMS2CNNLSTM":
            model_factory = PeptideMS2CNNLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _build_loss_fn(self):
        return SpectralAngleLoss(reduction="none")
