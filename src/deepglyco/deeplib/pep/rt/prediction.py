__all__ = ["PeptideRTPredictor"]

import os
from logging import Logger
from typing import Any, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch

from ....util.progress import ProgressFactoryProto
from ...common.prediction import PredictorBase
from ..common.data import PeptideData, PeptideDataBatch
from .data import PeptideRTOutputConverter
from .model import PeptideRTBiLSTM


class PeptideRTPredictor(PredictorBase[PeptideData, torch.Tensor]):
    def __init__(
        self,
        converter: PeptideRTOutputConverter,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "rtmodel.yaml"
            )

        self.converter = converter
        super().__init__(
            configs=configs, progress_factory=progress_factory, logger=logger
        )

    def set_configs(self, configs: Mapping[str, Any]):
        data = configs.get("data", None)
        if data is not None:
            self.converter.set_configs(data)
            configs = {k: v for k, v in configs.items() if k != "data"}
        return super().set_configs(configs)

    def get_configs(self, deep: bool = True):
        r = super().get_configs(deep)
        if deep:
            r["data"] = self.converter.get_configs()
        return r

    def _build_model(self):
        model_type = self.get_config("model", "type", typed=str)
        if model_type == "PeptideRTBiLSTM":
            model_factory = PeptideRTBiLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")
        return model_factory(self.get_configs())

    def _collate_data(self, batch_data: List[PeptideData]):
        return PeptideDataBatch.collate(batch_data)

    def _unbatch_prediction(self, batch: PeptideDataBatch, pred: torch.Tensor):
        result: list[Any] = [None] * batch.batch_size
        for i in range(batch.batch_size):
            idx = int(batch.indices[i].item())
            result[idx] = pred[i]
        return result

    def predict(
        self, input: pd.DataFrame, batch_size: int = 128
    ) -> pd.DataFrame:
        peps = input[["modified_sequence"]].itertuples(
            index=False, name="PeptideData"
        )
        data = (
            self.converter.peptide_to_tensor(
                sequence=t.modified_sequence,
            )
            for t in peps
        )
        pred = self.predict_data(data, batch_size=batch_size)

        rt = np.array([x.item() for x in pred])
        rt = pd.Series(rt, name="retention_time", index=input.index)

        if "retention_time" in input.columns:
            input = input.drop(columns=["retention_time"])
        result = pd.concat((input, rt), axis=1, copy=False)
        return result
