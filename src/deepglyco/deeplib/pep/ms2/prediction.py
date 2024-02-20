__all__ = ["PeptideMS2Predictor"]

import itertools
import os
from logging import Logger
from typing import Any, Iterable, List, Mapping, Optional, Union

import pandas as pd
import torch

from ....speclib.pep.spec import PeptideMS2Spectrum
from ....util.progress import ProgressFactoryProto
from ...common.prediction import PredictorBase
from ..common.data import PeptideData, PeptideDataBatch
from .data import PeptideMS2OutputConverter, unbatch_peptide_fragment_intensity
from .model import PeptideMS2BiLSTM, PeptideMS2CNNLSTM


class PeptideMS2Predictor(PredictorBase[PeptideData, torch.Tensor]):
    def __init__(
        self,
        converter: PeptideMS2OutputConverter,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ms2model.yaml"
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
        if model_type == "PeptideMS2BiLSTM":
            model_factory = PeptideMS2BiLSTM
        elif model_type == "PeptideMS2CNNLSTM":
            model_factory = PeptideMS2CNNLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")
        return model_factory(self.get_configs())

    def _collate_data(self, batch_data: List[PeptideData]):
        return PeptideDataBatch.collate(batch_data)

    def _unbatch_prediction(self, batch: PeptideDataBatch, pred: torch.Tensor):
        return unbatch_peptide_fragment_intensity(batch, pred)

    def predict(
        self, input: pd.DataFrame, batch_size: int = 64
    ) -> Iterable[PeptideMS2Spectrum]:
        peps, peps1 = itertools.tee(
            input[["modified_sequence", "precursor_charge"]].itertuples(
                index=False, name="PeptideData"
            )
        )
        data = (
            self.converter.peptide_to_tensor(
                sequence=t.modified_sequence,
                precursor_charge=t.precursor_charge,
            )
            for t in peps1
        )
        pred = self.predict_data(data, batch_size=batch_size)

        return (
            self.converter.tensor_to_spectrum(
                sequence=t.modified_sequence,
                precursor_charge=t.precursor_charge,
                fragment_intensity=x,
            )
            for t, x in zip(peps, pred)
        )
