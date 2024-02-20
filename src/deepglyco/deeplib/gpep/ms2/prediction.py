__all__ = ["GlycoPeptideMS2Predictor"]

import itertools
import os
from logging import Logger
from typing import Any, Iterable, List, Mapping, Optional, Union

import pandas as pd

from ....speclib.gpep.spec import GlycoPeptideMS2Spectrum
from ....util.progress import ProgressFactoryProto
from ...common.prediction import PredictorBase
from ..common.data import GlycoPeptideData, GlycoPeptideDataBatch
from .data import (
    GlycoPeptideMS2Output,
    GlycoPeptideMS2OutputConverter,
    unbatch_glycopeptide_ms2_output,
)
from .model import GlycoPeptideMS2TreeLSTM


class GlycoPeptideMS2Predictor(PredictorBase[GlycoPeptideData, GlycoPeptideMS2Output]):
    def __init__(
        self,
        converter: GlycoPeptideMS2OutputConverter,
        configs: Union[str, dict, None] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "gpms2model.yaml"
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
        if model_type == "GlycoPeptideMS2TreeLSTM":
            model_factory = GlycoPeptideMS2TreeLSTM
        else:
            raise ValueError(f"unknown model type {model_type}")

        return model_factory(self.get_configs())

    def _collate_data(self, batch_data: List[GlycoPeptideData]):
        return GlycoPeptideDataBatch.collate(batch_data)

    def _unbatch_prediction(
        self, batch: GlycoPeptideDataBatch, pred: GlycoPeptideMS2Output
    ):
        return unbatch_glycopeptide_ms2_output(batch, pred)

    def predict(
        self,
        input: pd.DataFrame,
        batch_size: int = 128,
        keep_zeros: bool = False,
    ) -> Iterable[GlycoPeptideMS2Spectrum]:
        peps, peps1 = itertools.tee(
            input[
                [
                    "modified_sequence",
                    "glycan_struct",
                    "glycan_position",
                    "precursor_charge",
                ]
            ].itertuples(index=False, name="GlycoPeptideData")
        )
        data = (
            self.converter.glycopeptide_to_tensor(
                sequence=t.modified_sequence,
                glycan_struct=t.glycan_struct,
                glycan_position=t.glycan_position,
                precursor_charge=t.precursor_charge,
            )
            for t in peps1
        )
        if self.progress_factory:
            data = self.progress_factory(
                data,
                total=len(input),
                desc=f"Preparing data",
            )
        data = list(data)

        pred = self.predict_data(data, batch_size=batch_size)

        return (
            self.converter.tensor_to_spectrum(
                sequence=t.modified_sequence,
                glycan_struct=t.glycan_struct,
                glycan_position=t.glycan_position,
                precursor_charge=t.precursor_charge,
                ms2=x,
                keep_zeros=keep_zeros,
            )
            for t, x in zip(peps, pred)
        )
