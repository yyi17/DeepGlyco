__all__ = ["PredictorBase"]

import abc
import math
from logging import Logger
from typing import (
    Generic,
    Iterable,
    List,
    Optional,
    Sized,
    TypeVar,
    Union,
)

import torch

from ...util.config import Configurable

from ...util.collections.dict import chain_item_typed
from ...util.collections.iter import batched
from ...util.progress import ProgressFactoryProto
from .data import BatchProto

DataType = TypeVar("DataType")
OutputType = TypeVar("OutputType")


class PredictorBase(abc.ABC, Generic[DataType, OutputType], Configurable):
    def __init__(
        self,
        configs: Union[str, dict],
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(configs)

        self.progress_factory = progress_factory
        self.logger = logger

    def load_model(self, model: Union[str, dict]):
        device = self.get_config("device", required=False, typed=str)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

        if isinstance(model, str):
            pretrained: dict = torch.load(model)
        else:
            pretrained = model
        self.set_configs(chain_item_typed(pretrained, dict, "config"))

        model_state = chain_item_typed(pretrained, dict, "state")
        self.model = self._build_model()
        self.model.load_state_dict(model_state)
        self.model.to(self.device)

    @abc.abstractmethod
    def _build_model(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def _collate_data(self, batch_data: List[DataType]) -> BatchProto[DataType]:
        pass

    @abc.abstractmethod
    def _unbatch_prediction(
        self, batch: BatchProto[DataType], pred: OutputType
    ) -> List[OutputType]:
        pass

    def predict_data(
        self, data: Iterable[DataType], batch_size: int = 64
    ) -> Iterable[OutputType]:
        batched_data = batched(data, n=batch_size)

        if self.progress_factory:
            batched_data = self.progress_factory(
                batched_data,
                total=math.ceil(len(data) / batch_size)
                if isinstance(data, Sized)
                else None,
                desc=f"Predicting",
            )

        self.model.eval()
        with torch.no_grad():
            for batch in batched_data:
                batch = self._collate_data(list(batch))
                batch = batch.to(self.device)
                pred: OutputType = self.model(batch)
                yield from self._unbatch_prediction(batch, pred)
