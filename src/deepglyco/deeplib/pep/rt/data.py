__all__ = [
    "PeptideRTData",
    "PeptideRTDataBatch",
    "PeptideRTDataConverter",
    "PeptideRTDataset",
]

import os
from logging import Logger
from typing import (
    Any,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from ....chem.pep.mods import ModifiedSequenceParser
from ....speclib.common.rtlib import RetentionTimeLibraryBase
from ....util.progress import ProgressFactoryProto
from ..common.data import PeptideDataConverter


class PeptideRTData(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    retention_time: torch.Tensor

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            retention_time=self.retention_time.to(device),
        )


class PeptideRTDataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    retention_time: torch.Tensor
    indices: torch.Tensor

    @property
    def batch_size(self):
        return self.indices.size(0)

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            retention_time=self.retention_time.to(device),
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[PeptideRTData]):
        length = torch.as_tensor([x.length for x in batch_data])
        indices = torch.argsort(length, descending=True)
        batch_data = [batch_data[i] for i in indices]
        length = length[indices]

        modifications = [
            x.modifications for x in batch_data if x.modifications is not None
        ]
        if len(modifications) == 0:
            modifications = None
        elif len(modifications) != len(batch_data):
            raise ValueError(f"modifications contains None")

        return cls(
            sequence=pad_sequence([x.sequence for x in batch_data], batch_first=True),
            length=length,
            modifications=pad_sequence(modifications, batch_first=True)
            if modifications is not None
            else None,
            retention_time=torch.as_tensor([x.retention_time for x in batch_data]),
            indices=indices,
        )


class PeptideRTEntryProto(Protocol):
    modified_sequence: str
    retention_time: float


class PeptideRTDataConverter(PeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        configs: Union[str, dict],
    ):
        super().__init__(sequence_parser=sequence_parser, configs=configs)

    def rt_entry_to_tensor(self, entry: PeptideRTEntryProto) -> PeptideRTData:
        parsed_sequence = self.sequence_parser.parse_modified_sequence(
            entry.modified_sequence
        )
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)

        retention_time = torch.tensor(entry.retention_time)
        return PeptideRTData(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            retention_time=retention_time,
        )


from torch.utils.data import Dataset

DataType = TypeVar("DataType", covariant=True)


class RTDataConverterProto(Generic[DataType], Protocol):
    def rt_entry_to_tensor(self, entry) -> DataType:
        ...


    def get_configs(self) -> Mapping[str, Any]:
        ...


class RTDataset(Generic[DataType], Dataset[DataType]):
    def __init__(
        self,
        converter: RTDataConverterProto[DataType],
        rtlib: RetentionTimeLibraryBase,
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__()
        self.rtlib = rtlib
        self.converter = converter
        self.progress_factory = progress_factory
        self.logger = logger
        self._data: Optional[List[DataType]] = None

    def __len__(self) -> int:
        if self._data is not None:
            return len(self._data)
        return self.rtlib.num_retention_time

    def __getitem__(self, idx) -> DataType:
        if self._data is not None:
            return self._data[idx]
        entry = self.rtlib.get_retention_time_data([idx])
        assert entry is not None
        entry = next(entry.itertuples(index=False, name="RTEntry"))
        return self.converter.rt_entry_to_tensor(entry)

    def load(self, cache: Optional[str] = None):
        if cache is not None and os.path.exists(cache):
            if self.logger:
                self.logger.info(f"loading dataset using cache {cache}")
            data = torch.load(cache)
            if data["config"] == self.converter.get_configs():
                self._data = data["data"]
                if self.logger:
                    self.logger.info(f"dataset cache check passed {cache}")
                return

            if self.logger:
                self.logger.info(
                    f"dataset cache not match, reloading the original rtlib"
                )
        if self.logger:
            self.logger.info(f"loading dataset from {self.rtlib}")
        entries = self.rtlib.get_retention_time_data()
        assert entries is not None
        entries = entries.itertuples(index=False, name="RTEntry")
        if self.progress_factory:
            entries = self.progress_factory(
                entries, total=self.rtlib.num_retention_time, desc="Loading"
            )
        self._data = [self.converter.rt_entry_to_tensor(entry) for entry in entries]
        if self.logger:
            self.logger.info(f"dataset loaded {self.rtlib}")
        if cache is not None:
            torch.save(
                {"data": self._data, "config": self.converter.get_configs()}, cache
            )
            if self.logger:
                self.logger.info(f"cache saved to {cache}")


PeptideRTDataset = RTDataset[PeptideRTData]

PeptideRTOutputConverter = PeptideRTDataConverter
