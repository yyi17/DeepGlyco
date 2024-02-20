__all__ = ["BatchProto", "MS2DataConverterProto", "MS2Dataset"]

import os
from logging import Logger
from typing import Any, Generic, List, Mapping, Optional, Protocol, TypeVar

import torch
from torch.utils.data import Dataset

from ...speclib.common.speclib import MS2SpectrumType, SpectralLibraryBase
from ...util.progress import ProgressFactoryProto


SpectrumType = TypeVar("SpectrumType", contravariant=True)
DataType = TypeVar("DataType", covariant=True)


class BatchProto(Generic[DataType], Protocol):
    indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        ...

    def to(self, device) -> "BatchProto[DataType]":
        ...


class MS2DataConverterProto(Generic[SpectrumType, DataType], Protocol):
    def spectrum_to_tensor(self, spectrum: SpectrumType) -> DataType:
        ...

    def get_configs(self) -> Mapping[str, Any]:
        ...


class MS2Dataset(Generic[MS2SpectrumType, DataType], Dataset[DataType]):
    def __init__(
        self,
        converter: MS2DataConverterProto[MS2SpectrumType, DataType],
        speclib: SpectralLibraryBase[MS2SpectrumType],
        progress_factory: Optional[ProgressFactoryProto] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__()
        self.speclib = speclib
        self.converter = converter
        self.progress_factory = progress_factory
        self.logger = logger
        self._data: Optional[List[DataType]] = None

    def __len__(self) -> int:
        if self._data is not None:
            return len(self._data)
        return self.speclib.num_spectra

    def __getitem__(self, idx) -> DataType:
        if self._data is not None:
            return self._data[idx]
        spectrum = self.speclib.get_spectra([idx])[0]
        return self.converter.spectrum_to_tensor(spectrum)

    def load(self, cache: Optional[str] = None):
        config = self.converter.get_configs()
        if cache is not None and os.path.exists(cache):
            if self.logger:
                self.logger.info(f"loading dataset using cache {cache}")
            data = torch.load(cache)
            if data["config"] == config:
                self._data = data["data"]
                if self.logger:
                    self.logger.info(f"dataset cache check passed {cache}")
                return

            if self.logger:
                self.logger.info(
                    f"dataset cache not match, reloading the original speclib"
                )
        if self.logger:
            self.logger.info(f"loading dataset from {self.speclib}")
        spectra = self.speclib.iter_spectra()
        if self.progress_factory:
            spectra = self.progress_factory(
                spectra, total=self.speclib.num_spectra, desc="Loading"
            )
        self._data = [
            self.converter.spectrum_to_tensor(spectrum) for spectrum in spectra
        ]
        if self.logger:
            self.logger.info(f"dataset loaded {self.speclib}")
        if cache is not None:
            torch.save({"data": self._data, "config": config}, cache)
            if self.logger:
                self.logger.info(f"cache saved to {cache}")
