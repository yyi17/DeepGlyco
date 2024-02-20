__all__ = [
    "SpectralLibraryBase",
    "IndicesNonSlice",
    "Indices",
    "Columns",
]

import abc
import itertools
from typing import Generic, Iterable, List, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from .spec import MS2SpectrumProto

IndicesNonSlice = Union[List[int], npt.NDArray[np.int_], "pd.Series[int]"]
Indices = Union[slice, IndicesNonSlice]
Columns = List[str]

MS2SpectrumType = TypeVar("MS2SpectrumType", bound=MS2SpectrumProto)


class SpectralLibraryBase(abc.ABC, Generic[MS2SpectrumType]):
    @abc.abstractmethod
    def get_precursor_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_spectrum_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_spectrum_fragments(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_spectrum_fragment_indices(
        self, indices: Optional[Indices] = None
    ) -> Optional[npt.NDArray[np.int32]]:
        pass

    @property
    def num_precursors(self) -> int:
        data = self.get_precursor_info()
        if data is not None:
            return len(data)
        else:
            return 0

    @property
    def num_spectra(self) -> int:
        data = self.get_spectrum_info()
        if data is not None:
            return len(data)
        else:
            return 0

    @property
    def num_spectrum_fragments(self) -> int:
        data = self.get_spectrum_fragments()
        if data is not None:
            return len(data)
        else:
            return 0

    @abc.abstractmethod
    def get_precursor_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        pass

    @abc.abstractmethod
    def get_spectrum_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        pass

    def get_fragments_by_spectrum_id(
        self, spectrum_id_list: Iterable[int], columns: Optional[Columns] = None
    ):
        spec_indices = list(
            itertools.chain.from_iterable((i, i + 1) for i in spectrum_id_list)
        )
        frag_se_indices = self.get_spectrum_fragment_indices(indices=spec_indices)
        if frag_se_indices is None:
            raise ValueError("empty fragment indices")
        frag_se_indices = list(zip(*[iter(frag_se_indices)] * 2))
        frag_indices = list(
            itertools.chain.from_iterable(range(t[0], t[1]) for t in frag_se_indices)
        )
        fragments = self.get_spectrum_fragments(indices=frag_indices, columns=columns)
        if fragments is None:
            raise ValueError("empty fragments")
        subset_se_indices = np.array(
            [0] + [t[1] - t[0] for t in frag_se_indices]
        ).cumsum()
        return [
            fragments.iloc[subset_se_indices[i] : subset_se_indices[i + 1]]
            for i in range(len(subset_se_indices) - 1)
        ]

    @abc.abstractmethod
    def get_spectra(self, spectrum_id_list: IndicesNonSlice) -> List[MS2SpectrumType]:
        pass

    @abc.abstractmethod
    def iter_spectra(self) -> Iterable[MS2SpectrumType]:
        pass

    @abc.abstractmethod
    def import_spectra(self, spectra: Iterable[MS2SpectrumType]):
        pass
