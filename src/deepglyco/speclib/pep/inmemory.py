__all__ = ["PeptideSpectralLibraryInMemory"]

from typing import Dict, Optional, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from .abs import (
    Columns,
    Indices,
    PeptideSpectralLibraryBase,
)


class PeptideSpectralLibraryInMemory(PeptideSpectralLibraryBase):
    def __init__(self, other: Union[PeptideSpectralLibraryBase, None] = None):
        if other is None:
            self.data: Dict[str, Union[pd.DataFrame, npt.NDArray]] = {}
        elif isinstance(other, PeptideSpectralLibraryInMemory):
            self.data = other.data.copy()
        else:
            self.data = {
                k: v
                for k, v in {
                    self.CONST.TABLE_PROTEIN_INFO: other.get_protein_info(),
                    self.CONST.TABLE_PEPTIDE_INFO: other.get_peptide_info(),
                    self.CONST.TABLE_PEPTIDE_PROTEIN_MAP: other.get_peptide_protein_map(),
                    self.CONST.TABLE_PRECURSOR_INFO: other.get_precursor_info(),
                    self.CONST.TABLE_RETENTION_TIME: other.get_retention_time(),
                    self.CONST.TABLE_ION_MOBILITY: other.get_ion_mobility(),
                    self.CONST.TABLE_SPECTRUM_INFO: other.get_spectrum_info(),
                    self.CONST.TABLE_FRAGMENTS: other.get_spectrum_fragments(),
                    self.CONST.INDICES_FRAGMENTS: other.get_spectrum_fragment_indices(),
                }.items()
                if v is not None
            }

    @overload
    def _get_data_subset(
        self,
        name: str,
        indices: Optional[Indices],
        columns: Optional[Columns],
    ) -> Union[pd.DataFrame, None]:
        ...

    @overload
    def _get_data_subset(
        self,
        name: str,
        indices: Optional[Indices],
    ) -> Union[npt.NDArray, None]:
        ...

    def _get_data_subset(
        self,
        name: str,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Union[pd.DataFrame, npt.NDArray, None]:
        data = self.data.get(name, None)
        if isinstance(data, pd.DataFrame):
            if columns is not None:
                data = data.loc[:, columns]
            if indices is not None:
                data = data.iloc[indices]
        if isinstance(data, np.ndarray):
            if columns is not None:
                raise ValueError("subseting numpy.ndarray by columns is not supported")
            if indices is not None:
                data = data[indices]
        return data

    def _append_prepared_data(
        self, prepared_data: Dict[str, Union[pd.DataFrame, npt.NDArray]]
    ):
        for k, v in prepared_data.items():
            data = self.data.get(k, None)
            if isinstance(data, pd.DataFrame) and isinstance(v, pd.DataFrame):
                v = pd.concat((data, v), copy=False)
            elif isinstance(data, np.ndarray) and isinstance(v, np.ndarray):
                v = np.concatenate((data, v[1:]))
            elif data is not None:
                raise TypeError(
                    f"data type of {k} not match: {type(data)} expected but {type(v)} given"
                )

            self.data[k] = v

    def get_protein_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_PROTEIN_INFO, indices, columns)

    def get_peptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_PEPTIDE_INFO, indices, columns)

    def get_peptide_protein_map(
        self, indices: Optional[Indices] = None
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(
            self.CONST.TABLE_PEPTIDE_PROTEIN_MAP, indices, None
        )

    def get_precursor_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_PRECURSOR_INFO, indices, columns)

    def get_spectrum_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_SPECTRUM_INFO, indices, columns)

    def get_spectrum_fragments(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_FRAGMENTS, indices, columns)

    def get_spectrum_fragment_indices(
        self, indices: Optional[Indices] = None
    ) -> Optional[np.ndarray]:
        return self._get_data_subset(self.CONST.INDICES_FRAGMENTS, indices)

    def get_retention_time(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_RETENTION_TIME, indices, columns)

    def get_ion_mobility(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_ION_MOBILITY, indices, columns)
