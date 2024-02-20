__all__ = ["PeptideSpectralLibraryHdf"]

from types import MappingProxyType
from typing import Optional, Union, overload
import numpy.typing as npt
import numpy as np
import pandas as pd

from .abs import (
    PeptideSpectralLibraryBase,
    Indices,
    Columns,
)
from ...util.io.hdf import HdfFile


class PeptideSpectralLibraryHdf(PeptideSpectralLibraryBase, HdfFile):
    class CONST(PeptideSpectralLibraryBase.CONST):
        GROUP_NAMES = MappingProxyType(
            {
                PeptideSpectralLibraryBase.CONST.TABLE_PROTEIN_INFO: "Analytes",
                PeptideSpectralLibraryBase.CONST.TABLE_PEPTIDE_INFO: "Analytes",
                PeptideSpectralLibraryBase.CONST.TABLE_PEPTIDE_PROTEIN_MAP: "Analytes",
                PeptideSpectralLibraryBase.CONST.TABLE_PRECURSOR_INFO: "Analytes",
                PeptideSpectralLibraryBase.CONST.TABLE_SPECTRUM_INFO: "Spectra",
                PeptideSpectralLibraryBase.CONST.TABLE_FRAGMENTS: "Spectra",
                PeptideSpectralLibraryBase.CONST.INDICES_FRAGMENTS: "Spectra",
                PeptideSpectralLibraryBase.CONST.TABLE_RETENTION_TIME: "Separation",
                PeptideSpectralLibraryBase.CONST.TABLE_ION_MOBILITY: "Separation",
            }
        )

    def __init__(self, file_name: str, **kwargs):
        super().__init__(file_name, **kwargs)

    def _read_dataset_length(
        self,
        name: str,
    ) -> int:
        group_name = self.CONST.GROUP_NAMES[name]
        dataset_name = name

        try:
            return self.read(
                group_name=group_name,
                dataset_name=dataset_name,
                return_dataset_shape=True,
            )[0]
        except KeyError:
            return 0

    @overload
    def _read_dataset(
        self,
        name: str,
        indices: Optional[Indices],
        columns: Optional[Columns],
    ) -> Union[pd.DataFrame, None]:
        ...

    @overload
    def _read_dataset(
        self,
        name: str,
        indices: Optional[Indices],
    ) -> Union[npt.NDArray, None]:
        ...

    def _read_dataset(
        self,
        name: str,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Union[pd.DataFrame, npt.NDArray, None]:
        group_name = self.CONST.GROUP_NAMES[name]
        dataset_name = name
        try:
            self.read(
                group_name=group_name,
                dataset_name=dataset_name,
                return_dataset_shape=True,
            )
        except KeyError:
            return None

        if indices is None and columns is None:
            return self.read(group_name=group_name, dataset_name=dataset_name)

        if indices is None or isinstance(indices, slice):
            should_sort = False
            reconstruct = slice(None)
        else:
            should_sort = True
            indices, reconstruct = np.unique(np.array(indices), return_inverse=True)

        if columns is None:
            data = self.read(
                group_name=group_name,
                dataset_name=dataset_name,
                return_dataset_slice=indices if indices is not None else slice(None),
            )
        else:
            data = pd.DataFrame.from_dict(
                {
                    col: self.read(
                        group_name=f"{group_name if group_name is not None else ''}/{dataset_name}",
                        dataset_name=col,
                        return_dataset_slice=indices
                        if indices is not None
                        else slice(None),
                    )
                    for col in columns
                }
            )

        if should_sort:
            if isinstance(data, pd.DataFrame):
                data = data.iloc[reconstruct].reset_index(drop=True)
            else:
                data = data[reconstruct]
        return data

    def get_protein_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_PEPTIDE_INFO,
            indices=indices,
            columns=columns,
        )

    def get_peptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_PEPTIDE_INFO,
            indices=indices,
            columns=columns,
        )

    def get_peptide_protein_map(
        self, indices: Optional[Indices] = None
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_PEPTIDE_PROTEIN_MAP,
            indices=indices,
            columns=None,
        )

    def get_precursor_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_PRECURSOR_INFO,
            indices=indices,
            columns=columns,
        )

    def get_spectrum_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_SPECTRUM_INFO,
            indices=indices,
            columns=columns,
        )

    def get_spectrum_fragments(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_FRAGMENTS,
            indices=indices,
            columns=columns,
        )

    def get_spectrum_fragment_indices(
        self, indices: Optional[Indices] = None
    ) -> Optional[npt.NDArray]:
        return self._read_dataset(
            name=self.CONST.INDICES_FRAGMENTS,
            indices=indices,
        )

    def get_retention_time(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_RETENTION_TIME,
            indices=indices,
            columns=columns,
        )

    def get_ion_mobility(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_ION_MOBILITY,
            indices=indices,
            columns=columns,
        )

    @property
    def num_proteins(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_PROTEIN_INFO)

    @property
    def num_peptides(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_PEPTIDE_INFO)

    @property
    def num_precursors(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_PRECURSOR_INFO)

    @property
    def num_spectra(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_SPECTRUM_INFO)

    @property
    def num_spectrum_fragments(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_FRAGMENTS)

    @property
    def num_retention_time(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_RETENTION_TIME)

    @property
    def num_ion_mobility(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_ION_MOBILITY)

    def _append_dataset(self, name: str, data, is_indices=False):
        group_name = self.CONST.GROUP_NAMES[name]
        dataset_name = name

        if group_name not in self.read():
            self.write(group_name)
        if dataset_name not in self.read(group_name=group_name):
            self.write(
                data,
                group_name=group_name,
                dataset_name=dataset_name,
                dataset_compression="gzip",
                compression_opts=9,
                dataset_resizable=True,
            )
            return
        else:
            if is_indices:
                data = data[1:]
            self.write(
                data, group_name=group_name, dataset_name=dataset_name, append=True
            )

    def _append_prepared_data(self, prepared_data: dict):
        for k, v in prepared_data.items():
            if isinstance(v, pd.DataFrame):
                if len(v) > 0:
                    self._append_dataset(name=k, data=v)
            elif isinstance(v, np.ndarray):
                if len(v) > 1:
                    self._append_dataset(name=k, data=v, is_indices=True)
            else:
                raise TypeError(
                    f"data type of {k} not supported: pandas.DataFrame or numpy.ndarray expected but {type(v)} given"
                )
