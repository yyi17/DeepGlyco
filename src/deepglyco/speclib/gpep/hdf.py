__all__ = ["GlycoPeptideSpectralLibraryHdf"]

from types import MappingProxyType
from typing import Optional

import pandas as pd

from ..common.speclib import Columns, Indices
from ..pep import PeptideSpectralLibraryHdf
from .abs import GlycoPeptideSpectralLibraryBase


class GlycoPeptideSpectralLibraryHdf(
    GlycoPeptideSpectralLibraryBase, PeptideSpectralLibraryHdf
):
    class CONST(GlycoPeptideSpectralLibraryBase.CONST):
        GROUP_NAMES = MappingProxyType(
            {
                **PeptideSpectralLibraryHdf.CONST.GROUP_NAMES,
                GlycoPeptideSpectralLibraryBase.CONST.TABLE_GLYCOPEPTIDE_INFO: "Analytes",
                GlycoPeptideSpectralLibraryBase.CONST.TABLE_GLYCAN_INFO: "Analytes",
            }
        )

    def __init__(self, file_name: str, **kwargs):
        super().__init__(file_name, **kwargs)

    def get_glycan_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_GLYCAN_INFO,
            indices=indices,
            columns=columns,
        )

    def get_glycopeptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._read_dataset(
            name=self.CONST.TABLE_GLYCOPEPTIDE_INFO,
            indices=indices,
            columns=columns,
        )

    @property
    def num_glycans(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_GLYCAN_INFO)

    @property
    def num_glycopeptides(self) -> int:
        return self._read_dataset_length(name=self.CONST.TABLE_GLYCOPEPTIDE_INFO)
