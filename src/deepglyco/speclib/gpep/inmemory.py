__all__ = ["GlycoPeptideSpectralLibraryInMemory"]

from typing import Dict, Optional, Union

import numpy.typing as npt
import pandas as pd

from ..common.speclib import Columns, Indices
from .abs import GlycoPeptideSpectralLibraryBase
from ..pep import PeptideSpectralLibraryInMemory


class GlycoPeptideSpectralLibraryInMemory(
    GlycoPeptideSpectralLibraryBase, PeptideSpectralLibraryInMemory
):
    def __init__(self, other: Optional[GlycoPeptideSpectralLibraryBase] = None):
        if other is None:
            self.data: Dict[str, Union[pd.DataFrame, npt.NDArray]] = {}
        elif isinstance(other, GlycoPeptideSpectralLibraryInMemory):
            self.data = other.data.copy()
        else:
            self.data = {
                k: v
                for k, v in {
                    self.CONST.TABLE_PROTEIN_INFO: other.get_protein_info(),
                    self.CONST.TABLE_PEPTIDE_INFO: other.get_peptide_info(),
                    self.CONST.TABLE_PEPTIDE_PROTEIN_MAP: other.get_peptide_protein_map(),
                    self.CONST.TABLE_GLYCAN_INFO: other.get_glycan_info(),
                    self.CONST.TABLE_GLYCOPEPTIDE_INFO: other.get_glycopeptide_info(),
                    self.CONST.TABLE_PRECURSOR_INFO: other.get_precursor_info(),
                    self.CONST.TABLE_RETENTION_TIME: other.get_retention_time(),
                    self.CONST.TABLE_ION_MOBILITY: other.get_ion_mobility(),
                    self.CONST.TABLE_SPECTRUM_INFO: other.get_spectrum_info(),
                    self.CONST.TABLE_FRAGMENTS: other.get_spectrum_fragments(),
                    self.CONST.INDICES_FRAGMENTS: other.get_spectrum_fragment_indices(),
                }.items()
                if v is not None
            }

    def get_glycan_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(self.CONST.TABLE_GLYCAN_INFO, indices, columns)

    def get_glycopeptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        return self._get_data_subset(
            self.CONST.TABLE_GLYCOPEPTIDE_INFO, indices, columns
        )
