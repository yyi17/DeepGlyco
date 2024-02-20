__all__ = ["GlycoPeptideSpectrumFilter"]

from typing import Generic, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np

from ..pep.filter import PeptideSpectrumFilter
from .spec import GlycoPeptideMS2SpectrumProto

MS2SpectrumType = TypeVar("MS2SpectrumType", bound=GlycoPeptideMS2SpectrumProto)


class GlycoPeptideSpectrumFilter(
    Generic[MS2SpectrumType], PeptideSpectrumFilter[MS2SpectrumType]
):
    def check_filter_num_fragments(
        self,
        spectrum: MS2SpectrumType,
        min_num_fragments: Union[int, Mapping[str, int], None] = None,
        min_num_peptide_fragments: Optional[int] = None,
        min_num_glycan_fragments: Optional[int] = None,
        **kwargs,
    ):
        return (
            super().check_filter_num_fragments(
                spectrum, min_num_fragments=min_num_fragments, **kwargs
            )
            and (
                min_num_peptide_fragments is None
                or (spectrum.fragment_number > 0).sum() >= min_num_peptide_fragments
            )
            and (
                min_num_glycan_fragments is None
                or (spectrum.fragment_number <= 0).sum() >= min_num_glycan_fragments
            )
        )

    def filter_fragments_by_annotations(
        self,
        spectrum: MS2SpectrumType,
        peptide_fragment_type: Union[str, Sequence[str], None] = None,
        peptide_fragment_loss_type: Union[str, Sequence[str], None] = None,
        peptide_fragment_charge: Union[int, Sequence[int], None] = None,
        glycan_fragment_type: Union[str, Sequence[str], None] = None,
        glycan_fragment_charge: Union[int, Sequence[int], None] = None,
        return_index: bool = False,
        **kwargs,
    ):
        peptide_fragment_index = spectrum.fragment_number > 0
        glycan_fragment_index = ~peptide_fragment_index

        fragment_index_1 = super().filter_fragments_by_annotations(
            spectrum,
            fragment_type=peptide_fragment_type,
            loss_type=peptide_fragment_loss_type,
            fragment_charge=peptide_fragment_charge,
            return_index=True,
        )
        assert (
            isinstance(fragment_index_1, np.ndarray)
            and fragment_index_1.dtype == np.bool_
        )
        peptide_fragment_index &= fragment_index_1

        fragment_index_1 = super().filter_fragments_by_annotations(
            spectrum,
            fragment_type=glycan_fragment_type,
            fragment_charge=glycan_fragment_charge,
            return_index=True,
        )
        assert (
            isinstance(fragment_index_1, np.ndarray)
            and fragment_index_1.dtype == np.bool_
        )
        glycan_fragment_index &= fragment_index_1

        fragment_index = peptide_fragment_index | glycan_fragment_index

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )
