__all__ = ["PeptideSpectrumFilter"]

from typing import Iterable, Mapping, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt

from ..common.filter import SpectrumFilter
from .spec import PeptideMS2SpectrumProto

MS2SpectrumType = TypeVar("MS2SpectrumType", bound=PeptideMS2SpectrumProto)


class PeptideSpectrumFilter(SpectrumFilter[MS2SpectrumType]):
    def check_filter_analyte_info(
        self,
        spectrum: MS2SpectrumType,
        precursor_charge: Union[int, Sequence[int], None] = None,
        **kwargs,
    ):
        if precursor_charge is None:
            return True
        elif isinstance(precursor_charge, int):
            return spectrum.precursor_charge == precursor_charge
        elif isinstance(precursor_charge, Iterable):
            return spectrum.precursor_charge in precursor_charge
        else:
            return spectrum.precursor_charge == precursor_charge

    def check_filter_num_fragments(
        self,
        spectrum: MS2SpectrumType,
        min_num_fragments: Union[int, Mapping[str, int], None] = None,
        **kwargs,
    ):
        if isinstance(min_num_fragments, Mapping):
            for frag_type, count in min_num_fragments.items():
                index = cast(
                    np.ndarray,
                    self.filter_fragments_by_type(
                        spectrum, frag_type, return_index=True
                    ),
                )
                if (index.sum() if index.dtype == np.bool_ else len(index)) <= count:
                    return False
            return True
        else:
            return super().check_filter_num_fragments(
                spectrum, min_num_fragments=min_num_fragments, **kwargs
            )

    def filter_fragments_by_annotations(
        self,
        spectrum: MS2SpectrumType,
        fragment_type: Union[str, Sequence[str], None] = None,
        loss_type: Union[str, Sequence[str], None] = None,
        fragment_charge: Union[int, Sequence[int], None] = None,
        return_index: bool = False,
        **kwargs,
    ):
        fragment_index = np.ones_like(spectrum.mz, dtype=np.bool_)
        if fragment_type is not None:
            fragment_index_1 = self.filter_fragments_by_type(
                spectrum, fragment_type=fragment_type, return_index=True
            )
            assert (
                isinstance(fragment_index_1, np.ndarray)
                and fragment_index_1.dtype == np.bool_
            )
            fragment_index &= fragment_index_1

        if loss_type is not None:
            fragment_index_1 = self.filter_fragments_by_loss_type(
                spectrum, loss_type=loss_type, return_index=True
            )
            assert (
                isinstance(fragment_index_1, np.ndarray)
                and fragment_index_1.dtype == np.bool_
            )
            fragment_index &= fragment_index_1

        if fragment_charge is not None:
            fragment_index_1 = self.filter_fragments_by_charge(
                spectrum, fragment_charge=fragment_charge, return_index=True
            )
            assert (
                isinstance(fragment_index_1, np.ndarray)
                and fragment_index_1.dtype == np.bool_
            )
            fragment_index &= fragment_index_1

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )

    def filter_fragments_by_type(
        self,
        spectrum: MS2SpectrumType,
        fragment_type: Union[str, Sequence[str], None] = None,
        return_index: bool = False,
    ):
        if fragment_type is None:
            fragment_index = np.ones_like(spectrum.fragment_type, dtype=np.bool_)
        elif isinstance(fragment_type, str):
            fragment_index: npt.NDArray[np.bool_] = (
                spectrum.fragment_type == fragment_type
            )
        else:
            fragment_index = np.isin(spectrum.fragment_type, fragment_type)

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )

    def filter_fragments_by_charge(
        self,
        spectrum: MS2SpectrumType,
        fragment_charge: Union[int, Sequence[int], None] = None,
        return_index: bool = False,
    ):
        if fragment_charge is None:
            fragment_index = np.ones_like(spectrum.fragment_charge, dtype=np.bool_)
        elif isinstance(fragment_charge, int):
            fragment_index: npt.NDArray[np.bool_] = (
                spectrum.fragment_charge == fragment_charge
            )
        else:
            fragment_index = np.isin(spectrum.fragment_charge, fragment_charge)

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )

    def filter_fragments_by_loss_type(
        self,
        spectrum: MS2SpectrumType,
        loss_type: Union[str, Sequence[str], None] = None,
        return_index: bool = False,
    ):
        if loss_type == None:
            fragment_index = np.ones_like(spectrum.loss_type, dtype=np.bool_)
        elif isinstance(loss_type, str):
            fragment_index: npt.NDArray[np.bool_] = spectrum.loss_type == loss_type
        else:
            fragment_index = np.isin(spectrum.loss_type, loss_type)

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )
