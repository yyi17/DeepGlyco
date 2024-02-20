__all__ = ["match_spectrum_mz", "MS2SpectrumAnnotatorBase"]

import abc
from typing import Generic, Literal, TypeVar, Union

import numpy as np

from ...chem.common.mz import MzAnnotationProto
from ...specio.spec import MassSpectrum
from ...util.config import Configurable
from .spec import MS2SpectrumProto, MzArray


def match_spectrum_mz(
    spectrum: MassSpectrum,
    target_mz: MzArray,
    tolerance: float = 10,
    tolerance_unit: Literal["ppm", "Da"] = "ppm",
    criteria: Literal["mostintense", "nearst"] = "mostintense",
):
    if tolerance_unit == "ppm":

        def mz_tolerance_range(mz, tolerance):
            return (mz * (1 - tolerance * 1e-6), mz * (1 + tolerance * 1e-6))

    elif tolerance_unit == "Da":

        def mz_tolerance_range(mz, tolerance):
            return (mz - tolerance, mz + tolerance)

    else:
        raise ValueError(f"invalid tolerance unit {tolerance_unit}")

    if criteria == "mostintense":

        def choose_mostintense(target_mz, index1, mz1, index2, mz2):
            return spectrum.intensity[index1] >= spectrum.intensity[index2]

        choose_first = choose_mostintense
    elif criteria == "nearst":

        def choose_nearst(target_mz, index1, mz1, index2, mz2):
            return abs(mz1 - target_mz) <= abs(mz2 - target_mz)

        choose_first = choose_nearst
    else:
        raise ValueError(f"invalid peak matching criteria {criteria}")

    index = []
    used = set()
    for i, x in enumerate(target_mz):
        l, r = mz_tolerance_range(x, tolerance)
        j0 = None
        y0 = None
        for j, y in enumerate(spectrum.mz):
            if y <= r and y >= l and j not in used:
                if j0 is None or not choose_first(
                    target_mz=x, index1=j0, mz1=y0, index2=j, mz2=y
                ):
                    j0 = j
                    y0 = y
        if j0 is not None:
            index.append(j0)
            used.add(j0)
        else:
            index.append(-1)
    return np.array(index)


MS2SpectrumType = TypeVar("MS2SpectrumType", bound=MS2SpectrumProto)


class MS2SpectrumAnnotatorBase(Generic[MS2SpectrumType], Configurable):
    def __init__(
        self,
        configs: Union[str, dict],
    ):
        super().__init__(configs)

    def annotate(
        self,
        spectrum: MassSpectrum,
        **analyte_info,
    ) -> MS2SpectrumType:
        matching_args = (
            self.get_config("peak_matching", required=False, typed=dict) or {}
        )

        frag_mz = self._calculate_fragment_mz(**analyte_info)

        index = match_spectrum_mz(spectrum, frag_mz.mz, **matching_args)
        frags = {k: v[index >= 0] for k, v in frag_mz.annotations()}
        index = index[index >= 0]
        frags["mz"] = spectrum.mz[index]
        frags["intensity"] = spectrum.intensity[index]
        frags["fragment_charge"] = frags.pop("charge")

        precursor_mz = analyte_info.get("precursor_mz", None)
        if precursor_mz is None:
            precursor_charge = analyte_info.get("precursor_charge", None)
            if precursor_charge is not None:
                precursor_mz = self._calculate_precursor_mz(**analyte_info)
                analyte_info["precursor_mz"] = precursor_mz

        return self._create_annotated_spectrum(**analyte_info, **frags)

    @abc.abstractmethod
    def _calculate_fragment_mz(self, **analyte_info) -> MzAnnotationProto:
        pass

    @abc.abstractmethod
    def _calculate_precursor_mz(self, **analyte_info) -> MzAnnotationProto:
        pass

    @abc.abstractmethod
    def _create_annotated_spectrum(self, **analyte_info_fragments) -> MS2SpectrumType:
        pass
