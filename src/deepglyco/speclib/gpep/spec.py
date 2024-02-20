__all__ = ["GlycoPeptideMS2SpectrumProto", "GlycoPeptideMS2Spectrum"]

import itertools
from typing import Any, Iterable, Optional, Protocol, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from ..common.spec import IntensityArray, MS2SpectrumProto, MzArray
from ..pep.spec import (
    ChargeArray,
    FragmentNumberArray,
    FragmentTypeArray,
    LossTypeArray,
)

FragmentGlycanArray = npt.NDArray[np.unicode_]


class GlycoPeptideMS2SpectrumProto(MS2SpectrumProto, Protocol):
    modified_sequence: str
    glycan_struct: str
    glycan_position: int
    precursor_charge: int

    mz: MzArray
    intensity: IntensityArray
    fragment_charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray
    fragment_glycan: FragmentGlycanArray


@dataclass(frozen=True)
class GlycoPeptideMS2Spectrum(GlycoPeptideMS2SpectrumProto):
    modified_sequence: str
    glycan_struct: str
    glycan_position: int
    precursor_charge: int

    mz: MzArray
    intensity: IntensityArray
    fragment_charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray
    fragment_glycan: FragmentGlycanArray

    precursor_mz: Optional[float] = None
    run_name: Optional[str] = None
    scan_number: Optional[int] = None
    score: Optional[float] = None

    @property
    def num_peaks(self) -> int:
        return self.mz.shape[0]

    def analyte_info(self) -> Iterable[Tuple[str, Any]]:
        yield "modified_sequence", self.modified_sequence
        yield "glycan_struct", self.glycan_struct
        yield "glycan_position", self.glycan_position
        yield "precursor_charge", self.precursor_charge
        yield "precursor_mz", self.precursor_mz

    def spectrum_metadata(self) -> Iterable[Tuple[str, Any]]:
        yield "run_name", self.run_name
        yield "scan_number", self.scan_number
        yield "score", self.score

    def frangment_annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        yield "fragment_charge", self.fragment_charge
        yield "fragment_type", self.fragment_type
        yield "fragment_number", self.fragment_number
        yield "loss_type", self.loss_type
        yield "fragment_glycan", self.fragment_glycan

    def __post_init__(self):
        num_peaks = -1
        for k, f in itertools.chain(
            [("mz", self.mz), ("intensity", self.intensity)],
            self.frangment_annotations(),
        ):
            if len(f.shape) != 1:
                raise ValueError(f"invalid array shape of {k}")
            if num_peaks < 0:
                num_peaks = f.shape[0]
            elif num_peaks != f.shape[0]:
                raise ValueError(f"array length of {k} not match")
