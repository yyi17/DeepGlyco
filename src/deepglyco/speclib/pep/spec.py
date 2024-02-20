__all__ = ["PeptideMS2SpectrumProto", "PeptideMS2Spectrum"]

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..common.spec import IntensityArray, MzArray, MS2SpectrumProto

ChargeArray = Union[npt.NDArray[np.int8], npt.NDArray[np.int16], npt.NDArray[np.int32]]
FragmentNumberArray = ChargeArray
FragmentTypeArray = npt.NDArray[np.unicode_]
LossTypeArray = FragmentTypeArray


class PeptideMS2SpectrumProto(MS2SpectrumProto, Protocol):
    modified_sequence: str
    precursor_charge: int
    fragment_charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray


@dataclass(frozen=True)
class PeptideMS2Spectrum(PeptideMS2SpectrumProto):
    modified_sequence: str
    precursor_charge: int

    mz: MzArray
    intensity: IntensityArray
    fragment_charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray

    precursor_mz: Optional[float] = None
    run_name: Optional[str] = None
    scan_number: Optional[int] = None
    score: Optional[float] = None

    @property
    def num_peaks(self) -> int:
        return self.mz.shape[0]

    def analyte_info(self) -> Iterable[Tuple[str, Any]]:
        yield "modified_sequence", self.modified_sequence
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
