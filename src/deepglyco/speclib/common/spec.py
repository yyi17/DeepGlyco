__all__ = [
    "MS2SpectrumProto",
    "MzArray",
    "IntensityArray",
]

from typing import Any, Iterable, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt

MzArray = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
IntensityArray = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]


class MS2SpectrumProto(Protocol):
    mz: MzArray
    intensity: IntensityArray

    precursor_mz: Optional[float] = None
    run_name: Optional[str] = None
    scan_number: Optional[int] = None
    score: Optional[float] = None

    @property
    def num_peaks(self) -> int:
        ...

    def analyte_info(self) -> Iterable[Tuple[str, Any]]:
        ...

    def spectrum_metadata(self) -> Iterable[Tuple[str, Any]]:
        ...

    def frangment_annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        ...
