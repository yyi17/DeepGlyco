__all__ = ["MassSpectrum"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

MassArray = npt.NDArray[np.float32]
IntensityArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class MassSpectrum:
    mz: MassArray
    intensity: IntensityArray
    ms_level: int
    run_name: str
    scan_number: int
    spectrum_name: str

    @property
    def num_peaks(self):
        return self.mz.shape[0]

    def __post_init__(self):
        num_peaks = -1
        for f in [
            self.mz,
            self.intensity,
        ]:
            if len(f.shape) != 1:
                raise ValueError("invalid array shape")
            if num_peaks < 0:
                num_peaks = f.shape[0]
            elif num_peaks != f.shape[0]:
                raise ValueError("array length not match")
