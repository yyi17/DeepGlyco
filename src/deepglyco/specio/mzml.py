__all__ = ["MzmlReader"]

import os
import re
from typing import Optional, Union, cast

import numpy as np
import pymzml.run
import pymzml.spec

from .abs import MassSpectrumReaderBase
from .spec import MassSpectrum


class MzmlReader(MassSpectrumReaderBase):
    def __init__(self, file):
        self.reader = pymzml.run.Reader(file)

    def __enter__(self):
        self.reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def close(self):
        return self.reader.close()

    def read_spectrum(self) -> Optional[MassSpectrum]:
        spec: Union[None, pymzml.spec.Spectrum, pymzml.spec.Chromatogram]
        while True:
            spec = next(self.reader, None)
            if spec is None:
                return None
            if isinstance(spec, pymzml.spec.Spectrum):
                break

        ms_level = cast(int, spec.ms_level)
        # if self.max_num_peaks and self.max_num_peaks > 0:
        #     peaks = spec.highestPeaks(self.max_num_peaks)
        #     order = np.argsort(peaks[:, 0])
        #     mz = cast(np.ndarray, peaks[order, 0])
        #     intensity = cast(np.ndarray, peaks[order, 1])

        mz = cast(np.ndarray, spec.mz)
        intensity = cast(np.ndarray, spec.i)

        if spec.id_dict:
            scan_number = cast(int, spec.id_dict.get("scan"))
        else:
            scan_number = cast(int, spec.ID)
        spectrum_title = spec.get("MS:1000796", None)
        if spectrum_title is not None:
            run_match = re.search('^(.*) File:"(.*)",', spectrum_title)  # type: ignore
            if run_match is not None:
                spectrum_name = run_match.group(1)
                run_name = os.path.splitext(run_match.group(2))[0]
            else:
                spectrum_name = cast(str, spectrum_title)
                run_name = ""
        else:
            spectrum_name = str(scan_number)
            run_name = ""

        return MassSpectrum(
            mz=mz,
            intensity=intensity,
            ms_level=ms_level,
            spectrum_name=spectrum_name,
            run_name=run_name,
            scan_number=scan_number,
        )
