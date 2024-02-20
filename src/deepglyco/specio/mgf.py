__all__ = ["MgfReader"]

import os
import re
from typing import Optional, Union, cast

import numpy as np

from .abs import MassSpectrumReaderBase
from .spec import MassSpectrum


class MgfReader(MassSpectrumReaderBase):
    def __init__(self, file):
        if isinstance(file, str):
            file = open(file, 'r')
        self.reader = file

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
        spectrum: Union[None, dict] = None
        while True:
            line = self.reader.readline()
            if not line:
                if spectrum is not None:
                    raise ValueError('unexpected EOF')
                return None

            line = line.strip()
            if len(line) == 0 or \
                line[0] in {'#', ';', '!', '/'}:
                continue

            if line == 'BEGIN IONS':
                if spectrum is not None:
                    raise ValueError('[Offset ' + str(self.reader.tell()) + \
                        '] invalid format: ' + line)
                spectrum = {
                    'ms_level': 2,
                    'mz': [],
                    'intensity': [],
                }
                continue

            if line == 'END IONS':
                if spectrum is None:
                    raise ValueError('[Offset ' + str(self.reader.tell()) + \
                        '] invalid format: ' + line)
                spectrum['mz'] = np.array(spectrum['mz'], dtype=np.float32)
                spectrum['intensity'] = np.array(spectrum['intensity'], dtype=np.float32)
                return MassSpectrum(**spectrum)

            if spectrum is None:
                continue

            s = line.split('=', 1)
            if len(s) == 2:
                if s[0] == 'TITLE':
                    spectrum_title = s[1]

                    run_match = re.search(r'^(.*)\.([0-9]+)\.[0-9]+\.[0-9]+\.[0-9]+\.dta$', spectrum_title)  # type: ignore
                    if run_match is not None:
                        run_name = run_match.group(1)
                        scan_number = int(run_match.group(2))
                    else:
                        run_name = ""
                        scan_number = 0

                    spectrum['spectrum_name'] = spectrum_title
                    spectrum['run_name'] = run_name
                    spectrum['scan_number'] = scan_number

                continue

            s = line.split(' ', 4)
            if len(s) < 2:
                raise ValueError('[Offset ' + str(self.reader.tell()) + \
                    '] invalid format: ' + line)
            if len(s) >= 3:
                charge = int(s[2].strip('+'))
            else:
                charge = None
            if len(s) == 4:
                annotation = s[3]
            else:
                annotation = None

            spectrum['mz'].append(float(s[0]))
            spectrum['intensity'].append(float(s[1]))
