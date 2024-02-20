__all__ = ["MassSpectrumReaderBase"]

import abc
from typing import Optional

from .spec import MassSpectrum


class MassSpectrumReaderBase(abc.ABC):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __next__(self) -> MassSpectrum:
        spec = self.read_spectrum()
        if spec is None:
            raise StopIteration()
        return spec

    def __iter__(self):
        return self

    def close(self):
        pass

    @abc.abstractmethod
    def read_spectrum(self) -> Optional[MassSpectrum]:
        pass
