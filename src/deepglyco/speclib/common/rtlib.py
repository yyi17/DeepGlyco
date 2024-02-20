__all__ = [
    "RetentionTimeLibraryBase",
    "IndicesNonSlice",
    "Indices",
    "Columns",
]

import abc
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

IndicesNonSlice = Union[List[int], npt.NDArray[np.int_], "pd.Series[int]"]
Indices = Union[slice, IndicesNonSlice]
Columns = List[str]


class RetentionTimeLibraryBase(abc.ABC):
    @abc.abstractmethod
    def get_retention_time(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_retention_time_data(
        self, indices: Optional[Indices] = None
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_ion_mobility(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @property
    def num_retention_time(self) -> int:
        data = self.get_retention_time()
        if data is not None:
            return len(data)
        else:
            return 0

    @property
    def num_ion_mobility(self) -> int:
        data = self.get_ion_mobility()
        if data is not None:
            return len(data)
        else:
            return 0

    @abc.abstractmethod
    def import_retention_time(self, retention_time: pd.DataFrame):
        pass
