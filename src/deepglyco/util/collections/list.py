"""
Extracted from alphapept.io
https://github.com/MannLabs/alphapept/blob/master/alphapept/io.py
under the Apache License 2.0
https://github.com/MannLabs/alphapept/blob/master/LICENSE
"""

__all__ = ["index_ragged_list"]


import functools
from typing import List, Sequence, TypeVar

import numpy as np
import numpy.typing as npt


def index_ragged_list(ragged_list: Sequence) -> npt.NDArray[np.int64]:
    """Create lookup indices for a list of arrays for concatenation.
    Args:
        value (list): Input list of arrays.
    Returns:
        indices: A numpy array with indices.
    """
    indices = np.zeros(len(ragged_list) + 1, np.int64)
    indices[1:] = [len(i) for i in ragged_list]
    indices = np.cumsum(indices)

    return indices


T = TypeVar("T")


def remove_duplicates(r: List[T]) -> List[T]:
    return functools.reduce(
        lambda x, y: x if y in x else x + [y], [[]] + r  # type: ignore
    )
