__all__ = ["batched"]

from itertools import islice
from typing import Iterable, Tuple, TypeVar

T = TypeVar('T')

def batched(iterable: Iterable[T], n: int) -> Iterable[Tuple[T, ...]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    # itertools 3.11
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if batch:
            yield batch
        else:
            break

