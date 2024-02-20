__all__ = ["ProgressProto", "ProgressFactoryProto", "TqdmProgressFactory"]

from typing import Generic, Iterable, Protocol, TypeVar, Union, runtime_checkable

import tqdm

T = TypeVar("T", covariant=True)


@runtime_checkable
class ProgressProto(Generic[T], Iterable[T], Protocol):
    def update(self, n: Union[float, None] = 1) -> Union[bool, None]:
        ...

    def reset(self, total=None):
        ...

    def set_description(self, desc=None):
        ...

    def set_postfix(self, ordered_dict=None, **kwargs):
        ...


class ProgressFactoryProto(Protocol):
    def __call__(self, iterable: Iterable[T], *args, **kwds) -> ProgressProto[T]:
        ...


class TqdmProgressFactory:
    def __call__(self, iterable: Iterable[T], *args, **kwds) -> ProgressProto[T]:
        return tqdm.tqdm(iterable, *args, **kwds)
