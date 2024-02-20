__all__ = [
    "MzCalculatorBase",
    "MassAnnotationProto",
    "MzAnnotationProto",
    "MassArray",
    "MzArray",
    "ChargeArray",
]

import abc
from typing import (
    Generic,
    Iterable,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from .elements import ElementCollection

MassArray = npt.NDArray[np.float64]
MzArray = npt.NDArray[np.float64]
ChargeArray = npt.NDArray[np.int32]


@runtime_checkable
class MassAnnotationProto(Protocol):
    mass: MassArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        ...


@runtime_checkable
class MzAnnotationProto(Protocol):
    mz: MzArray
    charge: ChargeArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        ...


MzAnnotationType = TypeVar("MzAnnotationType", bound=MzAnnotationProto)


class MzCalculatorBase(abc.ABC, Generic[MzAnnotationType]):
    def __init__(
        self,
        elements: ElementCollection,
    ):
        self.elements = elements

    @overload
    def mass_to_mz(
        self,
        mass: np.float64,
        charge: int,
    ) -> np.float64:
        ...

    @overload
    def mass_to_mz(
        self,
        mass: np.float64,
        charge: Sequence[int],
    ) -> MzArray:
        ...

    @overload
    def mass_to_mz(
        self,
        mass: MassArray,
        charge: Union[int, Sequence[int]],
    ) -> MzArray:
        ...

    @overload
    def mass_to_mz(
        self,
        mass: MassAnnotationProto,
        charge: Union[int, Sequence[int]],
    ) -> MzAnnotationType:
        ...

    def mass_to_mz(
        self,
        mass: Union[np.float64, MassArray, MassAnnotationProto],
        charge: Union[int, Sequence[int]],
    ) -> Union[np.float64, MzArray, MzAnnotationType]:
        if isinstance(mass, MassAnnotationProto):
            result = {}
            result["mz"] = self.mass_to_mz(mass.mass, charge).flatten(order="F")
            if isinstance(charge, int):
                result.update(mass.annotations())
                result["charge"] = np.repeat(charge, len(result["mz"]))
            else:
                result.update(
                    {k: np.tile(v, len(charge)) for k, v in mass.annotations()}
                )
                result["charge"] = np.tile(charge, (len(mass.mass), 1)).flatten(
                    order="F"
                )
            return self._create_mz_annotation(**result)
        else:
            p = self.elements["proton"].mass
            if isinstance(charge, Sequence) or np.iterable(charge):
                result = np.array([(mass + ch * p) / np.abs(ch) for ch in charge])
                result = np.moveaxis(result, 0, -1)
                return result
            else:
                return (mass + charge * p) / np.abs(charge)

    @abc.abstractmethod
    def _create_mz_annotation(
        self, mz: MzArray, charge: ChargeArray, **annotations: npt.NDArray
    ) -> MzAnnotationType:
        pass
