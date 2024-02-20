__all__ = [
    "PeptideMassCalculatorBase",
    "PeptideMassCalculator",
    "PeptideMassAnnotationProto",
    "PeptideMzAnnotationProto",
    "PeptideMassAnnotation",
    "PeptideMzAnnotation",
    "FragmentTypeArray",
    "FragmentNumberArray",
    "LossTypeArray",
]

import abc
from typing import (
    Generic,
    Iterable,
    NamedTuple,
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

from ..common.elements import ElementCollection
from ..common.mz import (
    ChargeArray,
    MassAnnotationProto,
    MassArray,
    MzAnnotationProto,
    MzArray,
    MzCalculatorBase,
)
from .aminoacids import AminoAcidCollection
from .fragments import PeptideFragmentTypeCollection
from .losses import NeutralLossTypeCollection
from .mods import ModificationCollection, ModifiedSequence

FragmentTypeArray = npt.NDArray[np.unicode_]
FragmentNumberArray = npt.NDArray[np.int16]
LossTypeArray = npt.NDArray[np.unicode_]


@runtime_checkable
class PeptideMassAnnotationProto(MassAnnotationProto, Protocol):
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray


@runtime_checkable
class PeptideMzAnnotationProto(MzAnnotationProto, Protocol):
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray


PeptideMassAnnotationType = TypeVar(
    "PeptideMassAnnotationType", bound=PeptideMassAnnotationProto
)
PeptideMzAnnotationType = TypeVar(
    "PeptideMzAnnotationType", bound=PeptideMzAnnotationProto
)

DEFAULT_FRAGMENT_TYPE = ("b", "y")
DEFAULT_LOSS_TYPE = ""
DEFAULT_PRECURSOR_CHARGE = (2, 3)
DEFAULT_FRAGMENT_CHARGE = (1, 2)


class PeptideMassCalculatorBase(
    Generic[PeptideMassAnnotationType, PeptideMzAnnotationType],
    MzCalculatorBase[PeptideMzAnnotationType],
):
    def __init__(
        self,
        elements: ElementCollection,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        neutral_loss_types: NeutralLossTypeCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
    ):
        super().__init__(elements=elements)
        self.amino_acids = amino_acids
        self.modifications = modifications
        self.neutral_loss_types = neutral_loss_types
        self.peptide_fragment_types = peptide_fragment_types

    def peptide_mass(self, parsed_sequence: ModifiedSequence) -> np.float64:
        mass = np.float64(0.0)
        for aa, mod in parsed_sequence:
            mass += self.amino_acids[aa].mass
            if mod != "":
                mass += self.modifications[mod].mass
        mass += self.elements["H"].mass * 2 + self.elements["O"].mass
        return mass

    @overload
    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        charge: int,
    ) -> np.float64:
        ...

    @overload
    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        charge: Sequence[int] = DEFAULT_PRECURSOR_CHARGE,
    ) -> MzArray:
        ...

    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        charge: Union[int, Sequence[int]] = DEFAULT_PRECURSOR_CHARGE,
    ):
        mass = self.peptide_mass(parsed_sequence)
        return self.mass_to_mz(mass, charge)

    @overload
    def fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        *,
        keep_fragment_placeholder: bool = False,
    ) -> PeptideMassAnnotationType:
        ...

    @overload
    def fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[PeptideMassAnnotationType, MassArray]:
        ...

    def fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[PeptideMassAnnotationType, MassArray]:
        should_squeeze = []
        if isinstance(fragment_type, str):
            fragment_type = [fragment_type]
            should_squeeze.append(1)
        if isinstance(loss_type, str):
            loss_type = [loss_type]
            should_squeeze.append(2)

        b_mass = self._cumulative_mass(parsed_sequence)
        frag_mass = np.zeros(
            (len(b_mass) - 1, len(fragment_type), len(loss_type)), dtype=np.float64
        )
        n_term = np.zeros(len(fragment_type), np.bool_)
        for i, ft in enumerate(fragment_type):
            frag_type_info = self.peptide_fragment_types[ft]
            n_term[i] = frag_type_info.n_term
            if n_term[i]:
                frag_mass_noloss = b_mass[:-1] + frag_type_info.mass
            else:
                frag_mass_noloss = b_mass[-1] - b_mass[:-1] + frag_type_info.mass
            for j, lt in enumerate(loss_type):
                if lt == "" or lt == "noloss" or lt == "none":
                    frag_mass[:, i, j] = frag_mass_noloss
                else:
                    frag_mass[:, i, j] = frag_mass_noloss - self._neutral_loss_mass(
                        parsed_sequence,
                        lt,
                        frag_type_info.n_term,
                    )

        if return_annotation:
            frag_num = [
                np.arange(1, frag_mass.shape[0] + 1, dtype=np.int16)
                if n_term[i]
                else np.arange(frag_mass.shape[0], 0, -1, dtype=np.int16)
                for i in range(frag_mass.shape[1])
            ]
            frag_num = np.tile(
                np.expand_dims(np.swapaxes(frag_num, 0, 1), 2), len(loss_type)
            )
            frag_mass = frag_mass.flatten(order="F")
            frag_type = np.tile(
                np.expand_dims(fragment_type, (0, 2)),
                (len(b_mass) - 1, 1, len(loss_type)),
            ).flatten(order="F")
            frag_num = frag_num.flatten(order="F")
            frag_loss_type = np.tile(
                np.expand_dims(loss_type, (0, 1)),
                (len(b_mass) - 1, len(fragment_type), 1),
            ).flatten(order="F")
            if not keep_fragment_placeholder:
                not_nan = ~np.isnan(frag_mass)
                frag_mass = frag_mass[not_nan]
                frag_type = frag_type[not_nan]
                frag_num = frag_num[not_nan]
                frag_loss_type = frag_loss_type[not_nan]
            return self._create_mass_annotation(
                mass=frag_mass,
                fragment_type=frag_type,
                fragment_number=frag_num,
                loss_type=frag_loss_type,
            )

        if should_squeeze:
            frag_mass = frag_mass.squeeze(axis=tuple(should_squeeze))
        return frag_mass

    @overload
    def fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        charge: Union[int, Sequence[int]] = DEFAULT_FRAGMENT_CHARGE,
        *,
        keep_fragment_placeholder: bool = False,
    ) -> PeptideMzAnnotationType:
        ...

    @overload
    def fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        charge: Union[int, Sequence[int]] = DEFAULT_FRAGMENT_CHARGE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[PeptideMzAnnotationType, MassArray]:
        ...

    def fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_LOSS_TYPE,
        charge: Union[int, Sequence[int]] = DEFAULT_FRAGMENT_CHARGE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[PeptideMzAnnotationType, MzArray]:
        mass = self.fragment_mass(
            parsed_sequence=parsed_sequence,
            fragment_type=fragment_type,
            loss_type=loss_type,
            return_annotation=return_annotation,
            keep_fragment_placeholder=keep_fragment_placeholder,
        )
        return self.mass_to_mz(mass, charge)

    def _cumulative_mass(self, parsed_sequence: ModifiedSequence):
        frag_mass = np.zeros(len(parsed_sequence), dtype=np.float64)
        sum_mass = 0.0
        for i, (aa, mod) in enumerate(parsed_sequence):
            sum_mass += self.amino_acids[aa].mass
            if mod != "":
                sum_mass += self.modifications[mod].mass
            frag_mass[i] = sum_mass
        return frag_mass

    def _neutral_loss_mass(
        self,
        parsed_sequence: ModifiedSequence,
        loss_type: str,
        n_term: bool,
    ):
        loss_info = self.neutral_loss_types[loss_type]
        loss_mass = np.zeros(len(parsed_sequence) - 1, dtype=np.float64)
        if not loss_info.mod_specific:
            loss_mass += loss_info.mass
            return loss_mass

        has_mod = False
        for i in range(len(parsed_sequence) - 1):
            if n_term:
                j = i
                mod = parsed_sequence[i][1]
            else:
                j = len(parsed_sequence) - 2 - i
                mod = parsed_sequence[j + 1][1]
            if has_mod:
                loss_mass[j] = loss_info.mass
                continue

            if mod != "":
                mod_info = self.modifications[mod]
                if mod_info.loss == loss_type:
                    has_mod = True
                    loss_mass[j] = loss_info.mass
                    continue
            loss_mass[j] = np.nan
        return loss_mass

    @abc.abstractmethod
    def _create_mass_annotation(
        self, mass: MassArray, **annotations: npt.NDArray
    ) -> PeptideMassAnnotationType:
        pass

    @abc.abstractmethod
    def _create_mz_annotation(
        self, mz: MzArray, charge: ChargeArray, **annotations: npt.NDArray
    ) -> PeptideMzAnnotationType:
        pass


class PeptideMassAnnotation(NamedTuple):
    mass: MassArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        yield ("fragment_type", self.fragment_type)
        yield ("fragment_number", self.fragment_number)
        yield ("loss_type", self.loss_type)


class PeptideMzAnnotation(NamedTuple):
    mz: MzArray
    charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        yield ("charge", self.charge)
        yield ("fragment_type", self.fragment_type)
        yield ("fragment_number", self.fragment_number)
        yield ("loss_type", self.loss_type)


class PeptideMassCalculator(
    PeptideMassCalculatorBase[PeptideMassAnnotation, PeptideMzAnnotation]
):
    def __init__(
        self,
        elements: ElementCollection,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        neutral_loss_types: NeutralLossTypeCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
    ):
        super().__init__(
            elements=elements,
            amino_acids=amino_acids,
            modifications=modifications,
            neutral_loss_types=neutral_loss_types,
            peptide_fragment_types=peptide_fragment_types,
        )

    def _create_mass_annotation(self, mass, **annotations) -> PeptideMassAnnotation:
        return PeptideMassAnnotation(mass=mass, **annotations)

    def _create_mz_annotation(self, mz, charge, **annotations) -> PeptideMzAnnotation:
        return PeptideMzAnnotation(mz=mz, charge=charge, **annotations)
