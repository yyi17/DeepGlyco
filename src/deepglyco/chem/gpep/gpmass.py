__all__ = [
    "GlycoPeptideMassCalculatorBase",
    "GlycoPeptideMassCalculator",
    "GlycanMassAnnotationProto",
    "GlycoPeptideMassAnnotationProto",
    "GlycoPeptideMzAnnotationProto",
    "GlycoPeptideMassAnnotation",
    "GlycoPeptideMzAnnotation",
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
from ..common.mz import ChargeArray, MassArray, MzArray
from ..pep.aminoacids import AminoAcidCollection
from ..pep.fragments import PeptideFragmentTypeCollection
from ..pep.losses import NeutralLossTypeCollection
from ..pep.mods import ModificationCollection, ModifiedSequence
from ..pep.pepmass import DEFAULT_FRAGMENT_CHARGE as DEFAULT_PEPTIDE_FRAGMENT_CHARGE
from ..pep.pepmass import DEFAULT_FRAGMENT_TYPE as DEFAULT_PEPTIDE_FRAGMENT_TYPE
from ..pep.pepmass import DEFAULT_LOSS_TYPE as DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE
from ..pep.pepmass import (
    DEFAULT_PRECURSOR_CHARGE,
    FragmentNumberArray,
    FragmentTypeArray,
    LossTypeArray,
    PeptideMassAnnotationProto,
    PeptideMassCalculatorBase,
    PeptideMzAnnotationProto,
)
from .fragments import GlycanFragmentTypeCollection
from .glycans import GlycanNode, MonosaccharideCollection
from .gmass import (
    FragmentGlycanArray,
    GlycanMassAnnotationProto,
    GlycanMassCalculatorBase,
)


@runtime_checkable
class GlycoPeptideMassAnnotationProto(
    PeptideMassAnnotationProto, GlycanMassAnnotationProto, Protocol
):
    pass


@runtime_checkable
class GlycoPeptideMzAnnotationProto(PeptideMzAnnotationProto, Protocol):
    fragment_glycan: FragmentGlycanArray


GlycoPeptideMassAnnotationType = TypeVar(
    "GlycoPeptideMassAnnotationType", bound=GlycoPeptideMassAnnotationProto
)
GlycoPeptideMzAnnotationType = TypeVar(
    "GlycoPeptideMzAnnotationType", bound=GlycoPeptideMzAnnotationProto
)

DEFAULT_PEPTIDE_FRAGMENT_GLYCAN = ""
DEFAULT_GLYCAN_FRAGMENT_TYPE = "Y"
DEFAULT_GLYCAN_BRANCH_FRAGMENT_CHARGE = 1
DEFAULT_GLYCAN_REDUCING_END_FRAGMENT_CHARGE = (1, 2, 3)


class GlycoPeptideMassCalculatorBase(
    Generic[GlycoPeptideMassAnnotationType, GlycoPeptideMzAnnotationType],
    PeptideMassCalculatorBase[
        GlycoPeptideMassAnnotationType, GlycoPeptideMzAnnotationType
    ],
    GlycanMassCalculatorBase[GlycoPeptideMassAnnotationType],
):
    def __init__(
        self,
        elements: ElementCollection,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        monosaccharides: MonosaccharideCollection,
        neutral_loss_types: NeutralLossTypeCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
    ):
        PeptideMassCalculatorBase.__init__(
            self,
            elements=elements,
            amino_acids=amino_acids,
            modifications=modifications,
            neutral_loss_types=neutral_loss_types,
            peptide_fragment_types=peptide_fragment_types,
        )
        GlycanMassCalculatorBase.__init__(
            self,
            elements=elements,
            monosaccharides=monosaccharides,
            glycan_fragment_types=glycan_fragment_types,
        )

    def glycopeptide_mass(
        self, parsed_sequence: ModifiedSequence, glycan: GlycanNode
    ) -> np.float64:
        mass = super().peptide_mass(parsed_sequence)
        mass += super().glycan_mass(glycan)
        return mass

    @overload
    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        charge: int,
    ) -> np.float64:
        ...

    @overload
    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        charge: Sequence[int] = DEFAULT_PRECURSOR_CHARGE,
    ) -> MzArray:
        ...

    def precursor_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        charge: Union[int, Sequence[int]] = DEFAULT_PRECURSOR_CHARGE,
    ):
        mass = self.glycopeptide_mass(parsed_sequence, glycan)
        return self.mass_to_mz(mass, charge)

    @overload
    def peptide_fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        *,
        keep_fragment_placeholder: bool = False,
    ) -> GlycoPeptideMassAnnotationType:
        ...

    @overload
    def peptide_fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[GlycoPeptideMassAnnotationType, MassArray]:
        ...

    def peptide_fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[GlycoPeptideMassAnnotationType, MassArray]:
        mass = PeptideMassCalculatorBase[
            GlycoPeptideMassAnnotationType, GlycoPeptideMzAnnotationType
        ].fragment_mass(
            self,
            parsed_sequence=parsed_sequence,
            fragment_type=fragment_type,
            loss_type=loss_type,
            return_annotation=return_annotation,
            keep_fragment_placeholder=keep_fragment_placeholder,
        )
        if fragment_glycan == "":
            return mass

        should_squeeze = False
        if isinstance(fragment_glycan, str):
            fragment_glycan = [fragment_glycan]
            should_squeeze = True

        gly_mass = np.zeros(len(fragment_glycan))
        for i, gly in enumerate(fragment_glycan):
            if gly == "":
                gly_mass[i] = 0.0
            elif gly == "$":
                gly_mass[i] = (
                    self.elements["C"].mass * 4
                    + self.elements["H"].mass * 5
                    + self.elements["N"].mass
                    + self.elements["O"].mass
                )
            else:
                gly_mass[i] = self.monosaccharides.mass_from_monosaccharide_composition(
                    gly
                )

        frag_mass = []
        if isinstance(mass, PeptideMassAnnotationProto):
            for i, gly in enumerate(fragment_glycan):
                if gly == "":
                    frag_mass.append(mass.mass)
                    continue
                delta_mass = np.repeat(np.nan, len(mass.mass))
                for ft in np.unique(mass.fragment_type):
                    should_add = mass.fragment_type == ft
                    if self.peptide_fragment_types[ft].n_term:
                        should_add &= mass.fragment_number >= glycan_position
                    else:
                        should_add &= (
                            len(parsed_sequence) - mass.fragment_number
                            < glycan_position
                        )
                    delta_mass[should_add] = gly_mass[i]
                frag_mass.append(mass.mass + delta_mass)
        else:
            for i, gly in enumerate(fragment_glycan):
                if gly == "":
                    frag_mass.append(mass)
                    continue
                delta_mass = np.where(
                    np.arange(1, len(parsed_sequence) >= glycan_position),
                    gly_mass[i],
                    np.nan,
                )
                delta_mass = np.expand_dims(
                    delta_mass, axis=tuple(range(1 - len(mass.shape), 0))
                )
                frag_mass.append(mass + delta_mass)

        frag_mass = np.stack(frag_mass, axis=-1)
        if isinstance(mass, PeptideMassAnnotationProto):
            frag_mass = frag_mass.flatten(order="F")
            result = {
                "mass": frag_mass,
                **{
                    k: np.tile(v, len(fragment_glycan))
                    for k, v in mass.annotations()
                    if k != "fragment_glycan"
                },
                "fragment_glycan": np.tile(
                    fragment_glycan, (len(mass.mass), 1)
                ).flatten(order="F"),
            }
            if not keep_fragment_placeholder:
                not_nan = ~np.isnan(frag_mass)
                result = {k: v[not_nan] for k, v in result.items()}
            return self._create_mass_annotation(**result)
        else:
            if should_squeeze:
                frag_mass = frag_mass.squeeze(axis=-1)
            return frag_mass

    def glycan_fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_GLYCAN_FRAGMENT_TYPE,
    ) -> GlycoPeptideMassAnnotationType:
        gly_mass = GlycanMassCalculatorBase[
            GlycoPeptideMassAnnotationType
        ].fragment_mass(
            self,
            glycan=glycan,
            fragment_type=fragment_type,
        )
        pep_mass = None

        mass = gly_mass.mass.copy()
        annotations = dict(gly_mass.annotations())
        nakedpep_mass = []
        nakedpep_annotations = {k: [] for k in annotations.keys()}

        if isinstance(fragment_type, str):
            fragment_type = [fragment_type]

        for ft in fragment_type:
            frag_type_info = self.glycan_fragment_types[ft]
            if frag_type_info.reducing_end:
                if pep_mass is None:
                    pep_mass = super().peptide_mass(parsed_sequence)

                mass[gly_mass.fragment_type == ft] += pep_mass

                nakedpep_mass.append(pep_mass + frag_type_info.mass)
                for k, v in annotations.items():
                    if k == "fragment_glycan":
                        nakedpep_annotations[k].append("")
                    else:
                        nakedpep_annotations[k].append(
                            v[gly_mass.fragment_type == ft][0]
                        )

        if len(nakedpep_mass) > 0:
            mass = np.concatenate((nakedpep_mass, mass))
            for k, v in nakedpep_annotations.items():
                annotations[k] = np.concatenate((v, annotations[k]))

        return self._create_mass_annotation(mass=mass, **annotations)

    def fragment_mass(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        glycan_position: int,
        peptide_fragment_type: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        peptide_fragment_loss_type: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        peptide_fragment_glycan: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        glycan_fragment_type: Union[str, Sequence[str]] = DEFAULT_GLYCAN_FRAGMENT_TYPE,
    ) -> GlycoPeptideMassAnnotationType:
        if len(peptide_fragment_type) > 0:
            pepfrags = self.peptide_fragment_mass(
                parsed_sequence=parsed_sequence,
                glycan_position=glycan_position,
                fragment_type=peptide_fragment_type,
                loss_type=peptide_fragment_loss_type,
                fragment_glycan=peptide_fragment_glycan,
            )
        else:
            pepfrags = None

        if len(glycan_fragment_type) > 0:
            glyfrags = self.glycan_fragment_mass(
                parsed_sequence=parsed_sequence,
                glycan=glycan,
                fragment_type=glycan_fragment_type,
            )
        else:
            glyfrags = None

        if pepfrags is None:
            if glyfrags is None:
                raise ValueError(
                    "neither peptide_fragment_type nor glycan_fragment_type are specified"
                )
            return glyfrags
        elif glyfrags is None:
            return pepfrags

        return self._create_mass_annotation(
            mass=np.concatenate((pepfrags.mass, glyfrags.mass)),
            fragment_type=np.concatenate(
                (pepfrags.fragment_type, glyfrags.fragment_type)
            ),
            fragment_number=np.concatenate(
                (pepfrags.fragment_number, glyfrags.fragment_number)
            ),
            loss_type=np.concatenate((pepfrags.loss_type, glyfrags.loss_type)),
            fragment_glycan=np.concatenate(
                (pepfrags.fragment_glycan, glyfrags.fragment_glycan)
            ),
        )

    @overload
    def peptide_fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        fragment_charge: Union[int, Sequence[int]] = DEFAULT_PEPTIDE_FRAGMENT_CHARGE,
        *,
        keep_fragment_placeholder: bool = False,
    ) -> GlycoPeptideMzAnnotationType:
        ...

    @overload
    def peptide_fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        fragment_charge: Union[int, Sequence[int]] = DEFAULT_PEPTIDE_FRAGMENT_CHARGE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[GlycoPeptideMzAnnotationType, MzArray]:
        ...

    def peptide_fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan_position: int,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        loss_type: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        fragment_glycan: Union[str, Sequence[str]] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        fragment_charge: Union[int, Sequence[int]] = DEFAULT_PEPTIDE_FRAGMENT_CHARGE,
        *,
        return_annotation: bool = True,
        keep_fragment_placeholder: bool = False,
    ) -> Union[GlycoPeptideMzAnnotationType, MzArray]:
        pepfrags = self.peptide_fragment_mass(
            parsed_sequence=parsed_sequence,
            glycan_position=glycan_position,
            fragment_type=fragment_type,
            loss_type=loss_type,
            fragment_glycan=fragment_glycan,
            return_annotation=return_annotation,
            keep_fragment_placeholder=keep_fragment_placeholder,
        )
        pepfrags = self.mass_to_mz(pepfrags, fragment_charge)
        return pepfrags

    def glycan_fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_GLYCAN_FRAGMENT_TYPE,
        fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = None,
        reducing_end_fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = DEFAULT_GLYCAN_REDUCING_END_FRAGMENT_CHARGE,
        branch_fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = DEFAULT_GLYCAN_BRANCH_FRAGMENT_CHARGE,
    ) -> GlycoPeptideMzAnnotationType:
        glyfrags = self.glycan_fragment_mass(
            parsed_sequence=parsed_sequence,
            glycan=glycan,
            fragment_type=fragment_type,
        )

        def get_same_glycan_fragment_charge():
            if fragment_charge is not None:
                return fragment_charge
            elif (
                reducing_end_fragment_charge is None and branch_fragment_charge is None
            ):
                raise ValueError(
                    "None of glycan fragment_charge, reducing_end_fragment_charge, and branch_fragment_charge is specified"
                )
            elif reducing_end_fragment_charge == branch_fragment_charge:
                return reducing_end_fragment_charge

        def get_reducing_end_glycan_fragment_charge():
            if reducing_end_fragment_charge is not None:
                return reducing_end_fragment_charge
            if fragment_charge is None:
                raise ValueError(
                    "Neither of glycan fragment_charge nor reducing_end_fragment_charge is specified"
                )
            else:
                return fragment_charge

        def get_branch_glycan_fragment_charge():
            if branch_fragment_charge is not None:
                return branch_fragment_charge
            if fragment_charge is None:
                raise ValueError(
                    "Neither of glycan fragment_charge nor branch_fragment_charge is specified"
                )
            else:
                return fragment_charge

        same_glycan_fragment_charge = get_same_glycan_fragment_charge()
        if same_glycan_fragment_charge is not None:
            glyfrags = self.mass_to_mz(glyfrags, same_glycan_fragment_charge)
        elif isinstance(fragment_type, str):
            if self.glycan_fragment_types[fragment_type].reducing_end:
                glyfrags = self.mass_to_mz(
                    glyfrags, get_reducing_end_glycan_fragment_charge()
                )
            else:
                glyfrags = self.mass_to_mz(
                    glyfrags, get_branch_glycan_fragment_charge()
                )
        else:
            reducing_end_fragment_types = list(
                ft
                for ft in fragment_type
                if self.glycan_fragment_types[ft].reducing_end
            )
            reducing_end_index = np.isin(
                glyfrags.fragment_type, reducing_end_fragment_types
            )
            if np.all(reducing_end_index):
                glyfrags = self.mass_to_mz(
                    glyfrags, get_reducing_end_glycan_fragment_charge()
                )
            elif not np.any(reducing_end_index):
                glyfrags = self.mass_to_mz(
                    glyfrags, get_branch_glycan_fragment_charge()
                )
            else:
                reducing_end_frags = self.mass_to_mz(
                    self._create_mass_annotation(
                        mass=glyfrags.mass[reducing_end_index],
                        **{k: v[reducing_end_index] for k, v in glyfrags.annotations()},
                    ),
                    get_reducing_end_glycan_fragment_charge(),
                )
                branch_frags = self.mass_to_mz(
                    self._create_mass_annotation(
                        mass=glyfrags.mass[~reducing_end_index],
                        **{
                            k: v[~reducing_end_index] for k, v in glyfrags.annotations()
                        },
                    ),
                    get_branch_glycan_fragment_charge(),
                )
                glyfrags = self._create_mz_annotation(
                    mz=np.concatenate((reducing_end_frags.mz, branch_frags.mz)),
                    **{
                        k1: np.concatenate((v1, v2))
                        for (k1, v1), (k2, v2) in zip(
                            reducing_end_frags.annotations(),
                            branch_frags.annotations(),
                        )
                    },
                )
        return glyfrags

    def fragment_mz(
        self,
        parsed_sequence: ModifiedSequence,
        glycan: GlycanNode,
        glycan_position: int,
        peptide_fragment_type: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_TYPE,
        peptide_fragment_loss_type: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_LOSS_TYPE,
        peptide_fragment_glycan: Union[
            str, Sequence[str]
        ] = DEFAULT_PEPTIDE_FRAGMENT_GLYCAN,
        peptide_fragment_charge: Union[
            int, Sequence[int]
        ] = DEFAULT_PEPTIDE_FRAGMENT_CHARGE,
        glycan_fragment_type: Union[str, Sequence[str]] = DEFAULT_GLYCAN_FRAGMENT_TYPE,
        glycan_fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = None,
        glycan_reducing_end_fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = DEFAULT_GLYCAN_REDUCING_END_FRAGMENT_CHARGE,
        glycan_branch_fragment_charge: Union[
            int,
            Sequence[int],
            None,
        ] = DEFAULT_GLYCAN_BRANCH_FRAGMENT_CHARGE,
    ) -> GlycoPeptideMzAnnotationType:
        if len(peptide_fragment_type) > 0:
            pepfrags = self.peptide_fragment_mz(
                parsed_sequence=parsed_sequence,
                glycan_position=glycan_position,
                fragment_type=peptide_fragment_type,
                loss_type=peptide_fragment_loss_type,
                fragment_glycan=peptide_fragment_glycan,
                fragment_charge=peptide_fragment_charge,
            )
        else:
            pepfrags = None

        if len(glycan_fragment_type) > 0:
            glyfrags = self.glycan_fragment_mz(
                parsed_sequence=parsed_sequence,
                glycan=glycan,
                fragment_type=glycan_fragment_type,
                fragment_charge=glycan_fragment_charge,
                reducing_end_fragment_charge=glycan_reducing_end_fragment_charge,
                branch_fragment_charge=glycan_branch_fragment_charge,
            )
        else:
            glyfrags = None

        if pepfrags is None:
            if glyfrags is None:
                raise ValueError(
                    "neither peptide_fragment_type nor glycan_fragment_type are specified"
                )
            return glyfrags
        elif glyfrags is None:
            return pepfrags

        return self._create_mz_annotation(
            mz=np.concatenate((pepfrags.mz, glyfrags.mz)),
            **{
                k1: np.concatenate((v1, v2))
                for (k1, v1), (k2, v2) in zip(
                    pepfrags.annotations(),
                    glyfrags.annotations(),
                )
            },
        )

    @abc.abstractmethod
    def _create_mass_annotation(
        self, mass: MassArray, **annotations: npt.NDArray
    ) -> GlycoPeptideMassAnnotationType:
        pass

    @abc.abstractmethod
    def _create_mz_annotation(
        self, mz: MzArray, charge: ChargeArray, **annotations: npt.NDArray
    ) -> GlycoPeptideMzAnnotationType:
        pass


class GlycoPeptideMassAnnotation(NamedTuple):
    mass: MassArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray
    fragment_glycan: FragmentGlycanArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        yield ("fragment_type", self.fragment_type)
        yield ("fragment_number", self.fragment_number)
        yield ("loss_type", self.loss_type)
        yield ("fragment_glycan", self.fragment_glycan)


class GlycoPeptideMzAnnotation(NamedTuple):
    mz: MzArray
    charge: ChargeArray
    fragment_type: FragmentTypeArray
    fragment_number: FragmentNumberArray
    loss_type: LossTypeArray
    fragment_glycan: FragmentGlycanArray

    def annotations(self) -> Iterable[Tuple[str, npt.NDArray]]:
        yield ("charge", self.charge)
        yield ("fragment_type", self.fragment_type)
        yield ("fragment_number", self.fragment_number)
        yield ("loss_type", self.loss_type)
        yield ("fragment_glycan", self.fragment_glycan)


class GlycoPeptideMassCalculator(
    GlycoPeptideMassCalculatorBase[GlycoPeptideMassAnnotation, GlycoPeptideMzAnnotation]
):
    def __init__(
        self,
        elements: ElementCollection,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        monosaccharides: MonosaccharideCollection,
        neutral_loss_types: NeutralLossTypeCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
    ):
        super().__init__(
            elements=elements,
            amino_acids=amino_acids,
            modifications=modifications,
            monosaccharides=monosaccharides,
            neutral_loss_types=neutral_loss_types,
            peptide_fragment_types=peptide_fragment_types,
            glycan_fragment_types=glycan_fragment_types,
        )

    def _create_mass_annotation(
        self, mass, **annotations
    ) -> GlycoPeptideMassAnnotation:
        if "fragment_number" not in annotations:
            annotations["fragment_number"] = -np.ones_like(mass, dtype=np.int16)
        if "loss_type" not in annotations:
            annotations["loss_type"] = np.zeros_like(mass, dtype=np.unicode_)
        if "fragment_glycan" not in annotations:
            annotations["fragment_glycan"] = np.zeros_like(mass, dtype=np.unicode_)
        return GlycoPeptideMassAnnotation(mass=mass, **annotations)

    def _create_mz_annotation(
        self, mz: MzArray, charge: ChargeArray, **annotations: npt.NDArray
    ) -> GlycoPeptideMzAnnotation:
        if "fragment_number" not in annotations:
            annotations["fragment_number"] = -np.ones_like(mz, dtype=np.int16)
        if "loss_type" not in annotations:
            annotations["loss_type"] = np.zeros_like(mz, dtype=np.unicode_)
        if "fragment_glycan" not in annotations:
            annotations["fragment_glycan"] = np.zeros_like(mz, dtype=np.unicode_)
        return GlycoPeptideMzAnnotation(mz=mz, charge=charge, **annotations)
