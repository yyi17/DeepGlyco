__all__ = [
    "GlycanMassCalculatorBase",
    "GlycanMassAnnotationProto",
    "FragmentTypeArray",
    "FragmentGlycanArray",
]

import abc
from typing import Generic, Protocol, Sequence, TypeVar, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..common.elements import ElementCollection
from ..common.mz import MassAnnotationProto, MassArray
from .fragments import GlycanFragmentTypeCollection, branch_fragments, reducing_end_fragments
from .glycans import GlycanNode, MonosaccharideCollection

FragmentTypeArray = npt.NDArray[np.unicode_]
FragmentGlycanArray = npt.NDArray[np.unicode_]


@runtime_checkable
class GlycanMassAnnotationProto(MassAnnotationProto, Protocol):
    fragment_type: FragmentTypeArray
    fragment_glycan: FragmentGlycanArray


GlycanMassAnnotationType = TypeVar(
    "GlycanMassAnnotationType", bound=GlycanMassAnnotationProto
)


DEFAULT_FRAGMENT_TYPE = "Y"


class GlycanMassCalculatorBase(Generic[GlycanMassAnnotationType]):
    def __init__(
        self,
        elements: ElementCollection,
        monosaccharides: MonosaccharideCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
    ):
        self.elements = elements
        self.monosaccharides = monosaccharides
        self.glycan_fragment_types = glycan_fragment_types

    def glycan_mass(self, glycan: GlycanNode) -> np.float64:
        mass = np.float64(0.0)
        for k, v in glycan.composition().items():
            mass += self.monosaccharides[k].mass * v
        return mass

    def fragment_mass(
        self,
        glycan: GlycanNode,
        fragment_type: Union[str, Sequence[str]] = DEFAULT_FRAGMENT_TYPE,
    ) -> GlycanMassAnnotationType:

        if isinstance(fragment_type, str):
            fragment_type = [fragment_type]

        reducing_end_mass = None
        reducing_end_glycan = None

        branch_mass = None
        branch_glycan = None

        frag_mass = []
        frag_type = []
        frag_glycan = []

        for ft in fragment_type:
            frag_type_info = self.glycan_fragment_types[ft]

            if frag_type_info.reducing_end:
                if reducing_end_mass is None or reducing_end_glycan is None:
                    frags = reducing_end_fragments(glycan)
                    reducing_end_mass = np.array(
                        [
                            self.monosaccharides.mass_from_monosaccharide_composition(
                                gly
                            )
                            for gly in frags
                        ]
                    )
                    reducing_end_glycan = np.array(
                        [self.monosaccharides.composition_str(c) for c in frags]
                    )

                frag_mass.append(reducing_end_mass + frag_type_info.mass)
                frag_type.append(np.repeat(ft, len(reducing_end_mass)))
                frag_glycan.append(reducing_end_glycan)
            else:
                if branch_mass is None or branch_glycan is None:
                    frags = branch_fragments(glycan)
                    branch_mass = np.array(
                        [
                            self.monosaccharides.mass_from_monosaccharide_composition(
                                gly
                            )
                            for gly in frags
                        ]
                    )
                    branch_glycan = np.array(
                        [self.monosaccharides.composition_str(c) for c in frags]
                    )

                frag_mass.append(branch_mass + frag_type_info.mass)
                frag_type.append(np.repeat(ft, len(branch_mass)))
                frag_glycan.append(branch_glycan)

        if len(frag_mass) > 0:
            frag_mass = np.concatenate(frag_mass)
            frag_type = np.concatenate(frag_type)
            frag_glycan = np.concatenate(frag_glycan)
        else:
            frag_mass = np.array([], dtype=np.float64)
            frag_type = np.array([], dtype=np.unicode_)
            frag_glycan = np.array([], dtype=np.unicode_)

        return self._create_mass_annotation(
            mass=frag_mass,
            fragment_type=frag_type,
            fragment_glycan=frag_glycan,
        )

    @abc.abstractmethod
    def _create_mass_annotation(
        self, mass: MassArray, **annotations: npt.NDArray
    ) -> GlycanMassAnnotationType:
        pass
