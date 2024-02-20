__all__ = [
    "PeptideFragmentTypeInfo",
    "PeptideFragmentTypeCollection",
]

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection, ElementComposition


@dataclass(frozen=True)
class PeptideFragmentTypeInfo:
    name: str
    n_term: bool = True
    mass: float = 0.0
    composition: Optional[ElementComposition] = None


class PeptideFragmentTypeCollection(Dict[str, PeptideFragmentTypeInfo]):
    def __init__(self, peptide_fragment_map: Mapping[str, PeptideFragmentTypeInfo]):
        super().__init__(**peptide_fragment_map)
        for key, ft in peptide_fragment_map.items():
            self.check_fragment_type_id(key, ft)

    @classmethod
    def load(
        cls,
        elements: ElementCollection,
        peptide_fragment_file: Optional[str] = None,
    ):
        if peptide_fragment_file is None:
            peptide_fragment_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "peptide_fragments.yaml"
            )

        dict_ = load_yaml(peptide_fragment_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")

            return PeptideFragmentTypeInfo(
                name=d["name"],
                n_term=d.get("n_term", True),
                mass=mass,
                composition=composition,
            )

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def __setitem__(self, key, value):
        raise TypeError("PeptideFragmentTypeTypeCollection is immutable")

    def __delitem__(self, key):
        raise TypeError("PeptideFragmentTypeTypeCollection is immutable")

    def check_fragment_type_id(self, id: str, fragment_type: PeptideFragmentTypeInfo):
        if not id:
            raise ValueError(f"invalid fragment type id '{id}': empty")
        if any(map(str.isdigit, id)):
            raise ValueError(
                f"invalid fragment type id {id}: id cannot contain numbers"
            )
