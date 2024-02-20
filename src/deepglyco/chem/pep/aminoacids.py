__all__ = [
    "AminoAcidInfo",
    "AminoAcidCollection",
]

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection, ElementComposition


@dataclass(frozen=True)
class AminoAcidInfo:
    name: str
    mass: float
    composition: Optional[ElementComposition] = None


class AminoAcidCollection(Dict[str, AminoAcidInfo]):
    def __init__(self, amino_acid_map: Mapping[str, AminoAcidInfo]):
        super().__init__(**amino_acid_map)
        for key, aa in amino_acid_map.items():
            self.check_aa_id(key, aa)

    @classmethod
    def load(cls, elements: ElementCollection, amino_acid_file: Optional[str] = None):
        if amino_acid_file is None:
            amino_acid_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "amino_acids.yaml"
            )

        dict_ = load_yaml(amino_acid_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")

            return AminoAcidInfo(name=d["name"], mass=mass, composition=composition)

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def __setitem__(self, key, value):
        raise TypeError("AminoAcidCollection is immutable")

    def __delitem__(self, key):
        raise TypeError("AminoAcidCollection is immutable")

    def check_aa_id(self, id: str, aa_info: AminoAcidInfo):
        if not id:
            raise ValueError(f"invalid amino acid id '{id}': empty")
        if not len(id) == 1 or not id.isalpha() or not id.isupper():
            raise ValueError(
                f"invalid amino acid id {id}: id must be an uppercase letter"
            )
