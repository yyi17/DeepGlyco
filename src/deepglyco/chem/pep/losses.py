__all__ = [
    "NeutralLossTypeInfo",
    "NeutralLossTypeCollection",
]

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection, ElementComposition


@dataclass(frozen=True)
class NeutralLossTypeInfo:
    name: str
    mass: float
    mod_specific: bool = False
    composition: Optional[ElementComposition] = None


class NeutralLossTypeCollection(Dict[str, NeutralLossTypeInfo]):
    def __init__(self, neutral_loss_map: Mapping[str, NeutralLossTypeInfo]):
        super().__init__(**neutral_loss_map)
        for key, loss in neutral_loss_map.items():
            self.check_loss_id(key, loss)

    @classmethod
    def load(cls, elements: ElementCollection, neutral_loss_file: Optional[str] = None):
        if neutral_loss_file is None:
            neutral_loss_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "neutral_losses.yaml"
            )

        dict_ = load_yaml(neutral_loss_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")

            return NeutralLossTypeInfo(
                name=d["name"],
                mass=mass,
                mod_specific=d.get("mod_specific", False),
                composition=composition,
            )

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def __setitem__(self, key, value):
        raise TypeError("NeutralLossTypeCollection is immutable")

    def __delitem__(self, key) -> None:
        raise TypeError("NeutralLossTypeCollection is immutable")

    def check_loss_id(self, id: str, loss_info: NeutralLossTypeInfo):
        if not id:
            raise ValueError(f"invalid neutral loss type id '{id}': empty")
        if not id.isalnum():
            raise ValueError(f"invalid neutral loss type id {id}: id must be alphanum")
