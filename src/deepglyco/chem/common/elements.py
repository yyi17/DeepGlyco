__all__ = [
    "ElementInfo",
    "ElementCollection",
    "ElementComposition",
]

import os
from typing import Dict, Mapping, NamedTuple, Optional, Union

from ...util.io.yaml import load_yaml


ElementComposition = Mapping[str, int]


class ElementInfo(NamedTuple):
    name: str
    mass: float


class ElementCollection(Dict[str, ElementInfo]):
    def __init__(self, element_map: Mapping[str, ElementInfo]):
        super().__init__(**element_map)
        for key, element in self.items():
            self.check_element_id(key, element)

    @classmethod
    def load(cls, element_file: Optional[str] = None):
        if element_file is None:
            element_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "elements.yaml"
            )

        dict_ = load_yaml(element_file)
        collection = cls({key: ElementInfo(**value) for key, value in dict_.items()})

        return collection

    def __setitem__(self, key, value):
        raise TypeError("ElementCollection is immutable")

    def __delitem__(self, key):
        raise TypeError("ElementCollection is immutable")

    def check_element_id(self, id: str, element_info: ElementInfo):
        if not id:
            raise ValueError(f"invalid element id '{id}': empty")
        if not id.isalnum():
            raise ValueError(f"invalid element id {id}: id must be alphanum")

    def parse_element_composition(self, s: str) -> ElementComposition:
        parsed = {}
        name = ""
        count = ""
        in_par = False
        errorcode = -1
        for ch in s:
            should_add = False
            if ch == "(":
                if in_par:
                    errorcode = -2
                    break
                in_par = True
            elif ch == ")":
                if not in_par:
                    errorcode = -2
                    break
                in_par = False
                if name == "":
                    errorcode = -3
                    break
                elif not (
                    len(count) > 1
                    and (count[0].isdigit() or count[0] in "+-")
                    and count[1:].isdigit()
                    or count.isdigit()
                ):
                    errorcode = -4
                    break
                else:
                    should_add = True
            elif in_par:
                count += ch
            else:
                if ch == " ":
                    if name != "":
                        should_add = True
                else:
                    name += ch
            if should_add:
                parsed[name] = int(count) if count != "" else 1
                errorcode = 0
                name = ""
                count = ""
        if in_par:
            errorcode = -2
        elif (not errorcode or errorcode == -1) and name != "":
            parsed[name] = 1
            errorcode = 0

        for name in parsed:
            if name not in self:
                errorcode = -5
                break

        if errorcode == -1:
            raise ValueError(f"invalid composition format {s}: empty")
        elif errorcode == -2:
            raise ValueError(f"invalid composition format {s}: parentheses not match")
        elif errorcode == -3:
            raise ValueError(f"invalid composition format {s}: element missing")
        elif errorcode == -4:
            raise ValueError(
                f"invalid composition format {s}: atom number {count} must be an integer"
            )
        elif errorcode == -5:
            raise ValueError(f"invalid composition format {s}: unknown element {name}")
        elif errorcode:
            raise ValueError(f"invalid composition format {s}")
        return parsed

    def mass_from_element_composition(
        self, composition: Union[str, ElementComposition]
    ) -> float:
        if isinstance(composition, str):
            composition = self.parse_element_composition(composition)
        mass = 0.0
        for key, value in composition.items():
            mass += self[key].mass * value
        return mass
