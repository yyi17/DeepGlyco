__all__ = [
    "MonosaccharideInfo",
    "MonosaccharideCollection",
    "MonosaccharideComposition",
    "GlycanNode",
    "glycan_node_graph",
]

import functools
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection

MonosaccharideComposition = Mapping[str, int]


@dataclass(frozen=True)
class MonosaccharideInfo:
    name: str
    mass: float
    composition: Optional[MonosaccharideComposition] = None


class MonosaccharideCollection(Dict[str, MonosaccharideInfo]):
    def __init__(self, monosaccharide_map: Mapping[str, MonosaccharideInfo]):
        super().__init__(**monosaccharide_map)
        for key, ms in monosaccharide_map.items():
            self.check_monosaccharide_id(key, ms)

    @classmethod
    def load(
        cls,
        elements: ElementCollection,
        monosaccharide_file: Optional[str] = None,
    ):
        if monosaccharide_file is None:
            monosaccharide_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "monosaccharides.yaml"
            )

        dict_ = load_yaml(monosaccharide_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")

            return MonosaccharideInfo(
                name=d["name"], mass=mass, composition=composition
            )

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def __setitem__(self, key, value):
        raise TypeError("MonosaccharideCollection is immutable")

    def __delitem__(self, key):
        raise TypeError("MonosaccharideCollection is immutable")

    def check_monosaccharide_id(self, id: str, monosaccharide: MonosaccharideInfo):
        if not id:
            raise ValueError(f"invalid monosaccharide id '{id}': empty")
        if not id.isalnum():
            raise ValueError(f"invalid monosaccharide id {id}: id must be alphanum")

    def parse_monosaccharide_composition(self, s: str) -> MonosaccharideComposition:
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
                elif not count.isdigit():
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
            raise ValueError(f"invalid composition format {s}: monosaccharide missing")
        elif errorcode == -4:
            raise ValueError(
                f"invalid composition format {s}: monosaccharide number {count} must be an integer"
            )
        elif errorcode == -5:
            raise ValueError(
                f"invalid composition format {s}: unknown monosaccharide {name}"
            )
        elif errorcode:
            raise ValueError(f"invalid composition format {s}")
        return parsed

    def mass_from_monosaccharide_composition(
        self, composition: Union[str, MonosaccharideComposition]
    ) -> float:
        if isinstance(composition, str):
            composition = self.parse_monosaccharide_composition(composition)
        mass = 0.0
        for key, value in composition.items():
            mass += self[key].mass * value
        return mass

    def composition_str(self, composition: MonosaccharideComposition) -> str:
        r = []
        for k in self:
            v = composition.get(k, None)
            if v is not None:
                r.append(f"{k}({v})")
        for k in composition:
            if k not in self:
                raise ValueError(
                    f"invalid monosaccharide composition: unknown monosaccharide {k}"
                )
        return "".join(r)


class GlycanNode:
    def __init__(
        self, monosaccharide: str, children: "Optional[List[GlycanNode]]" = None
    ):
        self.monosaccharide = monosaccharide
        self.children = children

    def __str__(self) -> str:
        s = "(" + self.monosaccharide
        if self.children is not None:
            s = s + "".join(str(c) for c in self.children)
        return s + ")"

    def __repr__(self) -> str:
        return f"{super().__repr__()} {self.__str__()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GlycanNode):
            return False
        if self.monosaccharide != other.monosaccharide:
            return False
        if self.children is None:
            if other.children is not None and len(other.children) > 0:
                return False
        if other.children is None:
            if self.children is not None and len(self.children) > 0:
                return False
        return self.children == other.children

    def iter_depth_first(self):
        yield self
        if self.children is not None:
            for c in self.children:
                yield from c.iter_depth_first()

    def composition(self) -> MonosaccharideComposition:
        def add(d1, d2):
            return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}

        d = {self.monosaccharide: 1}
        if isinstance(self.children, list):
            d = functools.reduce(add, (c.composition() for c in self.children), d)
        elif self.children is not None:
            d = add(d, self.children.composition())
        return d

    @classmethod
    def from_str(cls, s: str):
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]

        nodes: List[GlycanNode] = []
        start = None
        for i, c in enumerate(s):
            if c != "(" and c != ")":
                if start is None:
                    start = i
            else:
                if start is not None:
                    node = GlycanNode(s[start:i])
                    if len(nodes) > 0:
                        if isinstance(nodes[-1].children, list):
                            nodes[-1].children.append(node)
                        elif nodes[-1].children is None:
                            nodes[-1].children = [node]
                        else:
                            nodes[-1].children = [nodes[-1].children, node]
                    nodes.append(node)
                    start = None

            if c == ")":
                if len(nodes) <= 1:
                    raise ValueError(f"invalid glycan format {s}")
                nodes.pop()

        if len(nodes) == 0 and start is not None:
            return GlycanNode(s[start:])
        elif len(nodes) == 1:
            return nodes[0]
        else:
            raise ValueError(f"invalid glycan format {s}")


class GlycanNodeGraph(NamedTuple):
    nodes: Sequence[GlycanNode]
    edges: Sequence[Tuple[int, int]]


def glycan_node_graph(glycan: GlycanNode) -> GlycanNodeGraph:
    def _to_graph(glycan, parent_id, nodes, edges):
        node_id = len(nodes)
        nodes.append(glycan)
        if parent_id >= 0:
            edges.append((parent_id, node_id))
        if glycan.children is None or len(glycan.children) == 0:
            return
        for c in glycan.children:
            _to_graph(c, node_id, nodes, edges)

    nodes = []
    edges = []
    _to_graph(glycan, -1, nodes, edges)
    return GlycanNodeGraph(nodes, edges)
