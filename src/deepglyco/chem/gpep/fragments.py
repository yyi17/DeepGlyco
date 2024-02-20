__all__ = [
    "GlycanFragmentTypeInfo",
    "GlycanFragmentTypeCollection",
    "reducing_end_fragments",
    "reducing_end_fragment_graph",
    "branch_fragments",
    "branch_fragment_graph",
]

import functools
import itertools
import os
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ...util.collections.list import remove_duplicates
from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection, ElementComposition
from .glycans import (
    GlycanNode,
    GlycanNodeGraph,
    MonosaccharideComposition,
    glycan_node_graph,
)
from .nglycans import NGlycan


@dataclass(frozen=True)
class GlycanFragmentTypeInfo:
    name: str
    reducing_end: bool = True
    mass: float = 0.0
    composition: Optional[ElementComposition] = None


class GlycanFragmentTypeCollection(Dict[str, GlycanFragmentTypeInfo]):
    def __init__(self, glycan_fragment_map: Mapping[str, GlycanFragmentTypeInfo]):
        super().__init__(**glycan_fragment_map)
        for key, ft in glycan_fragment_map.items():
            self.check_fragment_type_id(key, ft)

    @classmethod
    def load(
        cls,
        elements: ElementCollection,
        glycan_fragment_file: Optional[str] = None,
    ):
        if glycan_fragment_file is None:
            glycan_fragment_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "glycan_fragments.yaml"
            )

        dict_ = load_yaml(glycan_fragment_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")

            return GlycanFragmentTypeInfo(
                name=d["name"],
                reducing_end=d.get("reducing_end", True),
                mass=mass,
                composition=composition,
            )

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def __setitem__(self, key, value):
        raise TypeError("GlycanFragmentTypeTypeCollection is immutable")

    def __delitem__(self, key):
        raise TypeError("GlycanFragmentTypeTypeCollection is immutable")

    def check_fragment_type_id(self, id: str, fragment_type: GlycanFragmentTypeInfo):
        if not id:
            raise ValueError(f"invalid fragment type id '{id}': empty")
        if any(map(str.isdigit, id)):
            raise ValueError(
                f"invalid fragment type id {id}: id cannot contain numbers"
            )


def reducing_end_fragments(glycan: GlycanNode) -> List[MonosaccharideComposition]:
    def add(d1, d2):
        return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}

    def cross_add(l1, l2):
        return [add(x1, x2) for x1 in l1 for x2 in l2]

    def concat(a, b):
        return a + b

    def _reducing_end_fragments(glycan: GlycanNode):
        d = {glycan.monosaccharide: 1}
        if glycan.children is None:
            r = [d]
        else:
            r: List[MonosaccharideComposition] = [
                add(d, x)
                for x in functools.reduce(
                    cross_add,
                    (concat([{}], _reducing_end_fragments(c)) for c in glycan.children),
                )
            ]
            r = remove_duplicates(r)
        return r

    return _reducing_end_fragments(glycan)[:-1]


def glycan_cleavage_nodes(glycan: GlycanNode) -> Iterable[List[GlycanNode]]:
    yield [glycan]
    if glycan.children is None or len(glycan.children) == 0:
        yield []
        return
    for t in itertools.product(*(glycan_cleavage_nodes(c) for c in glycan.children)):
        yield list(itertools.chain.from_iterable(t))


class GlycanFragmentGraph(NamedTuple):
    monosaccharide_nodes: Sequence[GlycanNode]
    monosaccharide_edges: Sequence[Tuple[int, int]]

    cleavage_nodes: Sequence[int]
    lost_monosaccharide_cleavage_edges: Sequence[Tuple[int, int]]
    retained_monosaccharide_cleavage_edges: Sequence[Tuple[int, int]]

    fragment_nodes: Sequence[Tuple[int, ...]]
    cleavage_fragment_edges: Sequence[Tuple[int, int]]

    composition_nodes: Sequence[MonosaccharideComposition]
    fragment_composition_edges: Sequence[Tuple[int, int]]

    @property
    def monosaccharide_graph(self):
        return GlycanNodeGraph(self.monosaccharide_nodes, self.monosaccharide_edges)


def reducing_end_fragment_graph(glycan: Union[GlycanNode, GlycanNodeGraph, GlycanFragmentGraph]):
    if isinstance(glycan, GlycanFragmentGraph):
        monosaccharide_graph = glycan.monosaccharide_graph
        cleavage_nodes = list(glycan.cleavage_nodes)
        cleavage_lost_edges = list(glycan.lost_monosaccharide_cleavage_edges)
        cleavage_retained_edges = list(glycan.retained_monosaccharide_cleavage_edges)
        glycan = monosaccharide_graph.nodes[0]
    else:
        if isinstance(glycan, GlycanNodeGraph):
            monosaccharide_graph = glycan
            glycan = monosaccharide_graph.nodes[0]
        elif isinstance(glycan, GlycanNode):
            monosaccharide_graph = glycan_node_graph(glycan)
        else:
            raise TypeError(f"invalid type: {type(glycan)}")
        cleavage_nodes = []
        cleavage_lost_edges = []
        cleavage_retained_edges = []

    monosaccharide_id_list = list(map(id, monosaccharide_graph.nodes))
    glycan_composition = glycan.composition()

    fragment_nodes = []
    fragment_edges = []

    composition_nodes = []
    composition_edges = []

    def try_add_cleavage(cleavage: GlycanNode):
        cleavage_node_id = monosaccharide_id_list.index(id(cleavage))
        if cleavage_node_id in cleavage_nodes:
            cleavage_index = cleavage_nodes.index(cleavage_node_id)
            return cleavage_index

        cleavage_index = len(cleavage_nodes)
        cleavage_nodes.append(cleavage_node_id)

        lost_nodes = [
            monosaccharide_id_list.index(id(g)) for g in cleavage.iter_depth_first()
        ]
        retained_nodes = [
            i for i in range(len(monosaccharide_id_list)) if i not in lost_nodes
        ]
        for i in lost_nodes:
            cleavage_lost_edges.append((i, cleavage_index))
        for i in retained_nodes:
            cleavage_retained_edges.append((i, cleavage_index))

        return cleavage_index

    def try_add_fragment(cleavage_indexes: Tuple[int]):
        if len(cleavage_indexes) == 0:
            return None

        fragment_index = len(fragment_nodes)
        fragment_nodes.append(tuple(cleavage_nodes[i] for i in cleavage_indexes))
        for cleavage_index in cleavage_indexes:
            fragment_edges.append((cleavage_index, fragment_index))

        return fragment_index

    def try_add_composition(fragment_index: int, cleavages: Iterable[GlycanNode]):
        composition = dict(glycan_composition)
        for cleavage in cleavages:
            for k, v in cleavage.composition().items():
                composition[k] -= v
                if composition[k] == 0:
                    composition.pop(k)

        if composition not in composition_nodes:
            composition_index = len(composition_nodes)
            composition_nodes.append(composition)
        else:
            composition_index = composition_nodes.index(composition)
        composition_edges.append((fragment_index, composition_index))

        return composition_index

    for cleavages in glycan_cleavage_nodes(glycan):
        cleavage_indexes = tuple(try_add_cleavage(cleavage) for cleavage in cleavages)
        fragment_index = try_add_fragment(cleavage_indexes)
        if fragment_index is None:
            continue
        try_add_composition(fragment_index, cleavages)

    return GlycanFragmentGraph(
        monosaccharide_nodes=monosaccharide_graph.nodes,
        monosaccharide_edges=monosaccharide_graph.edges,
        cleavage_nodes=cleavage_nodes,
        lost_monosaccharide_cleavage_edges=cleavage_lost_edges,
        retained_monosaccharide_cleavage_edges=cleavage_retained_edges,
        fragment_nodes=fragment_nodes,
        cleavage_fragment_edges=fragment_edges,
        composition_nodes=composition_nodes,
        fragment_composition_edges=composition_edges,
    )


def branch_fragments(
    glycan: Union[NGlycan, GlycanNode]
) -> List[MonosaccharideComposition]:
    if isinstance(glycan, GlycanNode):
        glycan = NGlycan(glycan)

    branches = glycan.branches
    core5 = glycan.core5
    if core5 is None:
        raise ValueError(f"{glycan.glycan} is not a N-glycan with a canonical core")

    r = []
    for branch in [*branches[0], *branches[1]]:
        branch_composition = branch.composition()
        for cleavages in glycan_cleavage_nodes(branch):
            composition = dict(branch_composition)
            for cl in cleavages:
                comp = cl.composition()
                if comp not in r:
                    r.append(comp)

                for k, v in comp.items():
                    composition[k] -= v
                    if composition[k] == 0:
                        composition.pop(k)
            if len(composition) > 0 and composition not in r:
                r.append(composition)

            if len(composition) > 0:
                composition = dict(composition)
                composition["H"] = composition.get("H", 0) + 1
                if composition not in r:
                    r.append(composition)

        branch_composition = dict(branch_composition)
        branch_composition["H"] = branch_composition.get("H", 0) + 1
        if branch_composition not in r:
            r.append(branch_composition)

    return r


def branch_fragment_graph(glycan: Union[NGlycan, GlycanNode, GlycanNodeGraph, GlycanFragmentGraph]):
    if isinstance(glycan, GlycanFragmentGraph):
        monosaccharide_graph = glycan.monosaccharide_graph
        cleavage_nodes = list(glycan.cleavage_nodes)
        cleavage_lost_edges = list(glycan.lost_monosaccharide_cleavage_edges)
        cleavage_retained_edges = list(glycan.retained_monosaccharide_cleavage_edges)
        glycan = NGlycan(monosaccharide_graph.nodes[0])
    else:
        if isinstance(glycan, GlycanNodeGraph):
            monosaccharide_graph = glycan
            glycan = NGlycan(monosaccharide_graph.nodes[0])
        elif isinstance(glycan, GlycanNode):
            monosaccharide_graph = glycan_node_graph(glycan)
            glycan = NGlycan(glycan)
        elif isinstance(glycan, NGlycan):
            monosaccharide_graph = glycan_node_graph(glycan.glycan)
        else:
            raise TypeError(f"invalid type: {type(glycan)}")
        cleavage_nodes = []
        cleavage_lost_edges = []
        cleavage_retained_edges = []

    monosaccharide_id_list = list(map(id, monosaccharide_graph.nodes))

    fragment_nodes = []
    fragment_edges = []

    composition_nodes = []
    composition_edges = []

    def try_add_cleavage(cleavage: GlycanNode):
        cleavage_node_id = monosaccharide_id_list.index(id(cleavage))
        if cleavage_node_id in cleavage_nodes:
            cleavage_index = cleavage_nodes.index(cleavage_node_id)
            return cleavage_index

        cleavage_index = len(cleavage_nodes)
        cleavage_nodes.append(cleavage_node_id)

        lost_nodes = [
            monosaccharide_id_list.index(id(g)) for g in cleavage.iter_depth_first()
        ]
        retained_nodes = [
            i for i in range(len(monosaccharide_id_list)) if i not in lost_nodes
        ]
        for i in lost_nodes:
            cleavage_lost_edges.append((i, cleavage_index))
        for i in retained_nodes:
            cleavage_retained_edges.append((i, cleavage_index))

        return cleavage_index

    def try_add_fragment(cleavage_indexes: Sequence[int]):
        if len(cleavage_indexes) == 0:
            return None

        fragment_node = tuple(cleavage_nodes[i] for i in cleavage_indexes)
        if fragment_node in fragment_nodes:
            return None

        fragment_index = len(fragment_nodes)
        fragment_nodes.append(fragment_node)
        for cleavage_index in cleavage_indexes:
            fragment_edges.append((cleavage_index, fragment_index))

        return fragment_index

    def try_add_composition(fragment_index: int, cleavages: Sequence[GlycanNode]):
        assert len(cleavages) > 0

        composition = dict(cleavages[0].composition())
        if len(cleavages) > 1:
            for cleavage in cleavages[1:]:
                for k, v in cleavage.composition().items():
                    composition[k] -= v
                    if composition[k] == 0:
                        composition.pop(k)

        if composition not in composition_nodes:
            composition_index = len(composition_nodes)
            composition_nodes.append(composition)
        else:
            composition_index = composition_nodes.index(composition)
        composition_edges.append((fragment_index, composition_index))

        return composition_index

    branches = glycan.branches
    core5 = glycan.core5
    if core5 is None:
        raise ValueError(f"{glycan.glycan} is not a N-glycan with a canonical core")

    for branch, corehex in zip(
        [*branches[0], *branches[1]],
        [core5[3]] * len(branches[0]) + [core5[4]] * len(branches[1]),
    ):
        branch_cleavage_index = try_add_cleavage(branch)
        other_branches = (
            [cl for cl in corehex.children if cl is not branch]
            if corehex.children is not None
            else []
        )
        corehex_cleavage_indexes = [try_add_cleavage(corehex)] + [
            try_add_cleavage(cl) for cl in other_branches
        ]

        for cleavages in glycan_cleavage_nodes(branch):
            cleavage_indexes = []

            for cleavage in cleavages:
                cleavage_index = try_add_cleavage(cleavage)
                cleavage_indexes.append(cleavage_index)
                fragment_index = try_add_fragment([cleavage_index])
                if fragment_index is not None:
                    try_add_composition(fragment_index, [cleavage])

            if len(cleavages) > 1 or len(cleavages) == 1 and cleavages[0] is not branch:
                fragment_index = try_add_fragment(
                    [branch_cleavage_index, *cleavage_indexes]
                )
                if fragment_index is not None:
                    try_add_composition(fragment_index, [branch, *cleavages])

                fragment_index = try_add_fragment(
                    [*corehex_cleavage_indexes, *cleavage_indexes]
                )
                if fragment_index is not None:
                    try_add_composition(fragment_index, [corehex, *other_branches, *cleavages])

        fragment_index = try_add_fragment(corehex_cleavage_indexes)
        if fragment_index is not None:
            try_add_composition(fragment_index, [corehex, *other_branches])

    return GlycanFragmentGraph(
        monosaccharide_nodes=monosaccharide_graph.nodes,
        monosaccharide_edges=monosaccharide_graph.edges,
        cleavage_nodes=cleavage_nodes,
        lost_monosaccharide_cleavage_edges=cleavage_lost_edges,
        retained_monosaccharide_cleavage_edges=cleavage_retained_edges,
        fragment_nodes=fragment_nodes,
        cleavage_fragment_edges=fragment_edges,
        composition_nodes=composition_nodes,
        fragment_composition_edges=composition_edges,
    )
