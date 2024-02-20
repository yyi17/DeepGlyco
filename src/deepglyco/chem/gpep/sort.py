__all__ = ["GlycanComparer"]

import functools
from typing import cast

from .glycans import GlycanNode, MonosaccharideCollection


class GlycanComparer:
    def __init__(self, monosaccharides: MonosaccharideCollection):
        self.monosaccharide_order = list(monosaccharides.keys())
        self._compare_sorted_glycans_key_fn = functools.cmp_to_key(
            self._compare_sorted_glycans
        )

    def sort_glycan(self, glycan: GlycanNode) -> GlycanNode:
        if glycan.children is None or len(glycan.children) == 0:
            return glycan

        children = [self.sort_glycan(c) for c in glycan.children]
        children.sort(key=self._compare_sorted_glycans_key_fn)

        return GlycanNode(monosaccharide=glycan.monosaccharide, children=children)

    def _compare_sorted_glycans(self, glycan: GlycanNode, other: GlycanNode) -> int:
        x = self.monosaccharide_order.index(glycan.monosaccharide)
        y = self.monosaccharide_order.index(other.monosaccharide)
        cmp = x - y
        if cmp != 0:
            return cmp

        xlen = int(glycan.children is not None and len(glycan.children))
        ylen = int(other.children is not None and len(other.children))

        if not xlen and not ylen:
            return 0
        elif not xlen:
            return 1
        elif not ylen:
            return -1

        for i in range(min(xlen, ylen)):
            cmp = self._compare_sorted_glycans(
                cast(list, glycan.children)[i],
                cast(list, other.children)[i],
            )
            if cmp != 0:
                return cmp
        return ylen - xlen

    def compare(self, glycan: GlycanNode, other: GlycanNode):
        glycan = self.sort_glycan(glycan)
        other = self.sort_glycan(other)
        return self._compare_sorted_glycans(glycan, other)

    @functools.cached_property
    def compare_key_fn(self):
        return functools.cmp_to_key(self._compare_sorted_glycans)
