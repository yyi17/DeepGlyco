__all__ = ["NGlycan"]

import functools
from typing import Optional, Tuple

from .glycans import GlycanNode


NGlycanCore5 = Tuple[GlycanNode, GlycanNode, GlycanNode, GlycanNode, GlycanNode]
NGlycanBranches = Tuple[GlycanNode, ...]


SYMBOL_HEX = "H"
SYMBOL_HEXNAC = "N"
SYMBOL_FUC = "F"
SYMBOLS_BISECTING = ("N", "X")


class NGlycan:
    def __init__(self, glycan: GlycanNode):
        self.glycan = glycan

    @functools.cached_property
    def core5(self) -> Optional[NGlycanCore5]:
        n1 = self.glycan
        if (
            n1.monosaccharide != SYMBOL_HEXNAC
            or n1.children is None
            or len(n1.children) == 0
        ):
            return None
        n2 = next(
            filter(lambda x: x.monosaccharide == SYMBOL_HEXNAC, n1.children), None
        )
        if n2 is None or n2.children is None or len(n2.children) == 0:
            return None
        h1 = next(filter(lambda x: x.monosaccharide == SYMBOL_HEX, n2.children), None)
        if h1 is None or h1.children is None or len(h1.children) == 0:
            return None

        h2 = None
        h3 = None
        for x in h1.children:
            if x.monosaccharide == SYMBOL_HEX:
                if h2 is None:
                    h2 = x
                else:
                    h3 = x
                    break
        if h2 is None or h3 is None:
            return None
        return n1, n2, h1, h2, h3

    @functools.cached_property
    def is_canonical(self):
        core = self.core5
        if core is None:
            return False
        n1, n2, h1, h2, h3 = core

        assert n1.children is not None
        if len(n1.children) > 2:
            return False
        for x in n1.children:
            if x is not n2 and x.monosaccharide != SYMBOL_FUC:
                return False

        assert h1.children is not None
        if len(h1.children) > 3:
            return False
        for x in h1.children:
            if (
                x is not h2
                and x is not h3
                and x.monosaccharide not in SYMBOLS_BISECTING
            ):
                return False

        return self.subtype != "unknown"

    @functools.cached_property
    def is_high_mannose(self):
        core = self.core5
        if core is None:
            return False
        n1, n2, h1, h2, h3 = core
        assert n1.children is not None
        assert n2.children is not None
        assert h1.children is not None

        found_branch = False
        if h2.children is not None and len(h2.children) > 0:
            if not all(c.monosaccharide == SYMBOL_HEX for c in h2.iter_depth_first()):
                return False
            found_branch = True
        if h3.children is not None and len(h3.children) > 0:
            if not all(c.monosaccharide == SYMBOL_HEX for c in h3.iter_depth_first()):
                return False
            found_branch = True
        return found_branch

    @functools.cached_property
    def is_complex(self):
        core = self.core5
        if core is None:
            return False
        n1, n2, h1, h2, h3 = core
        assert n1.children is not None
        assert n2.children is not None
        assert h1.children is not None

        found_branch = False
        if h2.children is not None and len(h2.children) > 0:
            if not all(c.monosaccharide == SYMBOL_HEXNAC for c in h2.children):
                return False
            found_branch = True
        if h3.children is not None and len(h3.children) > 0:
            if not all(c.monosaccharide == SYMBOL_HEXNAC for c in h3.children):
                return False
            found_branch = True
        return found_branch

    @functools.cached_property
    def is_hybrid(self):
        core = self.core5
        if core is None:
            return False
        n1, n2, h1, h2, h3 = core
        assert n1.children is not None
        assert n2.children is not None
        assert h1.children is not None

        found_branch_n = False
        found_branch_h = False
        if h2.children is not None and len(h2.children) > 0:
            if all(c.monosaccharide == SYMBOL_HEXNAC for c in h2.children):
                found_branch_n = True
            elif all(c.monosaccharide == SYMBOL_HEX for c in h2.iter_depth_first()):
                found_branch_h = True
        if h3.children is not None and len(h3.children) > 0:
            if all(c.monosaccharide == SYMBOL_HEXNAC for c in h3.children):
                found_branch_n = True
            elif all(c.monosaccharide == SYMBOL_HEX for c in h3.iter_depth_first()):
                found_branch_h = True
        return found_branch_n and found_branch_h

    @property
    def subtype(self):
        if self.is_high_mannose:
            return "high_mannose"
        elif self.is_complex:
            return "complex"
        elif self.is_hybrid:
            return "hybrid"
        else:
            return "unknown"

    @functools.cached_property
    def core_fucoses(self) -> Optional[GlycanNode]:
        n1 = self.glycan
        if (
            n1.monosaccharide != SYMBOL_HEXNAC
            or n1.children is None
            or len(n1.children) == 0
        ):
            return None

        n1f = next(filter(lambda x: x.monosaccharide == SYMBOL_FUC, n1.children), None)
        return n1f

    @functools.cached_property
    def bisection(self) -> Optional[GlycanNode]:
        core = self.core5
        if core is None:
            return None
        h1 = core[2]
        assert h1.children is not None
        if len(h1.children) != 3:
            return None
        for x in h1.children:
            if x.monosaccharide in SYMBOLS_BISECTING:
                return x
        return None

    @functools.cached_property
    def branches(self) -> Tuple[NGlycanBranches, NGlycanBranches]:
        core = self.core5
        if core is None:
            return tuple()
        h2, h3 = core[3], core[4]

        return tuple(
            tuple(
                x for x in (h.children if h.children is not None else [])
                if not all(c.monosaccharide == SYMBOL_HEX for c in x.iter_depth_first())
            )
            for h in [h2, h3]
        )
