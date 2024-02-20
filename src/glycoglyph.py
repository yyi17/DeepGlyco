from deepglyco.chem.gpep.glycans import GlycanNode


class GlycoGlyphGlycanFormatter():
    def __init__(self, monosaccharide_names=None):
        if monosaccharide_names is None:
            monosaccharide_names = {
                'H': 'Man',
                'N': 'GlcNAc',
                'F': 'Fuc',
                'A': 'Neu5Ac',
                'G': 'Neu5Gc'
            }
        self.monosaccharide_names = monosaccharide_names

    def glycoglyph_name(self, glycan) -> str:
        if not isinstance(glycan, GlycanNode):
            glycan = GlycanNode.from_str(glycan)

        s = self.monosaccharide_names[glycan.monosaccharide]
        if isinstance(glycan.children, list):
            if len(glycan.children) == 1:
                s = self.glycoglyph_name(glycan.children[0]) + '??-?' + s
            else:
                s = ''.join(
                    self.glycoglyph_name(c) + '??-?' if i == 0 \
                    else '(' + self.glycoglyph_name(c) + '??-?' + ')'
                    for i, c in enumerate(reversed(glycan.children))
                ) + s
        return s


if __name__ == '__main__':
    gfmt = GlycoGlyphGlycanFormatter()

    for glycan in [
        '(N(F)(N(H(H(N(F)))(H(N(H(A)))))))',
        '(N(F)(N(H(H(N))(H(N(F)(H(A)))))))',
        '(N(F)(N(H(H)(H(N(F))(N(H(A)))))))',
        '(N(N(H(H(N(F)))(H(N(F)(H(A)))))))',
        '(N(N(H(H)(H(N(F))(N(F)(H(A)))))))',
        '(N(N(H(H(N(H)(F))(N(F)(H(A)))))))'
    ]:
        print(gfmt.glycoglyph_name(glycan))