from typing import Mapping, Sequence, Union

import numpy as np

from ..chem.gpep.sort import GlycanComparer
from ..chem.gpep.glycans import GlycanNode, MonosaccharideCollection
from ..chem.gpep.gpmass import GlycoPeptideMassCalculator, GlycoPeptideMzAnnotation
from ..chem.pep.mods import ModifiedSequence
from ..speclib.gpep.parser.annotation import GlycoPeptideMS2SpectrumAnnotator


class GlycoPeptideMS2SpectrumAnnotatorByComposition(GlycoPeptideMS2SpectrumAnnotator):
    def __init__(
        self,
        monosaccharides: MonosaccharideCollection,
        glycan_comparer: GlycanComparer,
        mass_calculator: GlycoPeptideMassCalculator,
        glycan_composition_map: Mapping[str, Sequence[str]],
        configs: Union[str, dict, None] = None,
    ):
        super().__init__(
            mass_calculator=mass_calculator,
            configs=configs,
        )
        self.monosaccharides = monosaccharides
        self.glycan_composition_map = glycan_composition_map
        self.glycan_comparer = glycan_comparer

    def _calculate_fragment_mz(
        self,
        modified_sequence: ModifiedSequence,
        glycan_struct: GlycanNode,
        glycan_position: int,
        **analyte_info,
    ):
        glycan_composition = self.monosaccharides.composition_str(
            glycan_struct.composition()
        )
        glycan_structs = self.glycan_composition_map.get(glycan_composition, None)
        glycan_struct = self.glycan_comparer.sort_glycan(glycan_struct)

        if glycan_structs is None or len(glycan_structs) == 0:
            raise ValueError(
                f"{glycan_composition} not found in glycan composition map"
            )
        if str(glycan_struct) not in glycan_structs:
            import warnings

            warnings.warn(
                f"{glycan_struct} : {glycan_composition} not found in glycan composition map"
            )

        fragment_args = self.get_config("fragments", typed=dict, required=False) or {}

        frag_mz = {}
        for gstruct in glycan_structs:
            frag = self.mass_calculator.fragment_mz(
                parsed_sequence=modified_sequence,
                glycan=GlycanNode.from_str(gstruct),
                glycan_position=glycan_position,
                **fragment_args,
            )

            if len(frag_mz) == 0:
                frag_mz.update(frag.annotations(), mz=frag.mz)
            else:
                is_new = ~np.isin(frag.mz, frag_mz["mz"])
                if sum(is_new) > 0:
                    frag_mz.update(
                        {
                            k: np.concatenate((frag_mz[k], v[is_new]))
                            for k, v in frag.annotations()
                        },
                        mz=np.concatenate((frag_mz["mz"], frag.mz[is_new])),
                    )
        frag_mz = GlycoPeptideMzAnnotation(**frag_mz)
        frag_mz = self._post_calculate_fragment_mz(frag_mz)
        return frag_mz
