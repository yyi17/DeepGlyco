__all__ = ["GlycoPeptideMS2SpectrumAnnotator"]

from typing import Union

import numpy as np
import pandas as pd

from ....chem.gpep.glycans import GlycanNode
from ....chem.gpep.gpmass import (
    GlycoPeptideMassCalculator,
    GlycoPeptideMzAnnotation,
)
from ....chem.pep.mods import ModifiedSequence, modified_sequence_to_str
from ....specio.spec import MassSpectrum
from ...common.annotation import MS2SpectrumAnnotatorBase
from ..spec import GlycoPeptideMS2Spectrum


class GlycoPeptideMS2SpectrumAnnotator(
    MS2SpectrumAnnotatorBase[GlycoPeptideMS2Spectrum]
):
    def __init__(
        self,
        mass_calculator: GlycoPeptideMassCalculator,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = {}

        self.mass_calculator = mass_calculator
        super().__init__(configs)

    def _calculate_fragment_mz(
        self,
        modified_sequence: ModifiedSequence,
        glycan_struct: GlycanNode,
        glycan_position: int,
        **analyte_info,
    ):
        fragment_args = self.get_config("fragments", required=False, typed=dict) or {}

        frag_mz = self.mass_calculator.fragment_mz(
            parsed_sequence=modified_sequence,
            glycan=glycan_struct,
            glycan_position=glycan_position,
            **fragment_args,
        )
        frag_mz = self._post_calculate_fragment_mz(frag_mz)
        return frag_mz

    def _post_calculate_fragment_mz(self, frag_mz: GlycoPeptideMzAnnotation):
        Y_mass_shift = self.get_config("Y_mass_shift", required=False, typed=list)
        if Y_mass_shift is not None:
            frag_mz_new = {}
            Y_indices = np.where(frag_mz.fragment_type == "Y")[0]
            frag_mz_new["mz"] = np.concatenate(
                [frag_mz.mz]
                + [
                    frag_mz.mz[Y_indices] + shift / frag_mz.charge[Y_indices]
                    for shift in Y_mass_shift
                ]
            )
            frag_mz_new["fragment_type"] = np.concatenate(
                [frag_mz.fragment_type]
                + [frag_mz.fragment_type[Y_indices]] * len(Y_mass_shift)
                # + [
                #     np.char.add(frag_mz.fragment_type[Y_indices], f"{shift:+g}")
                #     for shift in Y_mass_shift
                # ]
            )
            for k, v in frag_mz.annotations():
                if k != "fragment_type":
                    frag_mz_new[k] = np.concatenate(
                        [v] + [v[Y_indices]] * len(Y_mass_shift)
                    )
            frag_mz = frag_mz.__class__(**frag_mz_new)

        return frag_mz

    def _pre_create_annotated_spectrum(self, **analyte_info_fragments):
        Y_mass_shift = self.get_config("Y_mass_shift", required=False, typed=list)
        if Y_mass_shift is None:
            return analyte_info_fragments

        fragments = pd.DataFrame(
            {
                k: v
                for k, v in analyte_info_fragments.items()
                if isinstance(v, np.ndarray)
            }
        )

        fragment_keys = fragments.columns.difference(["mz", "intensity"]).tolist()
        indices = (
            fragments.groupby(fragment_keys, dropna=False, sort=False)["intensity"]
            .idxmax()
            .tolist()
        )

        return {
            k: v[indices] if isinstance(v, np.ndarray) else v
            for k, v in analyte_info_fragments.items()
        }

    def _calculate_precursor_mz(
        self,
        modified_sequence: ModifiedSequence,
        glycan_struct: GlycanNode,
        precursor_charge: int,
        **analyte_info,
    ):
        return self.mass_calculator.precursor_mz(
            parsed_sequence=modified_sequence,
            glycan=glycan_struct,
            charge=precursor_charge,
        )

    def _create_annotated_spectrum(
        self,
        modified_sequence: ModifiedSequence,
        glycan_struct: GlycanNode,
        glycan_position: int,
        precursor_charge: int,
        **analyte_info_fragments,
    ):
        analyte_info_fragments = self._pre_create_annotated_spectrum(
            **analyte_info_fragments
        )
        return GlycoPeptideMS2Spectrum(
            modified_sequence=modified_sequence_to_str(modified_sequence),
            glycan_struct=str(glycan_struct),
            glycan_position=glycan_position,
            precursor_charge=precursor_charge,
            **analyte_info_fragments,
        )

    def annotate(
        self,
        spectrum: MassSpectrum,
        modified_sequence: ModifiedSequence,
        glycan_struct: GlycanNode,
        glycan_position: int,
        precursor_charge: int,
        **analyte_info,
    ):
        return super().annotate(
            spectrum=spectrum,
            modified_sequence=modified_sequence,
            glycan_struct=glycan_struct,
            glycan_position=glycan_position,
            precursor_charge=precursor_charge,
            **analyte_info,
        )


