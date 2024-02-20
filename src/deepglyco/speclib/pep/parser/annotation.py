__all__ = ["PeptideMS2SpectrumAnnotator"]

from typing import Union

from ....chem.pep.mods import ModifiedSequence, modified_sequence_to_str
from ....chem.pep.pepmass import PeptideMassCalculator
from ....specio.spec import MassSpectrum
from ...common.annotation import MS2SpectrumAnnotatorBase
from ..spec import PeptideMS2Spectrum


class PeptideMS2SpectrumAnnotator(MS2SpectrumAnnotatorBase[PeptideMS2Spectrum]):
    def __init__(
        self,
        mass_calculator: PeptideMassCalculator,
        configs: Union[str, dict],
    ):
        self.mass_calculator = mass_calculator
        super().__init__(configs)

    def _calculate_fragment_mz(
        self,
        modified_sequence: ModifiedSequence,
        **analyte_info,
    ):
        fragment_args = self.get_config("fragments", required=False, typed=dict) or {}

        return self.mass_calculator.fragment_mz(
            parsed_sequence=modified_sequence,
            **fragment_args,
        )

    def _calculate_precursor_mz(
        self,
        modified_sequence: ModifiedSequence,
        precursor_charge: int,
        **analyte_info,
    ):
        return self.mass_calculator.precursor_mz(
            parsed_sequence=modified_sequence,
            charge=precursor_charge,
        )

    def _create_annotated_spectrum(
        self,
        modified_sequence: ModifiedSequence,
        precursor_charge: int,
        **analyte_info_fragments,
    ):
        return PeptideMS2Spectrum(
            modified_sequence=modified_sequence_to_str(modified_sequence),
            precursor_charge=precursor_charge,
            **analyte_info_fragments,
        )

    def annotate(
        self,
        spectrum: MassSpectrum,
        modified_sequence: ModifiedSequence,
        precursor_charge: int,
        **analyte_info,
    ) -> PeptideMS2Spectrum:
        return super().annotate(
            spectrum,
            modified_sequence=modified_sequence,
            precursor_charge=precursor_charge,
            **analyte_info,
        )
