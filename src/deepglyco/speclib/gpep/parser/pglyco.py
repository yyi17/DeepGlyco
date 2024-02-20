__all__ = ["pGlycoReportParser"]

import os
import re
from typing import Any, Iterable, Union

import pandas as pd

from ....chem.pep.aminoacids import AminoAcidCollection
from ....chem.pep.mods import (
    ModificationCollection,
    ModifiedSequence,
    ModPosition,
    modified_sequence_to_str,
)
from ....chem.gpep.glycans import GlycanNode
from ....specio.spec import MassSpectrum
from ..filter import GlycoPeptideSpectrumFilter
from ..spec import GlycoPeptideMS2Spectrum
from .abs import GlycoPeptideReportSpectrumParserBase
from .annotation import GlycoPeptideMS2SpectrumAnnotator


class pGlycoReportParser(GlycoPeptideReportSpectrumParserBase):
    class CONST:
        COLUMN_SPEC_KEY = "GlySpec"
        COLUMN_SEQ = "Peptide"
        COLUMN_MOD = "Mod"
        COLUMN_GLYCAN = "PlausibleStruct"
        COLUMN_GLYCAN_POSITION = "GlySite"
        COLUMN_PRECURSOR_CHARGE = "Charge"
        COLUMN_PRECURSOR_MZ = "PrecursorMZ"
        COLUMN_RUN_NAME = "RawName"
        COLUMN_SCAN_NUMBER = "Scan"
        COLUMN_SCORE = "TotalScore"

    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        annotator: GlycoPeptideMS2SpectrumAnnotator,
        spectrum_filter: GlycoPeptideSpectrumFilter,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "pglyco.yaml"
            )
        super().__init__(
            annotator=annotator,
            spectrum_filter=spectrum_filter,
            configs=configs,
        )

        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_psm_report(
        self, pglyco_report: pd.DataFrame, spectra: Iterable[MassSpectrum]
    ):
        pglyco_report = self.filter_report(pglyco_report)
        return super().parse_psm_report(pglyco_report, spectra)

    def parse_psm_report_row(
        self, row: pd.Series, spectrum: MassSpectrum
    ) -> GlycoPeptideMS2Spectrum:
        kwargs: dict[str, Any] = self.get_psm_report_row_info(row)
        kwargs.update(
            {
                "run_name": spectrum.spectrum_name,
                "scan_number": spectrum.scan_number,
            }
        )
        return self.annotator.annotate(spectrum, **kwargs)

    def get_psm_report_row_info(self, row: pd.Series):
        modified_sequence = self.parse_sequence_modification(
            row[self.CONST.COLUMN_SEQ], row[self.CONST.COLUMN_MOD]
        )

        glycan_struct = GlycanNode.from_str(row[self.CONST.COLUMN_GLYCAN])
        glycan_position = row[self.CONST.COLUMN_GLYCAN_POSITION]

        return dict(
            modified_sequence=modified_sequence,
            glycan_struct=glycan_struct,
            glycan_position=glycan_position,
            precursor_charge=row[self.CONST.COLUMN_PRECURSOR_CHARGE],
            # run_name=row.get(self.CONST.COLUMN_RUN_NAME, None),
            run_name=row[self.CONST.COLUMN_SPEC_KEY],
            scan_number=row.get(self.CONST.COLUMN_SCAN_NUMBER, None),
            precursor_mz=row.get(self.CONST.COLUMN_PRECURSOR_MZ, None),
            score=row.get(self.CONST.COLUMN_SCORE, float("nan")),  # type: ignore
        )

    def find_psm_report_row(self, report: pd.DataFrame, spectrum: MassSpectrum):
        if not isinstance(report.index, pd.MultiIndex):
            report.set_index(
                self.CONST.COLUMN_SPEC_KEY,
                drop=False,
                inplace=True,
            )

        try:
            row = report.loc[[spectrum.spectrum_name]]
            if len(row) > 0:
                return row.iloc[0]
        except KeyError:
            pass
        return None


    def parse_sequence_modification(
        self, sequence: str, modification: str
    ) -> ModifiedSequence:
        modifications = self.get_config("modifications", required=False, typed=dict)

        aa_list = []
        for aa in sequence:
            if aa == "J":
                aa = "N"
            if aa not in self.amino_acids:
                raise ValueError(
                    f"invalid pGlyco peptide {sequence}: unknown amino acid {aa}"
                )
            aa_list.append(aa)

        mod_list = [""] * len(sequence)

        if not isinstance(modification, str):
            return list(zip(aa_list, mod_list))

        for modstr in modification.split(";"):
            if len(modstr) == 0:
                continue
            match = re.match(
                "^(?P<pos>[0-9]+),(?P<name>[^\\[]+)\\[[A-Za-z0-9\\-]+\\]$", modstr
            )
            if match is None:
                raise ValueError(f"invalid pGlyco modification format {modstr}")
            pos = int(match.group("pos"))
            name = match.group("name")

            mod = modifications.get(name, None) if modifications is not None else None
            modpos = (
                ModPosition.n_term
                if pos == 1
                else ModPosition.c_term
                if pos == len(aa_list)
                else ModPosition.not_term
            )
            if mod is not None:
                mod = self.modifications.search(mod + aa_list[pos - 1], modpos)
            else:
                mod = self.modifications.search_by_name(
                    name,
                    aa_list[pos - 1],
                    modpos,
                )
                if len(mod) == 0:
                    mod = ""
                else:
                    mod = mod[0]
            if mod == "":
                raise ValueError(
                    f"invalid pGlyco modification: unknown modification {name} at {aa_list[pos - 1]}"
                )
            mod_list[pos - 1] = mod

        return list(zip(aa_list, mod_list))
