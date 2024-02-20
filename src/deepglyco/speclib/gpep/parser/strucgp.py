__all__ = ["StrucGPReportParser"]

import os
import re
from typing import Iterable, Union

import pandas as pd

from ..filter import GlycoPeptideSpectrumFilter

from .annotation import GlycoPeptideMS2SpectrumAnnotator

from ....chem.gpep.glycans import GlycanNode
from ....chem.gpep.gpmass import GlycoPeptideMassCalculator
from ....chem.pep.aminoacids import AminoAcidCollection
from ....chem.pep.mods import (
    ModificationCollection,
    ModifiedSequence,
    ModPosition,
    modified_sequence_to_str,
)
from ....specio.spec import MassSpectrum
from ...common.combine import NonRedundantSpectraConsensus
from ..spec import GlycoPeptideMS2Spectrum
from .abs import GlycoPeptideReportSpectrumParserBase


class StrucGPReportParser(GlycoPeptideReportSpectrumParserBase):
    class CONST:
        COLUMN_MOD_SEQ = "Peptide"
        COLUMN_GLYCAN = "structure_coding"
        COLUMN_PRECURSOR_CHARGE = "PrecursorCharge"
        COLUMN_PRECURSOR_MZ = "PrecursorMz"
        COLUMN_RUN_NAME = "FileName"
        COLUMN_SCAN_NUMBER = "MS2Scan"
        COLUMN_SCAN_NUMBER_LOW_ENERGY = "LowEnergy_MS2Scan"
        COLUMN_SCORE_PEPTIDE = "PeptideScore"
        COLUMN_SCORE_GLYCAN = "GlycanScore"

    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        annotator: GlycoPeptideMS2SpectrumAnnotator,
        spectrum_filter: GlycoPeptideSpectrumFilter,
        configs: Union[str, dict, None] = None,
    ):
        self.amino_acids = amino_acids
        self.modifications = modifications

        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "strucgp.yaml"
            )

        super().__init__(
            annotator=annotator,
            spectrum_filter=spectrum_filter,
            configs=configs,
        )

    def parse_psm_report(self, report: pd.DataFrame, spectra: Iterable[MassSpectrum]):
        report = self.filter_report(report)
        if all(
            report[self.CONST.COLUMN_SCAN_NUMBER]
            == report[self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY]
        ):
            return super().parse_psm_report(report, spectra)
        else:
            report = self.filter_report(report)
            result = self._parse_psm_report(report, spectra)

            filter_config = self.get_config(
                "spectrum_filter", required=False, typed=dict
            )
            if filter_config is not None:
                result = self.spectrum_filter.filter_spectra(result, **filter_config)
            return result

    def _parse_psm_report(
        self, report: pd.DataFrame, spectra: Iterable[MassSpectrum]
    ) -> Iterable[GlycoPeptideMS2Spectrum]:
        nr_spec = NonRedundantSpectraConsensus(
            dict(
                use_score_weight=False,
                replicate_similarity_threshold=0,
                peak_quorum=0,
                min_replicates_combined=0,
                max_replicates_combined=2,
            )
        )

        spectra_high: dict[tuple[str, int], GlycoPeptideMS2Spectrum] = {}
        spectra_low: dict[tuple[str, int], GlycoPeptideMS2Spectrum] = {}
        for spec in spectra:
            if spec.ms_level != 2:
                continue
            row = self.find_psm_report_row(report, spec)
            if row is not None:
                gpspec = self.parse_psm_report_row(row, spec)
                assert gpspec.run_name is not None and gpspec.scan_number is not None
                if (
                    row[self.CONST.COLUMN_SCAN_NUMBER]
                    == row[self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY]
                ):
                    yield gpspec
                elif row[self.CONST.COLUMN_SCAN_NUMBER] == spec.scan_number:
                    gpspec_low = spectra_low.get(
                        (
                            row[self.CONST.COLUMN_RUN_NAME],
                            row[self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY],
                        ),
                        None,
                    )
                    if gpspec_low is None:
                        spectra_high[(gpspec.run_name, gpspec.scan_number)] = gpspec
                        continue
                    spec_comb = nr_spec.merge_spectra([gpspec, gpspec_low])
                    if spec_comb is None:
                        continue
                    yield spec_comb
                elif row[self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY] == spec.scan_number:
                    gpspec_high = spectra_high.get(
                        (
                            row[self.CONST.COLUMN_RUN_NAME],
                            row[self.CONST.COLUMN_SCAN_NUMBER],
                        ),
                        None,
                    )
                    if gpspec_high is None:
                        spectra_low[(gpspec.run_name, gpspec.scan_number)] = gpspec
                        continue
                    spec_comb = nr_spec.merge_spectra([gpspec_high, gpspec])
                    if spec_comb is None:
                        continue
                    yield spec_comb

    def find_psm_report_row(self, report: pd.DataFrame, spectrum: MassSpectrum):
        if not isinstance(report.index, pd.MultiIndex):
            report.set_index(
                [
                    self.CONST.COLUMN_RUN_NAME,
                    self.CONST.COLUMN_SCAN_NUMBER,
                    self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY,
                ],
                drop=False,
                inplace=True,
            )

        try:
            row = report.xs(
                key=(spectrum.run_name, spectrum.scan_number),
                level=(self.CONST.COLUMN_RUN_NAME, self.CONST.COLUMN_SCAN_NUMBER),
            )
            if len(row) > 0:
                return row.iloc[0]
        except KeyError:
            pass
        try:
            row = report.xs(
                key=(spectrum.run_name, spectrum.scan_number),
                level=(
                    self.CONST.COLUMN_RUN_NAME,
                    self.CONST.COLUMN_SCAN_NUMBER_LOW_ENERGY,
                ),
            )
            if len(row) > 0:
                return row.iloc[0]
        except KeyError:
            pass
        return None

    def get_psm_report_row_info(self, row: pd.Series):
        modified_sequence = self.parse_modified_sequence(row[self.CONST.COLUMN_MOD_SEQ])
        glycan_struct = self.parse_glycan_struct(row[self.CONST.COLUMN_GLYCAN])

        striped_sequence = "".join([t[0] for t in modified_sequence])
        contig_match = re.search("N[A-Z]([ST]|$)", striped_sequence)
        if contig_match is None:
            raise ValueError(f"glycan position not found in {striped_sequence}")
        glycan_position = contig_match.span()[0] + 1

        return dict(
            modified_sequence=modified_sequence,
            glycan_struct=glycan_struct,
            glycan_position=glycan_position,
            precursor_charge=row[self.CONST.COLUMN_PRECURSOR_CHARGE],
            run_name=row.get(self.CONST.COLUMN_RUN_NAME, None),
            scan_number=row.get(self.CONST.COLUMN_SCAN_NUMBER, None),
            precursor_mz=row.get(self.CONST.COLUMN_PRECURSOR_MZ, None),
            score=row.get(self.CONST.COLUMN_SCORE_PEPTIDE, float("nan"))  # type: ignore
            + row.get(self.CONST.COLUMN_SCORE_GLYCAN, float("nan")),  # type: ignore
        )

    def parse_glycan_struct(self, struct: str) -> GlycanNode:
        monosaccharides = self.get_config("monosaccharides", typed=list)
        current = None
        stack = []
        for i, ch in enumerate(struct):
            if ch.isalpha():
                if ch.isupper():
                    layer = ord(ch) - ord("A")
                    if layer != len(stack):
                        raise ValueError(
                            f"invalid StrucGP glycan format {struct}: incorrect layer start {ch}"
                        )
                else:
                    layer = ord(ch) - ord("a")
                    if layer != len(stack) - 1:
                        raise ValueError(
                            f"invalid StrucGP glycan format {struct}: incorrect layer end {ch}"
                        )
                    current = stack.pop(-1)
            elif ch.isdigit():
                ms_index = int(ch) - 1
                node = GlycanNode(monosaccharides[ms_index])
                if len(stack) > 0:
                    if stack[-1].children is None:
                        stack[-1].children = [node]
                    else:
                        stack[-1].children.append(node)
                stack.append(node)
            else:
                raise ValueError(f"invalid StrucGP glycan format {struct}")
        if current is None:
            raise ValueError(
                f"invalid StrucGP glycan format {struct}: no monosaccharides"
            )
        return current

    def parse_modified_sequence(self, sequence: str) -> ModifiedSequence:
        modifications = self.get_config("modifications", required=False, typed=dict)
        parsed = []
        match_list = re.findall(r"([A-Z])(?:\[([^\]]+)\])?|(.+)", sequence)
        t = (None,)
        aa, mod = "", ""
        errorcode = -1
        for pos, t in enumerate(match_list):
            if t[-1]:
                errorcode = -2
                break

            aa = t[0]
            mod = t[1]
            if aa not in self.amino_acids:
                errorcode = -5
                break
            if mod:
                mod_ = (
                    modifications.get(mod, None) if modifications is not None else None
                )
                modpos = (
                    ModPosition.n_term
                    if pos == 0
                    else ModPosition.c_term
                    if pos == len(match_list) - 1
                    else ModPosition.not_term
                )
                if mod_ is not None:
                    mod_ = self.modifications.search(mod_ + aa, modpos)
                    if mod_ == "":
                        errorcode = -6
                        break
                else:
                    mod_ = self.modifications.search_by_name(
                        mod,
                        aa,
                        modpos,
                    )
                    if len(mod_) == 0:
                        errorcode = -6
                        break
                    mod_ = mod_[0]
            else:
                mod_ = ""
            parsed.append((aa, mod_))
            errorcode = 0

        if errorcode == -1:
            raise ValueError(
                f"invalid StrucGP peptide format {sequence}: no amino acids"
            )
        elif errorcode == -2:
            raise ValueError(
                f"invalid StrucGP peptide format {sequence}: unknown pattern {t[-1]}"
            )
        elif errorcode == -5:
            raise ValueError(
                f"invalid StrucGP peptide {sequence}: unknown amino acid {aa}"
            )
        elif errorcode == -6:
            raise ValueError(
                f"invalid StrucGP peptide {sequence}: unknown modification {mod} at {aa}"
            )
        elif errorcode:
            raise ValueError(f"invalid StrucGP peptide format {sequence}")
        return parsed

    def parse_evidence_report(self, evidence_report: pd.DataFrame):
        evidence_report = self.filter_report(evidence_report)

        retention_time = evidence_report["RetentionTime"]

        def _parse_row(row: pd.Series):
            r = self.get_psm_report_row_info(row)
            r["modified_sequence"] = modified_sequence_to_str(r["modified_sequence"])  # type: ignore
            r["glycan_struct"] = str(r["glycan_struct"])
            return pd.Series(r)

        evidence_report = evidence_report.apply(_parse_row, axis=1)

        evidence_report["retention_time"] = retention_time
        return evidence_report
