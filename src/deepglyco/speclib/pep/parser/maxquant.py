__all__ = ["MaxQuantReportParser"]

import os
import re
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd

from ....chem.pep.aminoacids import AminoAcidCollection
from ....chem.pep.mods import (
    ModificationCollection,
    ModifiedSequence,
    ModPosition,
    modified_sequence_to_str,
)
from ..spec import PeptideMS2Spectrum
from .abs import PeptideReportParserBase


class MaxQuantReportParser(PeptideReportParserBase):
    class CONST:
        COLUMN_MODSEQ = "Modified sequence"
        COLUMN_PRECURSOR_CHARGE = "Charge"
        COLUMN_MZ = "Masses"
        COLUMN_INTENSITY = "Intensities"
        COLUMN_ANNOTATION = "Matches"
        COLUMN_PRECURSOR_MZ = "m/z"
        COLUMN_RUN_NAME = "Raw file"
        COLUMN_SCAN_NUMBER = "Scan number"
        COLUMN_SCORE = "Score"

    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "maxquant.yaml"
            )
        super().__init__(configs)

        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_psm_report_row(self, row: pd.Series) -> PeptideMS2Spectrum:
        run_name = row.get(self.CONST.COLUMN_RUN_NAME, None)
        scan_number = row.get(self.CONST.COLUMN_SCAN_NUMBER, None)

        modified_sequence = modified_sequence_to_str(
            self.parse_modified_sequence(row[self.CONST.COLUMN_MODSEQ])
        )

        mz = np.array(row[self.CONST.COLUMN_MZ].split(";"), dtype=np.float32)
        intensity = np.array(
            row[self.CONST.COLUMN_INTENSITY].split(";"), dtype=np.float32
        )
        annotation = row[self.CONST.COLUMN_ANNOTATION].split(";")
        if len(set(annotation)) != len(annotation):
            raise ValueError(
                f"duplicated fragments in {run_name} scan {scan_number}: {annotation}"
            )
        annotation = list(map(self.parse_fragment_annotation, annotation))
        fragment_type = np.array([t[0] for t in annotation], dtype=np.unicode_)
        fragment_number = np.array([t[1] for t in annotation], dtype=np.int16)
        fragment_charge = np.array([t[2] for t in annotation], dtype=np.int8)
        loss_type = np.array([t[3] for t in annotation], dtype=np.unicode_)

        # protein_name = row["Proteins"]
        # if isinstance(protein_name, str):
        #     protein_name = protein_name.split(";")
        # elif protein_name is None or np.isnan(protein_name):
        #     protein_name = None
        # else:
        #     protein_name = str(protein_name)

        return PeptideMS2Spectrum(
            modified_sequence=modified_sequence,
            precursor_charge=row[self.CONST.COLUMN_PRECURSOR_CHARGE],
            mz=mz,
            intensity=intensity,
            fragment_type=fragment_type,
            fragment_number=fragment_number,
            fragment_charge=fragment_charge,
            loss_type=loss_type,
            run_name=row.get(self.CONST.COLUMN_RUN_NAME, None),  # type: ignore
            scan_number=row.get(self.CONST.COLUMN_SCAN_NUMBER, None),  # type: ignore
            precursor_mz=row.get(self.CONST.COLUMN_PRECURSOR_MZ, None),  # type: ignore
            score=row.get(self.CONST.COLUMN_SCORE, None),  # type: ignore
        )

    def parse_msms_report_row(self, row: pd.Series) -> PeptideMS2Spectrum:
        return self.parse_psm_report_row(row)

    def parse_msms_report(
        self, msms_report: pd.DataFrame
    ) -> Iterable[PeptideMS2Spectrum]:
        return self.parse_psm_report(msms_report)

    def parse_fragment_annotation(self, s: str) -> Tuple[str, int, int, str]:
        match = re.match(
            r"^(?P<type>[^0-9]+)"
            r"(?P<number>[0-9]+)?"
            r"(?:-(?P<loss>[^()]+))?"
            r"(?:\((?P<charge>[0-9]+)\+\))?$",
            s,
        )  # TODO: b20*; y17-H2O	duplicated 1985, 1887

        if match is not None:
            fragment_type = match.group("type")
            fragment_number = int(match.group("number") or -1)
            fragment_charge = int(match.group("charge") or 1)
            loss_type = match.group("loss") or ""
        else:
            fragment_type = s
            fragment_number = -1
            fragment_charge = 0
            loss_type = ""

        return (fragment_type, fragment_number, fragment_charge, loss_type)

    def parse_modified_sequence(self, sequence: str) -> ModifiedSequence:
        variable_modifications = self.get_config(
            "modifications", "variable_modifications", required=False, typed=dict
        )
        fixed_modifications = self.get_config(
            "modifications", "fixed_modifications", required=False, typed=dict
        )

        parsed = []
        terminus = 0
        terminal_mod = ""
        t = (None,)
        aa = ""
        mod = ""
        errorcode = -1
        for t in re.findall(r"([_A-Z])(?:\(([a-z0-9]+)\))?|(.+)", sequence):
            if t[-1]:
                errorcode = -2
                break
            if t[0] == "_":
                terminus += 1
                if terminus == 1 or terminus == 2:
                    terminal_mod = t[1]
                else:
                    errorcode = -3
                    break
            else:
                if terminus == 0:
                    terminus = 1
                elif terminus > 1:
                    errorcode = -3
                    break
                if errorcode:
                    errorcode = 0
                aa = t[0]
                mod = t[1]
            if aa:
                if terminal_mod:
                    if mod:
                        errorcode = -4
                        break
                    mod = terminal_mod
                    terminal_mod = ""
                if aa not in self.amino_acids:
                    errorcode = -5
                    break
                if mod:
                    mod_ = (
                        variable_modifications.get(mod, mod)
                        if variable_modifications is not None
                        else mod
                    )
                else:
                    mod_ = (
                        fixed_modifications.get(aa, "")
                        if fixed_modifications is not None
                        else ""
                    )
                if mod_ != "":
                    mod_ = self.modifications.search(
                        mod_ + aa,
                        (
                            ModPosition.n_term
                            if len(parsed) == 0
                            else ModPosition.c_term
                            if terminus == 2
                            else ModPosition.not_term
                        ),
                    )
                    if mod_ == "":
                        errorcode = -6
                        break
                if terminus == 2:
                    parsed.pop()
                parsed.append((aa, mod_))

        if errorcode == -1:
            raise ValueError(
                f"invalid MaxQuant peptide format {sequence}: no amino acids"
            )
        elif errorcode == -2:
            raise ValueError(
                f"invalid MaxQuant peptide format {sequence}: unknown pattern {t[-1]}"
            )
        elif errorcode == -3:
            raise ValueError(
                f"invalid MaxQuant peptide format {sequence}: incorrect terminus symbols"
            )
        elif errorcode == -4:
            raise ValueError(
                f"invalid MaxQuant peptide format {sequence}: terminal and internal modifications ({terminal_mod}, {mod}) at the same AA position are not supported"
            )
        elif errorcode == -5:
            raise ValueError(
                f"invalid MaxQuant peptide {sequence}: unknown amino acid {aa}"
            )
        elif errorcode == -6:
            raise ValueError(
                f"invalid MaxQuant peptide {sequence}: unknown modification {mod} at {aa}"
            )
        elif errorcode:
            raise ValueError(f"invalid MaxQuant peptide format {sequence}")
        return parsed

    def parse_evidence_report(self, evidence_report: pd.DataFrame):
        evidence_report = self.filter_report(evidence_report)
        evidence_report = evidence_report[
            [
                "Proteins",
                "Modified sequence",
                "Charge",
                "m/z",
                "Raw file",
                "Retention time",
                "Score",
                "Match score",
            ]
        ]
        score = evidence_report["Score"].fillna(0.0)
        score += evidence_report["Match score"].fillna(0.0)
        evidence_report = evidence_report.drop(columns=["Score", "Match score"]).rename(
            columns={
                "Proteins": "protein_name",
                "Modified sequence": "modified_sequence",
                "Charge": "precursor_charge",
                "m/z": "precursor_mz",
                "Raw file": "run_name",
                "Retention time": "retention_time",
            }
        )
        evidence_report["score"] = score
        evidence_report["modified_sequence"] = evidence_report["modified_sequence"].map(
            lambda s: modified_sequence_to_str(self.parse_modified_sequence(s))
        )
        return evidence_report
