__all__ = ["pGlycoReportParser"]

import os
import re
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ....chem.pep.aminoacids import AminoAcidCollection
from ....chem.pep.mods import (
    ModificationCollection,
    ModifiedSequence,
    ModPosition,
    modified_sequence_to_str,
)
from ..spec import GlycoPeptideMS2Spectrum
from .abs import GlycoPeptideReportParserBase


class pGlycoReportParser(GlycoPeptideReportParserBase):
    class CONST:
        COLUMN_SPEC_KEY = "GlySpec"
        COLUMN_SEQ = "Peptide"
        COLUMN_MOD = "Mod"
        COLUMN_GLYCAN = "PlausibleStruct"
        COLUMN_GLYCAN_POSITION = "GlySite"
        COLUMN_PRECURSOR_CHARGE = "Charge"
        COLUMNG_FRAGMENTS = "matched_ion"
        COLUMN_PRECURSOR_MZ = "PrecursorMZ"
        COLUMN_RUN_NAME = "RawName"
        COLUMN_SCAN_NUMBER = "Scan"
        COLUMN_SCORE = "TotalScore"
        COLUMNG_SPEC_KEY = "spec"

    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "pglyco.yaml"
            )
        super().__init__(config=configs)

        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_psm_report(
        self, pglyco_report: pd.DataFrame, glabel_report: pd.DataFrame
    ):
        pglyco_report = self.filter_report(pglyco_report)
        report = pd.merge(
            pglyco_report[
                pglyco_report.columns.intersection(
                    [v for k, v in vars(self).items() if k.startswith("COLUMN_")]
                )
            ],
            glabel_report[
                glabel_report.columns.intersection(
                    [v for k, v in vars(self).items() if k.startswith("COLUMNG_")]
                )
            ].rename(columns={self.CONST.COLUMNG_SPEC_KEY: self.CONST.COLUMN_SPEC_KEY}),
            on=[self.CONST.COLUMN_SPEC_KEY],
            copy=False,
        )

        for i, row in report.iterrows():
            yield self.parse_psm_report_row(row)

    def parse_psm_report_row(self, row):
        modified_sequence = modified_sequence_to_str(
            self.parse_sequence_modification(
                row[self.CONST.COLUMN_SEQ], row[self.CONST.COLUMN_MOD]
            )
        )

        matched_ion = [
            s.split("=") for s in row[self.CONST.COLUMNG_FRAGMENTS].split(";")
        ]
        annotation = [s[0] for s in matched_ion]
        if len(set(annotation)) != len(annotation):
            raise ValueError(
                f"duplicated fragments in {row.get(self.CONST.COLUMN_RUN_NAME, None)} "
                f"scan {row.get(self.CONST.COLUMN_SCAN_NUMBER, None)}: {annotation}"
            )
        annotation = list(map(self.parse_fragment_annotation, annotation))

        peaks = [s[1].split(",") for s in matched_ion]
        mz = np.array([t[0] for t in peaks], dtype=np.float32)
        intensity = np.array([t[1] for t in peaks], dtype=np.float32)

        fragment_type = np.array([t[0] for t in annotation], dtype=np.unicode_)
        fragment_number = np.array([t[1] for t in annotation], dtype=np.int16)
        fragment_charge = np.array([t[2] for t in annotation], dtype=np.int8)
        fragment_glycan = np.array([t[3] for t in annotation], dtype=np.unicode_)
        loss_type = np.array([""] * len(annotation), dtype=np.unicode_)

        return GlycoPeptideMS2Spectrum(
            modified_sequence=modified_sequence,
            glycan_struct=row[self.CONST.COLUMN_GLYCAN],
            glycan_position=row[self.CONST.COLUMN_GLYCAN_POSITION],
            precursor_charge=row[self.CONST.COLUMN_PRECURSOR_CHARGE],
            mz=mz,
            intensity=intensity,
            fragment_type=fragment_type,
            fragment_number=fragment_number,
            fragment_charge=fragment_charge,
            fragment_glycan=fragment_glycan,
            loss_type=loss_type,
            run_name=row.get(self.CONST.COLUMN_RUN_NAME, None),  # type: ignore
            scan_number=row.get(self.CONST.COLUMN_SCAN_NUMBER, None),  # type: ignore
            precursor_mz=row.get(self.CONST.COLUMN_PRECURSOR_MZ, None),  # type: ignore
            score=row.get(self.CONST.COLUMN_SCORE, None),  # type: ignore
        )

    def parse_fragment_annotation(self, s: str) -> Tuple[str, int, int, str]:
        match = re.match(
            r"^(?P<type>[A-Za-z]+)"
            r"(?P<cross>\$)?"
            r"(?P<number>[0-9]+)?"
            r"(?:-(?P<glycan>[^+]+))?"
            r"(?:\+(?P<charge>[0-9]+))?$",
            s,
        )

        if match is not None:
            fragment_type = match.group("type")
            cross_ring = match.group("cross")
            fragment_number = int(match.group("number") or -1)
            fragment_charge = int(match.group("charge") or 1)
            fragment_glycan = match.group("glycan") or ""
            if cross_ring and fragment_glycan:
                raise ValueError(
                    f"invalid fragment annotation {s}: both cross ring fragment symbol ($) and glycan are found"
                )
            if cross_ring:
                fragment_glycan = cross_ring
            if fragment_number == 0:
                fragment_number = -1
        else:
            fragment_type = s
            fragment_number = -1
            fragment_charge = 0
            fragment_glycan = ""

        return (fragment_type, fragment_number, fragment_charge, fragment_glycan)

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
            return list(zip(sequence, mod_list))

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

        return list(zip(sequence, mod_list))
