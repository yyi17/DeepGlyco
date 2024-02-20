__all__ = ["GlycoPeptideSpectralLibraryBase"]

import abc
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ...util.collections.list import index_ragged_list
from ..common.rtlib import RetentionTimeLibraryBase
from ..common.speclib import Columns, Indices, SpectralLibraryBase
from ..pep.abs import PeptideLibraryBase, PreparedImportedData
from .spec import GlycoPeptideMS2Spectrum, GlycoPeptideMS2SpectrumProto

PD_INDEX_NAME = "index"


class GlycoPeptideSpectralLibraryBase(
    PeptideLibraryBase,
    SpectralLibraryBase[GlycoPeptideMS2Spectrum],
    RetentionTimeLibraryBase,
):
    class CONST(PeptideLibraryBase.CONST):
        TABLE_GLYCAN_INFO = "glycan_info"
        COLUMN_GLYCAN_STRUCT = "glycan_struct"
        ID_GLYCAN_INFO = "glycan_id"

        TABLE_GLYCOPEPTIDE_INFO = "glycopeptide_info"
        COLUMN_GLYCAN_POSITION = "glycan_position"
        ID_GLYCOPEPTIDE_INFO = "glycopeptide_id"

        ID_PRECURSOR_INFO = "precursor_id"

        TABLE_SPECTRUM_INFO = "spectrum_info"
        COLUMNS_SPECTRUM_INFO = (
            "run_name",
            "scan_number",
            "score",
        )

        TABLE_FRAGMENTS = "spectrum_fragments"
        COLUMNS_FRAGMENTS = (
            "mz",
            "intensity",
            "fragment_charge",
            "fragment_type",
            "fragment_number",
            "loss_type",
            "fragment_glycan",
        )
        INDICES_FRAGMENTS = "spectrum_fragment_indices"

        TABLE_RETENTION_TIME = "retention_time"
        TABLE_ION_MOBILITY = "ion_mobility"

    @abc.abstractmethod
    def get_glycan_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_glycopeptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @property
    def num_glycans(self) -> int:
        data = self.get_glycan_info()
        if data is not None:
            return len(data)
        else:
            return 0

    @property
    def num_glycopeptides(self) -> int:
        data = self.get_glycopeptide_info()
        if data is not None:
            return len(data)
        else:
            return 0

    def get_glycopeptide_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        if columns is not None:
            if self.CONST.ID_PEPTIDE_INFO not in columns:
                columns = [self.CONST.ID_PEPTIDE_INFO] + columns
            if self.CONST.ID_GLYCAN_INFO not in columns:
                columns = [self.CONST.ID_GLYCAN_INFO] + columns

        data = self.get_glycopeptide_info(indices, columns)
        if data is None:
            return data

        if indices is None:
            peptide_info = self.get_peptide_info()
            assert peptide_info is not None
            peptide_info = peptide_info.loc[data[self.CONST.ID_PEPTIDE_INFO]]
            glycan_info = self.get_glycan_info()
            assert glycan_info is not None
            glycan_info = glycan_info.loc[data[self.CONST.ID_GLYCAN_INFO]]
        else:
            peptide_info = self.get_peptide_info(
                indices=data[self.CONST.ID_PEPTIDE_INFO]
            )
            assert peptide_info is not None
            glycan_info = self.get_glycan_info(indices=data[self.CONST.ID_GLYCAN_INFO])
            assert glycan_info is not None

        peptide_info = peptide_info.set_index(data.index)
        glycan_info = glycan_info.set_index(data.index)
        data = pd.concat(
            (peptide_info, glycan_info, data),
            axis=1,
            copy=False,
        )
        return data

    def get_precursor_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        if columns is not None and not self.CONST.ID_GLYCOPEPTIDE_INFO in columns:
            columns = [self.CONST.ID_GLYCOPEPTIDE_INFO] + columns

        data = self.get_precursor_info(indices, columns)
        if data is None:
            return data

        if indices is None:
            glycopeptides = self.get_glycopeptide_data()
            assert glycopeptides is not None
            glycopeptides = glycopeptides.loc[data[self.CONST.ID_GLYCOPEPTIDE_INFO]]
        else:
            glycopeptides = self.get_glycopeptide_data(
                indices=data[self.CONST.ID_GLYCOPEPTIDE_INFO]
            )
            assert glycopeptides is not None

        glycopeptides = glycopeptides.set_index(data.index)
        data = pd.concat(
            (glycopeptides, data),
            axis=1,
            copy=False,
        )
        return data

    def get_spectrum_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        if columns is not None and not self.CONST.ID_PRECURSOR_INFO in columns:
            columns = [self.CONST.ID_PRECURSOR_INFO] + columns
        data = self.get_spectrum_info(indices, columns)
        if data is None:
            return data

        if indices is None:
            precursors = self.get_precursor_data()
            assert precursors is not None
            precursors = precursors.loc[data[self.CONST.ID_PRECURSOR_INFO]]
        else:
            precursors = self.get_precursor_data(
                indices=data[self.CONST.ID_PRECURSOR_INFO]
            )
            assert precursors is not None

        precursors = precursors.set_index(data.index)
        data = pd.concat(
            (precursors, data),
            axis=1,
            copy=False,
        )
        return data

    def get_spectra(self, spectrum_id_list: List[int]) -> List[GlycoPeptideMS2Spectrum]:
        spectrum_info = self.get_spectrum_info(indices=spectrum_id_list)
        if spectrum_info is None:
            raise ValueError("empty spectrum info")
        spectrum_info = spectrum_info.astype(object)
        fragments = self.get_fragments_by_spectrum_id(spectrum_id_list)
        precursor_info = self.get_precursor_info(
            indices=spectrum_info[self.CONST.ID_PRECURSOR_INFO].astype(np.int64)
        )
        if precursor_info is None:
            raise ValueError("empty precursor info")
        precursor_info = precursor_info.astype(object)
        glycopeptide_info = self.get_glycopeptide_info(
            indices=precursor_info[self.CONST.ID_GLYCOPEPTIDE_INFO].astype(np.int64)
        )
        if glycopeptide_info is None:
            raise ValueError("empty glycopeptide info")
        peptide_info = self.get_peptide_info(
            indices=glycopeptide_info[self.CONST.ID_PEPTIDE_INFO].astype(np.int64)
        )
        if peptide_info is None:
            raise ValueError("empty peptide info")
        peptide_info = peptide_info.astype(object)
        glycan_info = self.get_glycan_info(
            indices=glycopeptide_info[self.CONST.ID_GLYCAN_INFO].astype(np.int64)
        )
        if glycan_info is None:
            raise ValueError("empty glycan info")
        glycan_info = glycan_info.astype(object)

        glycopeptide_info = glycopeptide_info.drop(
            columns=[self.CONST.ID_PEPTIDE_INFO, self.CONST.ID_GLYCAN_INFO]
        )
        precursor_info = precursor_info.drop(columns=[self.CONST.ID_GLYCOPEPTIDE_INFO])
        spectrum_info = spectrum_info.drop(columns=[self.CONST.ID_PRECURSOR_INFO])
        spectra = []
        for i in range(0, len(spectrum_info)):
            arg_dict = {}
            arg_dict.update(peptide_info.iloc[i])
            arg_dict.update(glycan_info.iloc[i])
            arg_dict.update(glycopeptide_info.iloc[i])
            arg_dict.update(precursor_info.iloc[i])
            arg_dict.update(spectrum_info.iloc[i])
            arg_dict.update(
                {col: series.values for col, series in fragments[i].items()}
            )
            spectra.append(GlycoPeptideMS2Spectrum(**arg_dict))
        return spectra

    def iter_spectra(self) -> Iterable[GlycoPeptideMS2Spectrum]:
        spectrum_info = self.get_spectrum_info()
        if spectrum_info is None:
            raise ValueError("empty spectrum info")
        spectrum_info = spectrum_info.astype(object)
        fragments = self.get_spectrum_fragments()
        if fragments is None:
            return
        fragment_indices = self.get_spectrum_fragment_indices()
        if fragment_indices is None:
            return
        precursor_info = self.get_precursor_info()
        if precursor_info is None:
            raise ValueError("empty precursor info")
        precursor_info = precursor_info.astype(object)
        glycopeptide_info = self.get_glycopeptide_info()
        if glycopeptide_info is None:
            raise ValueError("empty glycopeptide info")
        peptide_info = self.get_peptide_info()
        if peptide_info is None:
            raise ValueError("empty peptide info")
        peptide_info = peptide_info.astype(object)
        glycan_info = self.get_glycan_info()
        if glycan_info is None:
            raise ValueError("empty glycan info")
        glycan_info = glycan_info.astype(object)

        for spec_id in range(0, len(spectrum_info)):
            prec_id = spectrum_info.iloc[spec_id][self.CONST.ID_PRECURSOR_INFO]
            gp_id = precursor_info.iloc[prec_id][self.CONST.ID_GLYCOPEPTIDE_INFO]
            pept_id = glycopeptide_info.iloc[gp_id][self.CONST.ID_PEPTIDE_INFO]
            gly_id = glycopeptide_info.iloc[gp_id][self.CONST.ID_GLYCAN_INFO]
            frag_id_se = (fragment_indices[spec_id], fragment_indices[spec_id + 1])
            arg_dict = dict(peptide_info.iloc[pept_id])
            arg_dict.update(glycan_info.iloc[gly_id])
            arg_dict.update(glycopeptide_info.iloc[gp_id])
            arg_dict.update(precursor_info.iloc[prec_id])
            arg_dict.update(spectrum_info.iloc[spec_id])
            arg_dict.pop(self.CONST.ID_PEPTIDE_INFO)
            arg_dict.pop(self.CONST.ID_GLYCAN_INFO)
            arg_dict.pop(self.CONST.ID_GLYCOPEPTIDE_INFO)
            arg_dict.pop(self.CONST.ID_PRECURSOR_INFO)
            arg_dict.update(
                {
                    col: series.values
                    for col, series in fragments[frag_id_se[0] : frag_id_se[1]].items()
                }
            )
            yield GlycoPeptideMS2Spectrum(**arg_dict)

    def import_glycans(self, glycans: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_glycans(glycans)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_glycans(self, glycans: pd.DataFrame):
        glycans = glycans.drop_duplicates(subset=[self.CONST.COLUMN_GLYCAN_STRUCT])
        existing_glycans = self.get_glycan_info(
            columns=[self.CONST.COLUMN_GLYCAN_STRUCT]
        )

        if existing_glycans is None:
            new_glycans = glycans.reset_index(drop=True)
            glycan_indices = (
                new_glycans[[self.CONST.COLUMN_GLYCAN_STRUCT]]
                .reset_index(drop=False)
                .set_index(self.CONST.COLUMN_GLYCAN_STRUCT)[PD_INDEX_NAME]
            )
            return PreparedImportedData(
                {self.CONST.TABLE_GLYCAN_INFO: new_glycans},
                {self.CONST.TABLE_GLYCAN_INFO: glycan_indices},
            )

        num_existing_glycans = len(existing_glycans)
        new_glycans = glycans.loc[
            ~glycans[self.CONST.COLUMN_GLYCAN_STRUCT].isin(
                existing_glycans[self.CONST.COLUMN_GLYCAN_STRUCT]
            )
        ]
        new_glycans = new_glycans.set_index(
            pd.RangeIndex(num_existing_glycans, num_existing_glycans + len(new_glycans))
        )
        glycan_indices = (
            pd.concat(
                (existing_glycans, new_glycans[[self.CONST.COLUMN_GLYCAN_STRUCT]]),
                axis=0,
                ignore_index=False,
                copy=False,
            )
            .reset_index(drop=False)
            .set_index(self.CONST.COLUMN_GLYCAN_STRUCT)[PD_INDEX_NAME]
        )
        return PreparedImportedData(
            {self.CONST.TABLE_GLYCAN_INFO: new_glycans},
            {self.CONST.TABLE_GLYCAN_INFO: glycan_indices},
        )

    def import_glycopeptides(self, glycopeptides: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_glycopeptides(glycopeptides)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_glycopeptides(self, glycopeptides: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_peptides(
            glycopeptides[
                glycopeptides.columns.intersection(
                    [self.CONST.COLUMN_PROTEIN_NAME, self.CONST.COLUMN_PEPTIDE_SEQUENCE]
                )
            ]
        )
        prepared_data_g, prepared_indices_g = self._prepare_imported_glycans(
            glycopeptides[[self.CONST.COLUMN_GLYCAN_STRUCT]]
        )
        prepared_data.update(prepared_data_g)
        prepared_indices.update(prepared_indices_g)
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]
        glycan_indices = prepared_indices[self.CONST.TABLE_GLYCAN_INFO]

        if self.CONST.COLUMN_PROTEIN_NAME in glycopeptides.columns:
            glycopeptides = glycopeptides.drop(columns=[self.CONST.COLUMN_PROTEIN_NAME])

        COLUMNS_GLYCOPEPTIDE_INFO = [
            self.CONST.ID_PEPTIDE_INFO,
            self.CONST.ID_GLYCAN_INFO,
            self.CONST.COLUMN_GLYCAN_POSITION,
        ]
        glycopeptides = glycopeptides.drop_duplicates(
            subset=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
                self.CONST.COLUMN_GLYCAN_POSITION,
            ]
        )
        glycopeptides.insert(
            0,
            self.CONST.ID_PEPTIDE_INFO,
            peptide_indices.loc[
                glycopeptides[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
            ].set_axis(glycopeptides.index),
        )
        glycopeptides.insert(
            1,
            self.CONST.ID_GLYCAN_INFO,
            glycan_indices.loc[glycopeptides[self.CONST.COLUMN_GLYCAN_STRUCT]].set_axis(
                glycopeptides.index
            ),
        )

        existing_glycopeptides = self.get_glycopeptide_info(
            columns=COLUMNS_GLYCOPEPTIDE_INFO
        )
        if existing_glycopeptides is None:
            new_glycopeptides = glycopeptides.reset_index(drop=True)
            glycopeptide_indices = (
                new_glycopeptides[COLUMNS_GLYCOPEPTIDE_INFO]
                .reset_index(drop=False)
                .set_index(COLUMNS_GLYCOPEPTIDE_INFO)[PD_INDEX_NAME]
            )
        else:
            num_existing_glycopeptides = len(existing_glycopeptides)
            new_glycopeptides = glycopeptides.loc[
                ~pd.concat(
                    (
                        glycopeptides[COLUMNS_GLYCOPEPTIDE_INFO],
                        existing_glycopeptides,
                    ),
                    axis=0,
                    copy=False,
                )
                .duplicated(keep=False)
                .head(len(glycopeptides))
            ]
            new_glycopeptides = new_glycopeptides.set_index(
                pd.RangeIndex(
                    num_existing_glycopeptides,
                    num_existing_glycopeptides + len(new_glycopeptides),
                )
            )
            glycopeptide_indices = (
                pd.concat(
                    (
                        existing_glycopeptides[COLUMNS_GLYCOPEPTIDE_INFO],
                        new_glycopeptides[COLUMNS_GLYCOPEPTIDE_INFO],
                    ),
                    axis=0,
                    ignore_index=False,
                    copy=False,
                )
                .reset_index(drop=False)
                .set_index(COLUMNS_GLYCOPEPTIDE_INFO)[PD_INDEX_NAME]
            )

        new_glycopeptides = new_glycopeptides.drop(
            columns=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
            ]
        )
        prepared_data[self.CONST.TABLE_GLYCOPEPTIDE_INFO] = new_glycopeptides
        prepared_indices[self.CONST.TABLE_GLYCOPEPTIDE_INFO] = glycopeptide_indices
        return PreparedImportedData(prepared_data, prepared_indices)

    def import_precursors(self, precursors: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_precursors(precursors)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_precursors(self, precursors: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_glycopeptides(
            precursors[
                precursors.columns.intersection(
                    [
                        self.CONST.COLUMN_PROTEIN_NAME,
                        self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                        self.CONST.COLUMN_GLYCAN_STRUCT,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                    ]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]
        glycan_indices = prepared_indices[self.CONST.TABLE_GLYCAN_INFO]
        glycopeptide_indices = prepared_indices[self.CONST.TABLE_GLYCOPEPTIDE_INFO]

        if self.CONST.COLUMN_PROTEIN_NAME in precursors.columns:
            precursors = precursors.drop(columns=[self.CONST.COLUMN_PROTEIN_NAME])

        COLUMNS_PRECURSOR_INFO = [
            self.CONST.ID_GLYCOPEPTIDE_INFO,
            self.CONST.COLUMN_PRECURSOR_CHARGE,
        ]
        precursors = precursors.drop_duplicates(
            subset=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
                self.CONST.COLUMN_GLYCAN_POSITION,
                self.CONST.COLUMN_PRECURSOR_CHARGE,
            ]
        )

        precursors.insert(
            0,
            self.CONST.ID_PEPTIDE_INFO,
            peptide_indices.loc[
                precursors[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
            ].set_axis(precursors.index),
        )
        precursors.insert(
            1,
            self.CONST.ID_GLYCAN_INFO,
            glycan_indices.loc[precursors[self.CONST.COLUMN_GLYCAN_STRUCT]].set_axis(
                precursors.index
            ),
        )
        precursors.insert(
            0,
            self.CONST.ID_GLYCOPEPTIDE_INFO,
            glycopeptide_indices.loc[
                precursors[
                    [
                        self.CONST.ID_PEPTIDE_INFO,
                        self.CONST.ID_GLYCAN_INFO,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                    ]
                ].itertuples(index=False, name=None)
            ].set_axis(precursors.index),
        )

        existing_precursors = self.get_precursor_info(columns=COLUMNS_PRECURSOR_INFO)
        if existing_precursors is None:
            new_precursors = precursors.reset_index(drop=True)
            precursor_indices = (
                new_precursors[COLUMNS_PRECURSOR_INFO]
                .reset_index()
                .set_index(COLUMNS_PRECURSOR_INFO)[PD_INDEX_NAME]
            )
        else:
            num_existing_precursors = len(existing_precursors)
            new_precursors = precursors.loc[
                ~pd.concat(
                    (
                        precursors[COLUMNS_PRECURSOR_INFO],
                        existing_precursors[COLUMNS_PRECURSOR_INFO],
                    ),
                    axis=0,
                    copy=False,
                )
                .duplicated(keep=False)
                .head(len(precursors))
            ]
            new_precursors = new_precursors.set_index(
                pd.RangeIndex(
                    num_existing_precursors,
                    num_existing_precursors + len(new_precursors),
                )
            )

            precursor_indices = (
                pd.concat(
                    (
                        existing_precursors[COLUMNS_PRECURSOR_INFO],
                        new_precursors[COLUMNS_PRECURSOR_INFO],
                    ),
                    axis=0,
                    ignore_index=False,
                    copy=False,
                )
                .reset_index(drop=False)
                .set_index(COLUMNS_PRECURSOR_INFO)[PD_INDEX_NAME]
            )

        new_precursors = new_precursors.drop(
            columns=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
                self.CONST.ID_PEPTIDE_INFO,
                self.CONST.ID_GLYCAN_INFO,
                self.CONST.COLUMN_GLYCAN_POSITION,
            ]
        )
        prepared_indices["precursor_info"] = precursor_indices
        return PreparedImportedData(prepared_data, prepared_indices)

    def import_spectra(self, spectra: Iterable[GlycoPeptideMS2SpectrumProto]):
        prepared_data, _ = self._prepare_imported_spectra(spectra)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_spectra(
        self, spectra: Iterable[GlycoPeptideMS2SpectrumProto]
    ):
        new_peptides = []
        existing_peptides = self.get_peptide_info(
            columns=[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
        )
        if existing_peptides is not None:
            num_existing_peptides = len(existing_peptides)
            peptide_indices: dict[str, int] = dict(
                existing_peptides.reset_index(drop=False).set_index(
                    self.CONST.COLUMN_PEPTIDE_SEQUENCE
                )[PD_INDEX_NAME]
            )
        else:
            num_existing_peptides = 0
            peptide_indices = {}

        new_glycans = []
        existing_glycans = self.get_glycan_info(
            columns=[self.CONST.COLUMN_GLYCAN_STRUCT]
        )
        if existing_glycans is not None:
            num_existing_glycans = len(existing_glycans)
            glycan_indices: dict[str, int] = dict(
                existing_glycans.reset_index(drop=False).set_index(
                    self.CONST.COLUMN_GLYCAN_STRUCT
                )[PD_INDEX_NAME]
            )
        else:
            num_existing_glycans = 0
            glycan_indices = {}

        COLUMNS_GLYCOPEPTIDE_INFO = [
            self.CONST.ID_PEPTIDE_INFO,
            self.CONST.ID_GLYCAN_INFO,
            self.CONST.COLUMN_GLYCAN_POSITION,
        ]
        new_glycopeptides = []
        existing_glycopeptides = self.get_glycopeptide_info(
            columns=COLUMNS_GLYCOPEPTIDE_INFO
        )
        if existing_glycopeptides is not None:
            num_existing_glycopeptides = len(existing_glycopeptides)
            glycopeptide_indices: dict[tuple[int, int, int], int] = dict(
                existing_glycopeptides.reset_index(drop=False).set_index(
                    COLUMNS_GLYCOPEPTIDE_INFO
                )[PD_INDEX_NAME]
            )
        else:
            num_existing_glycopeptides = 0
            glycopeptide_indices = {}

        COLUMNS_PRECURSOR_INFO = [
            self.CONST.ID_GLYCOPEPTIDE_INFO,
            self.CONST.COLUMN_PRECURSOR_CHARGE,
        ]
        new_precursors = []
        existing_precursors = self.get_precursor_info(columns=COLUMNS_PRECURSOR_INFO)
        if existing_precursors is not None:
            num_existing_precursors = len(existing_precursors)
            precursor_indices: dict[tuple[int, int], int] = dict(
                existing_precursors.reset_index(drop=False).set_index(
                    COLUMNS_PRECURSOR_INFO
                )[PD_INDEX_NAME]
            )
        else:
            num_existing_precursors = 0
            precursor_indices = {}

        new_spectra = []
        num_existing_spectra = self.num_spectra

        new_spectrum_fragments = []
        num_existing_spectrum_fragments = self.num_spectrum_fragments

        for spec in spectra:
            pep_id = peptide_indices.get(spec.modified_sequence, None)
            if pep_id is None:
                new_peptides.append((spec.modified_sequence))
                pep_id = num_existing_peptides
                peptide_indices[spec.modified_sequence] = pep_id
                num_existing_peptides += 1

            gly_id = glycan_indices.get(spec.glycan_struct, None)
            if gly_id is None:
                new_glycans.append((spec.glycan_struct))
                gly_id = num_existing_glycans
                glycan_indices[spec.glycan_struct] = gly_id
                num_existing_glycans += 1

            gp_id = glycopeptide_indices.get(
                (pep_id, gly_id, spec.glycan_position), None
            )
            if gp_id is None:
                new_glycopeptides.append((pep_id, gly_id, spec.glycan_position))
                gp_id = num_existing_glycopeptides
                glycopeptide_indices[(pep_id, gly_id, spec.glycan_position)] = gp_id
                num_existing_glycopeptides += 1

            prec_id = precursor_indices.get((gp_id, spec.precursor_charge), None)
            if prec_id is None:
                new_precursors.append((gp_id, spec.precursor_charge, spec.precursor_mz))
                prec_id = num_existing_precursors
                precursor_indices[(gp_id, spec.precursor_charge)] = prec_id
                num_existing_precursors += 1

            new_spectra.append((prec_id, spec.run_name, spec.scan_number, spec.score))
            new_spectrum_fragments.append(
                {k: spec.__getattribute__(k) for k in self.CONST.COLUMNS_FRAGMENTS}
            )

        prepared_data = {}
        if len(new_peptides) > 0:
            new_peptides = pd.DataFrame(
                new_peptides,
                columns=[self.CONST.COLUMN_PEPTIDE_SEQUENCE],
                index=pd.RangeIndex(
                    num_existing_peptides - len(new_peptides), num_existing_peptides
                ),
            )
            prepared_data[self.CONST.TABLE_PEPTIDE_INFO] = new_peptides

        if len(new_glycans) > 0:
            new_glycans = pd.DataFrame(
                new_glycans,
                columns=[self.CONST.COLUMN_GLYCAN_STRUCT],
                index=pd.RangeIndex(
                    num_existing_glycans - len(new_glycans), num_existing_glycans
                ),
            )
            prepared_data[self.CONST.TABLE_GLYCAN_INFO] = new_glycans
        if len(new_glycopeptides) > 0:
            new_glycopeptides = pd.DataFrame(
                new_glycopeptides,
                columns=COLUMNS_GLYCOPEPTIDE_INFO,
                index=pd.RangeIndex(
                    num_existing_glycopeptides - len(new_glycopeptides),
                    num_existing_glycopeptides,
                ),
            )
            prepared_data[self.CONST.TABLE_GLYCOPEPTIDE_INFO] = new_glycopeptides
        if len(new_precursors) > 0:
            new_precursors = pd.DataFrame(
                new_precursors,
                columns=COLUMNS_PRECURSOR_INFO + [self.CONST.COLUMN_PRECURSOR_MZ],
                index=pd.RangeIndex(
                    num_existing_precursors - len(new_precursors),
                    num_existing_precursors,
                ),
            )
            prepared_data[self.CONST.TABLE_PRECURSOR_INFO] = new_precursors
        new_spectra = pd.DataFrame(
            new_spectra,
            columns=[self.CONST.ID_PRECURSOR_INFO, *self.CONST.COLUMNS_SPECTRUM_INFO],
            index=pd.RangeIndex(
                num_existing_spectra, num_existing_spectra + len(new_spectra)
            ),
        )
        new_spectrum_fragment_indices = (
            index_ragged_list(
                [s[self.CONST.COLUMNS_FRAGMENTS[0]] for s in new_spectrum_fragments]
            )
            + num_existing_spectrum_fragments
        )
        new_spectrum_fragments = pd.concat(
            pd.DataFrame.from_dict(s) for s in new_spectrum_fragments
        ).set_index(
            pd.RangeIndex(
                new_spectrum_fragment_indices[0], new_spectrum_fragment_indices[-1]
            )
        )
        prepared_data.update(
            {
                self.CONST.TABLE_SPECTRUM_INFO: new_spectra,
                self.CONST.TABLE_FRAGMENTS: new_spectrum_fragments,
                self.CONST.INDICES_FRAGMENTS: new_spectrum_fragment_indices,
            }
        )

        prepared_indices = {
            self.CONST.TABLE_PEPTIDE_INFO: pd.Series(
                peptide_indices.values(), index=list(peptide_indices.keys())
            ),
            self.CONST.TABLE_GLYCAN_INFO: pd.Series(
                glycan_indices.values(), index=list(glycan_indices.keys())
            ),
            self.CONST.TABLE_GLYCOPEPTIDE_INFO: pd.Series(
                glycopeptide_indices.values(), index=list(glycopeptide_indices.keys())
            ),
            self.CONST.TABLE_PRECURSOR_INFO: pd.Series(
                precursor_indices.values(), index=list(precursor_indices.keys())
            ),
        }
        return PreparedImportedData(prepared_data, prepared_indices)

    def import_retention_time(self, retention_time: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_retention_time(retention_time)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_retention_time(self, retention_time: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_glycopeptides(
            retention_time[
                retention_time.columns.intersection(
                    [
                        self.CONST.COLUMN_PROTEIN_NAME,
                        self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                        self.CONST.COLUMN_GLYCAN_STRUCT,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                    ]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]
        glycan_indices = prepared_indices[self.CONST.TABLE_GLYCAN_INFO]
        glycopeptide_indices = prepared_indices[self.CONST.TABLE_GLYCOPEPTIDE_INFO]

        if self.CONST.COLUMN_PROTEIN_NAME in retention_time.columns:
            retention_time = retention_time.drop(
                columns=[self.CONST.COLUMN_PROTEIN_NAME]
            )

        retention_time.insert(
            0,
            self.CONST.ID_PEPTIDE_INFO,
            peptide_indices.loc[
                retention_time[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
            ].set_axis(retention_time.index),
        )
        retention_time.insert(
            1,
            self.CONST.ID_GLYCAN_INFO,
            glycan_indices.loc[
                retention_time[self.CONST.COLUMN_GLYCAN_STRUCT]
            ].set_axis(retention_time.index),
        )
        retention_time.insert(
            0,
            self.CONST.ID_GLYCOPEPTIDE_INFO,
            glycopeptide_indices.loc[
                retention_time[
                    [
                        self.CONST.ID_PEPTIDE_INFO,
                        self.CONST.ID_GLYCAN_INFO,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                    ]
                ].itertuples(index=False, name=None)
            ].set_axis(retention_time.index),
        )
        retention_time = retention_time.drop(
            columns=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
                self.CONST.ID_PEPTIDE_INFO,
                self.CONST.ID_GLYCAN_INFO,
                self.CONST.COLUMN_GLYCAN_POSITION,
            ]
        )

        num_existing_retention_time = self.num_retention_time
        if num_existing_retention_time == 0:
            retention_time = retention_time.reset_index(drop=True)
        else:
            retention_time = retention_time.set_index(
                pd.RangeIndex(
                    num_existing_retention_time,
                    num_existing_retention_time + len(retention_time),
                )
            )

        prepared_data[self.CONST.TABLE_RETENTION_TIME] = retention_time
        return PreparedImportedData(prepared_data, prepared_indices)

    def get_retention_time_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        if columns is not None and not self.CONST.ID_GLYCOPEPTIDE_INFO in columns:
            columns = [self.CONST.ID_GLYCOPEPTIDE_INFO] + columns

        retention_time = self.get_retention_time(indices, columns)
        if retention_time is None:
            return None

        if indices is None:
            glycopeptides = self.get_glycopeptide_data()
            assert glycopeptides is not None
            glycopeptides = glycopeptides.loc[
                retention_time[self.CONST.ID_GLYCOPEPTIDE_INFO]
            ]
        else:
            glycopeptides = self.get_glycopeptide_data(
                indices=retention_time[self.CONST.ID_GLYCOPEPTIDE_INFO]
            )
            assert glycopeptides is not None

        glycopeptides = glycopeptides.set_index(retention_time.index)
        retention_time = pd.concat(
            (glycopeptides, retention_time),
            axis=1,
            copy=False,
        )
        return retention_time

    def import_ion_mobility(self, ion_mobility: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_ion_mobility(ion_mobility)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_ion_mobility(self, ion_mobility: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_precursors(
            ion_mobility[
                ion_mobility.columns.intersection(
                    [
                        self.CONST.COLUMN_PROTEIN_NAME,
                        self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                        self.CONST.COLUMN_GLYCAN_STRUCT,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                        self.CONST.COLUMN_PRECURSOR_CHARGE,
                    ]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]
        glycan_indices = prepared_indices[self.CONST.TABLE_GLYCAN_INFO]
        glycopeptide_indices = prepared_indices[self.CONST.TABLE_GLYCOPEPTIDE_INFO]
        precursor_indices = prepared_indices[self.CONST.TABLE_PRECURSOR_INFO]

        if self.CONST.COLUMN_PROTEIN_NAME in ion_mobility.columns:
            ion_mobility = ion_mobility.drop(columns=[self.CONST.COLUMN_PROTEIN_NAME])

        ion_mobility.insert(
            0,
            self.CONST.ID_PEPTIDE_INFO,
            peptide_indices.loc[
                ion_mobility[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
            ].set_index(ion_mobility.index, drop=True),
        )
        ion_mobility.insert(
            1,
            self.CONST.ID_GLYCAN_INFO,
            glycan_indices.loc[ion_mobility[self.CONST.COLUMN_GLYCAN_STRUCT]].set_index(
                ion_mobility.index, drop=True
            ),
        )
        ion_mobility.insert(
            0,
            self.CONST.ID_GLYCOPEPTIDE_INFO,
            glycopeptide_indices.loc[
                ion_mobility[
                    [
                        self.CONST.ID_PEPTIDE_INFO,
                        self.CONST.ID_GLYCAN_INFO,
                        self.CONST.COLUMN_GLYCAN_POSITION,
                    ]
                ].itertuples(index=False, name=None)
            ].set_index(ion_mobility.index, drop=True),
        )
        ion_mobility.insert(
            0,
            self.CONST.ID_PRECURSOR_INFO,
            precursor_indices.loc[
                ion_mobility[
                    [
                        self.CONST.ID_GLYCOPEPTIDE_INFO,
                        self.CONST.COLUMN_PRECURSOR_CHARGE,
                    ]
                ].itertuples(index=False, name=None)
            ].set_index(ion_mobility.index, drop=True),
        )
        ion_mobility = ion_mobility.drop(
            columns=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.COLUMN_GLYCAN_STRUCT,
                self.CONST.ID_PEPTIDE_INFO,
                self.CONST.ID_GLYCAN_INFO,
                self.CONST.COLUMN_GLYCAN_POSITION,
                self.CONST.ID_GLYCOPEPTIDE_INFO,
                self.CONST.COLUMN_PRECURSOR_CHARGE,
            ]
        )

        num_existing_ion_mobility = self.num_ion_mobility
        if num_existing_ion_mobility == 0:
            ion_mobility = ion_mobility.reset_index(drop=True)
        else:
            ion_mobility = ion_mobility.set_index(
                pd.RangeIndex(
                    num_existing_ion_mobility,
                    num_existing_ion_mobility + len(ion_mobility),
                )
            )

        num_existing_ion_mobility = self.num_ion_mobility
        if num_existing_ion_mobility == 0:
            ion_mobility = ion_mobility.reset_index(drop=True)
        else:
            ion_mobility = ion_mobility.set_index(
                pd.RangeIndex(
                    num_existing_ion_mobility,
                    num_existing_ion_mobility + len(ion_mobility),
                )
            )

        prepared_data[self.CONST.TABLE_ION_MOBILITY] = ion_mobility
        return PreparedImportedData(prepared_data, prepared_indices)
