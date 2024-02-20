__all__ = [
    "PeptideLibraryBase",
    "PeptideSpectralLibraryBase",
    "PreparedImportedData",
]

import abc
import itertools
from typing import Dict, Iterable, List, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from ...util.collections.list import index_ragged_list
from ..common.rtlib import RetentionTimeLibraryBase
from ..common.speclib import Columns, Indices, IndicesNonSlice, SpectralLibraryBase
from .spec import PeptideMS2Spectrum, PeptideMS2SpectrumProto

PD_INDEX_NAME = "index"


class PreparedImportedData(NamedTuple):
    data: Dict[str, Union[pd.DataFrame, npt.NDArray]]
    indices: Dict[str, pd.Series]


class PeptideLibraryBase:
    class CONST:
        TABLE_PROTEIN_INFO = "protein_info"
        COLUMN_PROTEIN_NAME = "protein_name"
        ID_PROTEIN_INFO = "protein_id"
        TABLE_PEPTIDE_INFO = "peptide_info"
        COLUMN_PEPTIDE_SEQUENCE = "modified_sequence"
        ID_PEPTIDE_INFO = "peptide_id"
        TABLE_PEPTIDE_PROTEIN_MAP = "peptide_protein_map"
        TABLE_PRECURSOR_INFO = "precursor_info"
        COLUMN_PRECURSOR_CHARGE = "precursor_charge"
        COLUMN_PRECURSOR_MZ = "precursor_mz"

    @abc.abstractmethod
    def get_protein_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_peptide_info(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_peptide_protein_map(
        self, indices: Optional[Indices] = None
    ) -> Optional[pd.DataFrame]:
        pass

    @property
    def num_proteins(self) -> int:
        data = self.get_protein_info()
        if data is not None:
            return len(data)
        else:
            return 0

    @property
    def num_peptides(self) -> int:
        data = self.get_peptide_info()
        if data is not None:
            return len(data)
        else:
            return 0

    @abc.abstractmethod
    def _append_prepared_data(self, prepared_data: dict):
        pass

    def import_proteins(self, proteins: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_proteins(proteins)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_proteins(self, proteins: pd.DataFrame):
        proteins = proteins.dropna(
            subset=[self.CONST.COLUMN_PROTEIN_NAME]
        ).drop_duplicates(subset=[self.CONST.COLUMN_PROTEIN_NAME])

        existing_proteins = self.get_protein_info(
            columns=[self.CONST.COLUMN_PROTEIN_NAME]
        )
        if existing_proteins is None:
            new_proteins = proteins.reset_index(drop=True)
            protein_indices = (
                new_proteins[[self.CONST.COLUMN_PROTEIN_NAME]]
                .reset_index(drop=False)
                .set_index(self.CONST.COLUMN_PROTEIN_NAME)[PD_INDEX_NAME]
            )
            return PreparedImportedData(
                {self.CONST.TABLE_PROTEIN_INFO: new_proteins},
                {self.CONST.TABLE_PROTEIN_INFO: protein_indices},
            )

        new_proteins = proteins.loc[
            ~proteins[self.CONST.COLUMN_PROTEIN_NAME].isin(
                existing_proteins[self.CONST.COLUMN_PROTEIN_NAME]
            )
        ]
        num_existing_proteins = len(existing_proteins)
        new_proteins = new_proteins.set_index(
            pd.RangeIndex(
                num_existing_proteins, num_existing_proteins + len(new_proteins)
            )
        )
        protein_indices = (
            pd.concat(
                (existing_proteins, new_proteins[[self.CONST.COLUMN_PROTEIN_NAME]]),
                axis=0,
                ignore_index=False,
                copy=False,
            )
            .reset_index(drop=False)
            .set_index(self.CONST.COLUMN_PROTEIN_NAME)[PD_INDEX_NAME]
        )
        return PreparedImportedData(
            {self.CONST.TABLE_PROTEIN_INFO: new_proteins},
            {self.CONST.TABLE_PROTEIN_INFO: protein_indices},
        )

    def import_peptides(self, peptides: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_peptides(peptides)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_peptides(self, peptides: pd.DataFrame):
        if self.CONST.COLUMN_PROTEIN_NAME in peptides.columns:
            if (
                peptides[self.CONST.COLUMN_PROTEIN_NAME]
                .map(lambda x: isinstance(x, list), na_action="ignore")
                .any()
            ):
                protein_name = pd.DataFrame(
                    {
                        self.CONST.COLUMN_PROTEIN_NAME: list(
                            itertools.chain.from_iterable(
                                peptides[self.CONST.COLUMN_PROTEIN_NAME].map(
                                    lambda x: x if isinstance(x, list) else [x],
                                    na_action="ignore",
                                )
                            )
                        )
                    }
                )
            else:
                protein_name = peptides[[self.CONST.COLUMN_PROTEIN_NAME]]

            prepared_data, prepared_indices = self._prepare_imported_proteins(
                protein_name
            )
            protein_indices = prepared_indices[self.CONST.TABLE_PROTEIN_INFO]
        else:
            prepared_data = {}
            prepared_indices = {}
            protein_indices = None

        existing_peptides = self.get_peptide_info(
            columns=[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
        )
        if existing_peptides is None:
            new_peptides = peptides.drop_duplicates(
                subset=[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
            ).reset_index(drop=True)
            peptide_indices = (
                new_peptides[[self.CONST.COLUMN_PEPTIDE_SEQUENCE]]
                .reset_index(drop=False)
                .set_index(self.CONST.COLUMN_PEPTIDE_SEQUENCE)[PD_INDEX_NAME]
            )
        else:
            num_existing_peptides = len(existing_peptides)
            new_peptides = peptides.loc[
                ~peptides[self.CONST.COLUMN_PEPTIDE_SEQUENCE].isin(
                    existing_peptides[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
                )
            ].drop_duplicates(subset=[self.CONST.COLUMN_PEPTIDE_SEQUENCE])
            new_peptides = new_peptides.set_index(
                pd.RangeIndex(
                    num_existing_peptides, num_existing_peptides + len(new_peptides)
                )
            )
            peptide_indices = (
                pd.concat(
                    (
                        existing_peptides,
                        new_peptides[[self.CONST.COLUMN_PEPTIDE_SEQUENCE]],
                    ),
                    axis=0,
                    ignore_index=False,
                    copy=False,
                )
                .reset_index(drop=False)
                .set_index(self.CONST.COLUMN_PEPTIDE_SEQUENCE)[PD_INDEX_NAME]
            )

        if protein_indices is not None:
            existing_peptide_protein_map = self.get_peptide_protein_map()
            if existing_peptide_protein_map is not None:
                num_existing_peptide_protein_map = len(existing_peptide_protein_map)
                existing_peptide_protein_map = set(
                    existing_peptide_protein_map.itertuples(index=False, name=None)
                )
            else:
                num_existing_peptide_protein_map = 0
                existing_peptide_protein_map = set()

            new_peptide_protein_map = []
            for pep, prot in (
                peptides[
                    [self.CONST.COLUMN_PEPTIDE_SEQUENCE, self.CONST.COLUMN_PROTEIN_NAME]
                ]
                .dropna()
                .itertuples(index=False, name=None)
            ):
                pep_id = peptide_indices.loc[pep]
                prot_id = protein_indices.loc[prot]
                if isinstance(prot_id, list):
                    for p in prot_id:
                        if (pep_id, p) not in existing_peptide_protein_map:
                            new_peptide_protein_map.append((pep_id, p))
                            existing_peptide_protein_map.add((pep_id, p))
                else:
                    if (pep_id, prot_id) not in existing_peptide_protein_map:
                        new_peptide_protein_map.append((pep_id, prot_id))
                        existing_peptide_protein_map.add((pep_id, prot_id))

            new_peptide_protein_map = pd.DataFrame(
                new_peptide_protein_map,
                columns=[self.CONST.ID_PEPTIDE_INFO, self.CONST.ID_PROTEIN_INFO],
                index=pd.RangeIndex(
                    num_existing_peptide_protein_map,
                    num_existing_peptide_protein_map + len(new_peptide_protein_map),
                ),
            )
            prepared_data[
                self.CONST.TABLE_PEPTIDE_PROTEIN_MAP
            ] = new_peptide_protein_map
            new_peptides = new_peptides.drop(columns=[self.CONST.COLUMN_PROTEIN_NAME])

        prepared_data[self.CONST.TABLE_PEPTIDE_INFO] = new_peptides
        prepared_indices[self.CONST.TABLE_PEPTIDE_INFO] = peptide_indices
        return PreparedImportedData(prepared_data, prepared_indices)


class PeptideSpectralLibraryBase(
    PeptideLibraryBase,
    SpectralLibraryBase[PeptideMS2Spectrum],
    RetentionTimeLibraryBase,
):
    class CONST(PeptideLibraryBase.CONST):
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
        )
        INDICES_FRAGMENTS = "spectrum_fragment_indices"
        TABLE_RETENTION_TIME = "retention_time"
        TABLE_ION_MOBILITY = "ion_mobility"

    def get_precursor_data(
        self,
        indices: Optional[Indices] = None,
        columns: Optional[Columns] = None,
    ):
        if columns is not None and not self.CONST.ID_PEPTIDE_INFO in columns:
            columns = [self.CONST.ID_PEPTIDE_INFO] + columns

        data = self.get_precursor_info(indices, columns)
        if data is None:
            return data

        if indices is None:
            peptides = self.get_peptide_info()
            assert peptides is not None
            peptides = peptides.loc[data[self.CONST.ID_PEPTIDE_INFO]]
        else:
            peptides = self.get_peptide_info(
                indices=data[self.CONST.ID_PEPTIDE_INFO]
            )
            assert peptides is not None

        peptides = peptides.set_index(data.index)
        data = pd.concat(
            (peptides, data),
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

    def get_spectra(
        self, spectrum_id_list: IndicesNonSlice
    ) -> List[PeptideMS2Spectrum]:
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
        peptide_info = self.get_peptide_info(
            indices=precursor_info[self.CONST.ID_PEPTIDE_INFO].astype(np.int64)
        )
        if peptide_info is None:
            raise ValueError("empty peptide info")
        peptide_info = peptide_info.astype(object)

        precursor_info = precursor_info.drop(columns=[self.CONST.ID_PEPTIDE_INFO])
        spectrum_info = spectrum_info.drop(columns=[self.CONST.ID_PRECURSOR_INFO])
        spectra = []
        for i in range(0, len(spectrum_info)):
            arg_dict = {}
            arg_dict.update(peptide_info.iloc[i])
            arg_dict.update(precursor_info.iloc[i])
            arg_dict.update(spectrum_info.iloc[i])
            arg_dict.update(
                {col: series.values for col, series in fragments[i].items()}
            )
            spectra.append(PeptideMS2Spectrum(**arg_dict))
        return spectra

    def iter_spectra(self) -> Iterable[PeptideMS2Spectrum]:
        spectrum_info = self.get_spectrum_info()
        if spectrum_info is None:
            return
        spectrum_info = spectrum_info.astype(object)
        fragments = self.get_spectrum_fragments()
        if fragments is None:
            return
        fragment_indices = self.get_spectrum_fragment_indices()
        if fragment_indices is None:
            return
        precursor_info = self.get_precursor_info()
        if precursor_info is None:
            return
        precursor_info = precursor_info.astype(object)
        peptide_info = self.get_peptide_info()
        if peptide_info is None:
            return
        peptide_info = peptide_info.astype(object)

        for spec_id in range(0, len(spectrum_info)):
            prec_id = spectrum_info.iloc[spec_id][self.CONST.ID_PRECURSOR_INFO]
            pept_id = precursor_info.iloc[prec_id][self.CONST.ID_PEPTIDE_INFO]
            frag_id_se = (fragment_indices[spec_id], fragment_indices[spec_id + 1])
            arg_dict = {}
            arg_dict.update(peptide_info.iloc[pept_id])
            arg_dict.update(precursor_info.iloc[prec_id])
            arg_dict.update(spectrum_info.iloc[spec_id])
            arg_dict.pop(self.CONST.ID_PEPTIDE_INFO)
            arg_dict.pop(self.CONST.ID_PRECURSOR_INFO)
            arg_dict.update(
                {
                    col: series.values
                    for col, series in fragments[frag_id_se[0] : frag_id_se[1]].items()
                }
            )
            yield PeptideMS2Spectrum(**arg_dict)

    def import_precursors(self, precursors: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_precursors(precursors)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_precursors(self, precursors: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_peptides(
            precursors[
                precursors.columns.intersection(
                    [self.CONST.COLUMN_PROTEIN_NAME, self.CONST.COLUMN_PEPTIDE_SEQUENCE]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]

        if self.CONST.COLUMN_PROTEIN_NAME in precursors.columns:
            precursors = precursors.drop(columns=[self.CONST.COLUMN_PROTEIN_NAME])

        COLUMNS_PRECURSOR_INFO = [
            self.CONST.ID_PEPTIDE_INFO,
            self.CONST.COLUMN_PRECURSOR_CHARGE,
        ]
        precursors = precursors.drop_duplicates(
            subset=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
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
            columns=[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
        )
        prepared_data[self.CONST.TABLE_PRECURSOR_INFO] = new_precursors
        prepared_indices[self.CONST.TABLE_PRECURSOR_INFO] = precursor_indices
        return PreparedImportedData(prepared_data, prepared_indices)

    def import_spectra(self, spectra: Iterable[PeptideMS2SpectrumProto]):
        prepared_data, _ = self._prepare_imported_spectra(spectra)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_spectra(self, spectra: Iterable[PeptideMS2SpectrumProto]):
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

        COLUMNS_PRECURSOR_INFO = [
            self.CONST.ID_PEPTIDE_INFO,
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

            prec_id = precursor_indices.get((pep_id, spec.precursor_charge), None)
            if prec_id is None:
                new_precursors.append(
                    (pep_id, spec.precursor_charge, spec.precursor_mz)
                )
                prec_id = num_existing_precursors
                precursor_indices[(pep_id, spec.precursor_charge)] = prec_id
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
        if len(new_precursors) > 0:
            new_precursors = pd.DataFrame(
                new_precursors,
                columns=[*COLUMNS_PRECURSOR_INFO, self.CONST.COLUMN_PRECURSOR_MZ],
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
            self.CONST.TABLE_PRECURSOR_INFO: pd.Series(
                precursor_indices.values(), index=list(precursor_indices.keys())
            ),
        }
        return PreparedImportedData(prepared_data, prepared_indices)

    def import_retention_time(self, retention_time: pd.DataFrame):
        prepared_data, _ = self._prepare_imported_retention_time(retention_time)
        self._append_prepared_data(prepared_data)

    def _prepare_imported_retention_time(self, retention_time: pd.DataFrame):
        prepared_data, prepared_indices = self._prepare_imported_peptides(
            retention_time[
                retention_time.columns.intersection(
                    [self.CONST.COLUMN_PROTEIN_NAME, self.CONST.COLUMN_PEPTIDE_SEQUENCE]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]

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
        retention_time = retention_time.drop(
            columns=[self.CONST.COLUMN_PEPTIDE_SEQUENCE]
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
        self, indices: Optional[Indices] = None
    ) -> Optional[pd.DataFrame]:
        retention_time = self.get_retention_time(indices)
        if retention_time is None:
            return None

        if indices is None:
            peptide_info = self.get_peptide_info()
            assert peptide_info is not None
            peptide_info = peptide_info.loc[retention_time[self.CONST.ID_PEPTIDE_INFO]]
        else:
            peptide_info = self.get_peptide_info(
                indices=retention_time[self.CONST.ID_PEPTIDE_INFO]
            )
            assert peptide_info is not None

        peptide_info = peptide_info.set_index(retention_time.index)
        retention_time = pd.concat(
            (peptide_info, retention_time.drop(columns=[self.CONST.ID_PEPTIDE_INFO])),
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
                        self.CONST.COLUMN_PRECURSOR_CHARGE,
                    ]
                )
            ]
        )
        peptide_indices = prepared_indices[self.CONST.TABLE_PEPTIDE_INFO]
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
            0,
            self.CONST.ID_PRECURSOR_INFO,
            precursor_indices.loc[
                ion_mobility[
                    [self.CONST.ID_PEPTIDE_INFO, self.CONST.COLUMN_PRECURSOR_CHARGE]
                ].itertuples(index=False, name=None)
            ].set_index(ion_mobility.index, drop=True),
        )
        ion_mobility = ion_mobility.drop(
            columns=[
                self.CONST.COLUMN_PEPTIDE_SEQUENCE,
                self.CONST.ID_PEPTIDE_INFO,
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

        prepared_data[self.CONST.TABLE_ION_MOBILITY] = ion_mobility
        return PreparedImportedData(prepared_data, prepared_indices)
