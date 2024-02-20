__all__ = [
    "NonRedundantSpectraProto",
    "NonRedundantSpectraFirstOccurrence",
    "NonRedundantSpectraBestScore",
    "NonRedundantSpectraConsensus",
]

import os
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ...util.config import Configurable

from ...util.math import dot_product
from ...util.table import merge_dataframes
from .spec import MS2SpectrumProto
from .speclib import SpectralLibraryBase

MS2SpectrumType = TypeVar("MS2SpectrumType", bound=MS2SpectrumProto)

PD_INDEX_NAME = "index"
ID_PRECURSOR_INFO = "precursor_id"
COLUMN_SPECTRUM_SCORE = "score"
COLUMN_MZ = "mz"
COLUMN_INTENSITY = "intensity"


class NonRedundantSpectraProto(Generic[MS2SpectrumType], Protocol):
    def nonredundant_spectra(
        self, speclib: SpectralLibraryBase[MS2SpectrumType]
    ) -> Iterable[MS2SpectrumType]:
        ...


class NonRedundantSpectraFirstOccurrence:
    def nonredundant_spectra(
        self, speclib: SpectralLibraryBase[MS2SpectrumType]
    ) -> Iterable[MS2SpectrumType]:
        spectrum_info = speclib.get_spectrum_info(columns=[ID_PRECURSOR_INFO])
        if spectrum_info is None:
            return []

        spectrum_precursor_map = spectrum_info.reset_index()
        spectrum_indices = spectrum_precursor_map.drop_duplicates(
            subset=[ID_PRECURSOR_INFO]
        )[PD_INDEX_NAME]
        return speclib.get_spectra(spectrum_indices)


class NonRedundantSpectraBestScore:
    def nonredundant_spectra(
        self, speclib: SpectralLibraryBase[MS2SpectrumType]
    ) -> Iterable[MS2SpectrumType]:
        spectrum_info = speclib.get_spectrum_info(
            columns=[ID_PRECURSOR_INFO, COLUMN_SPECTRUM_SCORE]
        )
        if spectrum_info is None:
            return []

        spectrum_precursor_map = spectrum_info.reset_index()
        spectrum_indices: pd.Series = spectrum_precursor_map.groupby(
            ID_PRECURSOR_INFO
        ).apply(
            lambda data: data[PD_INDEX_NAME].iloc[data[COLUMN_SPECTRUM_SCORE].argmax()]
        )
        return speclib.get_spectra(spectrum_indices)


class NonRedundantSpectraConsensus(Configurable):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "consensus.yaml"
            )
        super().__init__(configs)

    def nonredundant_spectra(
        self, speclib: SpectralLibraryBase[MS2SpectrumType]
    ) -> Iterable[MS2SpectrumType]:
        max_replicates_combined = self.get_config("max_replicates_combined", typed=int)
        min_replicates_combined = self.get_config("min_replicates_combined", typed=int)

        spectrum_info = speclib.get_spectrum_info(
            columns=[ID_PRECURSOR_INFO, COLUMN_SPECTRUM_SCORE]
        )
        if spectrum_info is None:
            return []

        spectrum_precursor_map = spectrum_info.reset_index()
        for prec_id, data in spectrum_precursor_map.groupby(ID_PRECURSOR_INFO):
            if len(data) > max_replicates_combined:
                data = data.sort_values(by=COLUMN_SPECTRUM_SCORE, ascending=False).head(
                    max_replicates_combined
                )

            spec_id = data[PD_INDEX_NAME]
            if len(spec_id) == 1:
                yield speclib.get_spectra(spec_id)[0]
                continue

            if len(spec_id) < min_replicates_combined:
                spec_id = [
                    data[PD_INDEX_NAME].iloc[data[COLUMN_SPECTRUM_SCORE].argmax()]
                ]
                yield speclib.get_spectra(spec_id)[0]
                continue

            spectrum_fragments = speclib.get_fragments_by_spectrum_id(spec_id)
            combined_fragments = self.combine_replicated_spectra(
                data, spectrum_fragments
            )
            if combined_fragments is None:
                continue
            template = speclib.get_spectra([spec_id.iloc[0]])[0]
            arg_dict: dict[Any, Any] = dict(template.analyte_info())
            arg_dict.update(template.spectrum_metadata())
            arg_dict.update({k: v.values for k, v in combined_fragments.items()})
            yield template.__class__(**arg_dict)  # type: ignore

    def combine_replicated_spectra(
        self, spectrum_info: pd.DataFrame, spectrum_fragments: List[pd.DataFrame]
    ):
        replicate_similarity_threshold = self.get_config(
            "replicate_similarity_threshold", typed=float, allow_convert=True
        )
        peak_quorum = self.get_config("peak_quorum", typed=float, allow_convert=True)
        use_score_weight = self.get_config("use_score_weight", typed=bool)

        fragment_keys = (
            spectrum_fragments[0]
            .columns.difference([COLUMN_MZ, COLUMN_INTENSITY])
            .to_list()
        )
        aligned_fragments = merge_dataframes(
            (
                data.drop_duplicates(fragment_keys).rename(
                    columns={
                        col: f"{col}_{i}"
                        for col in data.columns.difference(fragment_keys)
                    },
                    copy=False,
                )
                for i, data in enumerate(spectrum_fragments)
            ),
            how="outer",
            on=fragment_keys,
            copy=False,
        )
        fragment_table = aligned_fragments[fragment_keys]
        mz_table = aligned_fragments[
            [f"{COLUMN_MZ}_{i}" for i in range(len(spectrum_fragments))]
        ]
        intensity_table = aligned_fragments[
            [f"{COLUMN_INTENSITY}_{i}" for i in range(len(spectrum_fragments))]
        ]

        if replicate_similarity_threshold > 0:
            best_index = cast(int, spectrum_info[COLUMN_SPECTRUM_SCORE].argmax())
            similarity = np.array(
                [
                    dot_product(
                        intensity_table.iloc[:, best_index].values.astype(np.float_),
                        intensity_table.iloc[:, i].values.astype(np.float_),
                    )
                    if i != best_index
                    else 1.0
                    for i in range(len(spectrum_fragments) - 1)
                ]
            )
            kept_indices = np.where(similarity > replicate_similarity_threshold)[0]
            if len(kept_indices) == 0:
                return None
            spectrum_info = spectrum_info.iloc[kept_indices]
            mz_table = mz_table.iloc[:, kept_indices]
            intensity_table = intensity_table.iloc[:, kept_indices]

        if peak_quorum > 0:
            peak_quorum_ = 1 - (mz_table.isna() | intensity_table.isna()).mean(axis=1)
            should_keep = peak_quorum_ > peak_quorum
            fragment_table = fragment_table.loc[should_keep]
            if len(fragment_table) == 0:
                return None
            mz_table = mz_table.loc[should_keep]
            intensity_table = intensity_table.loc[should_keep]

        if use_score_weight:
            score_weight = (
                spectrum_info[COLUMN_SPECTRUM_SCORE]
                .divide(spectrum_info[COLUMN_SPECTRUM_SCORE].sum(skipna=True))
                .values.astype(np.float_)
            )
            intensity = intensity_table.multiply(score_weight, axis=1).sum(
                axis=1, skipna=True
            )
            mz = mz_table.multiply(score_weight, axis=1).sum(axis=1, skipna=True).divide(
                (~mz_table.isna()).astype(np.int_).multiply(score_weight, axis=1).sum(axis=1, skipna=True),
                axis=0
            )

        else:
            intensity = intensity_table.divide(len(spectrum_info)).sum(
                axis=1, skipna=True
            )
            mz = mz_table.divide(
                (~mz_table.isna()).astype(np.int_).sum(axis=1, skipna=True),
                axis=0,
            ).sum(axis=1, skipna=True)


        fragment_table.insert(fragment_table.shape[1], COLUMN_MZ, mz)
        fragment_table.insert(fragment_table.shape[1], COLUMN_INTENSITY, intensity)
        fragment_table = fragment_table.dropna(
            subset=[COLUMN_MZ, COLUMN_INTENSITY]
        ).reset_index(drop=True)
        return fragment_table

    def merge_spectra(
        self, spectra: Iterable[MS2SpectrumType]
    ) -> Optional[MS2SpectrumType]:
        template = None
        scores = []
        spectrum_fragments = []
        for spec in spectra:
            if template is None or (
                spec.score is not None
                and (template.score is None or template.score < spec.score)
            ):
                template = spec
            scores.append(spec.score)
            d = dict(mz=spec.mz, intensity=spec.intensity)
            d.update(spec.frangment_annotations())
            spectrum_fragments.append(pd.DataFrame.from_dict(d))

        if template is None:
            raise ValueError("empty spectra sequence")

        spectrum_info = pd.DataFrame.from_dict({COLUMN_SPECTRUM_SCORE: scores})
        combined_fragments = self.combine_replicated_spectra(
            spectrum_info, spectrum_fragments
        )
        if combined_fragments is None:
            return None

        arg_dict: dict[Any, Any] = dict(template.analyte_info())
        arg_dict.update(template.spectrum_metadata())
        arg_dict.update({k: v.values for k, v in combined_fragments.items()})
        return template.__class__(**arg_dict)  # type: ignore
