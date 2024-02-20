import itertools
from logging import Logger
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, cast
import numpy as np
import pandas as pd

from deepglyco.chem.gpep.sort import GlycanComparer

from ..speclib.gpep.spec import GlycoPeptideMS2SpectrumProto
from ..util.math import jaccard, spectral_angle
from .annotation import GlycoPeptideMS2SpectrumAnnotatorByComposition
from ..specio.spec import MassSpectrum
from ..speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from ..speclib.gpep.inmemory import GlycoPeptideSpectralLibraryInMemory
from ..speclib.gpep.abs import GlycoPeptideSpectralLibraryBase
from ..chem.pep.mods import modified_sequence_to_str
from ..chem.gpep.glycans import GlycanNode, MonosaccharideCollection
from ..deeplib.gpep.ms2.prediction import GlycoPeptideMS2Predictor
from ..speclib.gpep.parser.strucgp import StrucGPReportParser
from ..util.progress import ProgressFactoryProto


budding_ratio = 0.0
combined_charge = True


def load_glycan_composition_map(
    monosaccharides: MonosaccharideCollection,
    glycan_comparer: GlycanComparer,
    glycan_file: str,
):
    gdb = pd.read_csv(glycan_file, sep="\t")
    gdb.columns = ["glycan_struct"]
    gdb.drop_duplicates(subset=["glycan_struct"], inplace=True)

    gdb["glycan_struct"] = gdb["glycan_struct"].map(
        lambda x: str(glycan_comparer.sort_glycan(GlycanNode.from_str(x)))
    )
    gdb.drop_duplicates(subset=["glycan_struct"], inplace=True)

    gdb["glycan_composition"] = gdb["glycan_struct"].map(
        lambda x: monosaccharides.composition_str(GlycanNode.from_str(x).composition())
    )
    return {
        k: v.tolist() for k, v in gdb.groupby("glycan_composition")["glycan_struct"]
    }


class GlycoPeptideStructScorer:
    def __init__(
        self,
        monosaccharides: MonosaccharideCollection,
        glycan_comparer: GlycanComparer,
        glycan_composition_map: Mapping[str, Sequence[str]],
        report_parser: StrucGPReportParser,
        ms2_predictor: GlycoPeptideMS2Predictor,
        logger: Optional[Logger] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
    ):
        self.monosaccharides = monosaccharides
        self.glycan_comparer = glycan_comparer
        self.parser = report_parser
        self.predictor = ms2_predictor
        self.glycan_composition_map = glycan_composition_map
        self.logger = logger
        self.progress_factory = progress_factory

        assert isinstance(
            self.parser.annotator, GlycoPeptideMS2SpectrumAnnotatorByComposition
        )

    def glycan_struct_to_composition(self, glycan_struct: Union[str, GlycanNode]):
        if isinstance(glycan_struct, str):
            glycan_struct = GlycanNode.from_str(glycan_struct)
        return self.monosaccharides.composition_str(glycan_struct.composition())

    def prepare_glycopeptides(self, report: pd.DataFrame):
        def parse_row(row):
            r: dict[str, Any] = self.parser.get_psm_report_row_info(row)
            r["modified_sequence"] = modified_sequence_to_str(r["modified_sequence"])
            r["glycan_composition"] = self.glycan_struct_to_composition(
                r["glycan_struct"]
            )
            r["glycan_struct"] = str(self.glycan_comparer.sort_glycan(r["glycan_struct"]))
            return r

        report = self.parser.filter_report(report)
        report = pd.DataFrame.from_records(
            [parse_row(row) for i, row in report.iterrows()]
        )

        unknown_composition = ~report["glycan_composition"].isin(
            self.glycan_composition_map.keys()
        )
        if any(unknown_composition):
            if self.logger:
                unknown = report["glycan_composition"].loc[unknown_composition].unique()
                self.logger.warning(
                    f"{sum(unknown_composition)} PSMs with "
                    f"{len(unknown)} glycan compositions not in glycan composition map: "
                    + ",".join(unknown)
                )
            raise ValueError(",".join(
                report["glycan_struct"].loc[unknown_composition].unique()
            ))
            # report = report.loc[~unknown_composition]

        unknown_struct = ~report["glycan_struct"].isin(
            set(itertools.chain.from_iterable(self.glycan_composition_map.values()))
        )
        if any(unknown_struct):
            if self.logger:
                unknown = report["glycan_struct"].loc[unknown_struct].unique()
                self.logger.warning(
                    f"{sum(unknown_struct)} PSMs with "
                    f"{len(unknown)} glycan structs not in glycan composition map: "
                    + ",".join(unknown)
                )
            report = report.loc[~unknown_struct]

        glycopeptides = report[
            [
                "modified_sequence",
                "glycan_composition",
                "glycan_position",
                "precursor_charge",
            ]
        ].drop_duplicates()
        glycopeptides.insert(
            2,
            "glycan_struct",
            [
                self.glycan_composition_map[x]
                for x in glycopeptides["glycan_composition"]
            ],
        )
        return glycopeptides.explode("glycan_struct")

    def load_speclib(self, speclib: GlycoPeptideSpectralLibraryBase):
        self.speclib = speclib

        spectrum_info = speclib.get_spectrum_data(columns=["precursor_id"])
        assert spectrum_info is not None
        spectrum_info["glycan_composition"] = spectrum_info["glycan_struct"].map(
            self.glycan_struct_to_composition
        )
        spectrum_info.reset_index(drop=False, inplace=True)
        spectrum_info.set_index(
            [
                "modified_sequence",
                "glycan_composition",
                "glycan_position",
                "precursor_charge",
                "glycan_struct",
            ],
            inplace=True,
        )
        spectrum_info.sort_index(inplace=True)
        self.ref_spec_map = spectrum_info

    def predict_speclib(
        self,
        pretrained_file: str,
        glycopeptides: pd.DataFrame,
        cache_file: Optional[str] = None,
    ):
        self.predictor.load_model(pretrained_file)

        if self.logger:
            self.logger.info(f"Source: {len(glycopeptides)} precursor entries")

        if cache_file:
            speclib = GlycoPeptideSpectralLibraryHdf(
                file_name=cache_file,
                is_read_only=False,
                is_new_file=True,
            )
        else:
            speclib = GlycoPeptideSpectralLibraryInMemory()

        speclib.import_spectra(
            self.predictor.predict(glycopeptides, keep_zeros=budding_ratio > 0)
        )
        if self.logger:
            self.logger.info(
                f"Total: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, {speclib.num_glycopeptides} glycopeptides"
            )

        self.load_speclib(speclib)
        return speclib

    def extract_spectral_fragments(
        self,
        report: pd.DataFrame,
        spectra: Iterable[MassSpectrum],
        cache_file: Optional[str] = None,
    ):
        parsed_spectra = self.parser.parse_psm_report(report, spectra)
        if self.progress_factory:
            parsed_spectra = self.progress_factory(parsed_spectra, desc="Loading")

        if cache_file:
            speclib = GlycoPeptideSpectralLibraryHdf(
                file_name=cache_file,
                is_read_only=False,
                is_new_file=True,
            )
        else:
            speclib = GlycoPeptideSpectralLibraryInMemory()

        speclib.import_spectra(parsed_spectra)

        return speclib

    def search_speclib(
        self,
        spec: GlycoPeptideMS2SpectrumProto,
    ):
        ref_spec_info = self.ref_spec_map.loc[
            (
                spec.modified_sequence,
                self.glycan_struct_to_composition(spec.glycan_struct),
                spec.glycan_position,
                spec.precursor_charge,
            ),
            :,
        ]

        spec_data = pd.DataFrame.from_dict(
            dict(
                spec.frangment_annotations(),
                intensity=spec.intensity,
            )
        )

        ref_spec_id = ref_spec_info["index"]
        if isinstance(ref_spec_id, pd.Series):
            ref_spec_id = ref_spec_id.tolist()
        else:
            ref_spec_id = [cast(int, ref_spec_id)]

        spectrum_fragments = self.speclib.get_fragments_by_spectrum_id(ref_spec_id)

        fragment_keys = spec_data.columns.difference(["mz", "intensity"]).to_list()
        if combined_charge:
            fragment_keys.remove("fragment_charge")

        spectrum_fragments.insert(0, spec_data)

        intensity_list = []
        for i, frag in enumerate(spectrum_fragments):
            intensity = frag.groupby(by=fragment_keys)["intensity"].sum()

            if budding_ratio and i > 0:
                intensity = intensity.max() * budding_ratio + intensity * (1 - budding_ratio)

            intensity_list.append(intensity)

        intensity = pd.concat(intensity_list, axis=1)
        intensity.fillna(0.0, inplace=True)
        intensity = intensity.loc[intensity.iloc[:, 1:].sum(axis=1) > 0, :]

        is_pep_frag = intensity.index.get_level_values("fragment_number") >= 0

        scores = {
            "sa": pd.DataFrame.from_dict(
                {
                    "total": [
                        spectral_angle(
                            cast(np.ndarray, intensity.iloc[:, 0].values),
                            cast(np.ndarray, intensity.iloc[:, i + 1].values),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "pep": [
                        spectral_angle(
                            cast(
                                np.ndarray, intensity.iloc[:, 0].loc[is_pep_frag].values
                            ),
                            cast(
                                np.ndarray,
                                intensity.iloc[:, i + 1].loc[is_pep_frag].values,
                            ),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "gly": [
                        spectral_angle(
                            cast(
                                np.ndarray,
                                intensity.iloc[:, 0].loc[~is_pep_frag].values,
                            ),
                            cast(
                                np.ndarray,
                                intensity.iloc[:, i + 1].loc[~is_pep_frag].values,
                            ),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                }
            ),
            "jac": pd.DataFrame.from_dict(
                {
                    "total": [
                        jaccard(
                            cast(np.ndarray, intensity.iloc[:, 0]),
                            cast(np.ndarray, intensity.iloc[:, i + 1]),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "pep": [
                        jaccard(
                            cast(np.ndarray, intensity.iloc[:, 0].loc[is_pep_frag]),
                            cast(np.ndarray, intensity.iloc[:, i + 1].loc[is_pep_frag]),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "gly": [
                        jaccard(
                            cast(np.ndarray, intensity.iloc[:, 0].loc[~is_pep_frag]),
                            cast(np.ndarray, intensity.iloc[:, i + 1].loc[~is_pep_frag]),
                        )
                        for i in range(len(ref_spec_id))
                    ]
                }
            ),
        }

        combined_score = pd.DataFrame.from_dict({
            "score": 1 - scores["sa"]["gly"]
            #+ scores["jac"]["gly"]
        })

        scores = pd.concat(
            [
                combined_score,
                *(v.rename(columns=lambda x: f"{k}_{x}") for k, v in scores.items()),
            ],
            axis=1,
            copy=False,
        )
        scores["ref_index"] = ref_spec_id
        scores.sort_values("score", ascending=False, inplace=True)
        scores.insert(0, "rank", 1 + np.arange(0, len(scores)))
        return scores

    def score(
        self,
        parsed_spectra: Union[
            GlycoPeptideSpectralLibraryBase, Iterable[GlycoPeptideMS2SpectrumProto]
        ],
    ):
        if isinstance(parsed_spectra, GlycoPeptideSpectralLibraryBase):
            if self.progress_factory:
                parsed_spectra = self.progress_factory(
                    parsed_spectra.iter_spectra(),
                    desc="Scoring",
                    total=parsed_spectra.num_spectra,
                )
            else:
                parsed_spectra = parsed_spectra.iter_spectra()
        else:
            if self.progress_factory:
                parsed_spectra = self.progress_factory(parsed_spectra, desc="Scoring")

        score_tables = []

        for spec in parsed_spectra:
            scores = self.search_speclib(spec)
            scores.insert(0, "original_glycan_struct", spec.glycan_struct)
            scores.insert(0, "scan_number", spec.scan_number)
            scores.insert(0, "run_name", spec.run_name)
            scores.insert(0, "num_candidates", len(scores))
            score_tables.append(scores)

        result = pd.concat(score_tables, copy=False, ignore_index=True)

        result = result.merge(
            self.ref_spec_map.reset_index(drop=False).set_index("index"),
            left_on="ref_index",
            right_index=True,
            copy=False,
        )
        return result
