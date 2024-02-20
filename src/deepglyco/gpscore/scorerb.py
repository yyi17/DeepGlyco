import itertools
from logging import Logger
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, cast
import numpy as np
import pandas as pd

from ..chem.gpep.sort import GlycanComparer
from .scorer import GlycoPeptideStructScorer
from ..speclib.gpep.spec import GlycoPeptideMS2SpectrumProto
from ..util.math import jaccard, spectral_angle
from .annotation import GlycoPeptideMS2SpectrumAnnotatorByComposition
from ..specio.spec import MassSpectrum
from ..speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from ..speclib.gpep.inmemory import GlycoPeptideSpectralLibraryInMemory
from ..speclib.gpep.abs import GlycoPeptideSpectralLibraryBase
from ..chem.pep.mods import modified_sequence_to_str
from ..chem.gpep.glycans import GlycanNode, MonosaccharideCollection
from ..deeplib.gpep.ms2b.prediction import GlycoPeptideBranchMS2Predictor
from ..speclib.gpep.parser.strucgp import StrucGPReportParser
from ..util.progress import ProgressFactoryProto


budding_ratio = 0.0
combined_charge = True


class GlycoPeptideStructBranchScorer(GlycoPeptideStructScorer):
    def __init__(
        self,
        monosaccharides: MonosaccharideCollection,
        glycan_comparer: GlycanComparer,
        glycan_composition_map: Mapping[str, Sequence[str]],
        report_parser: StrucGPReportParser,
        ms2_predictor: GlycoPeptideBranchMS2Predictor,
        logger: Optional[Logger] = None,
        progress_factory: Optional[ProgressFactoryProto] = None,
    ):
        super().__init__(
            monosaccharides = monosaccharides,
            glycan_comparer = glycan_comparer,
            report_parser = report_parser,
            ms2_predictor = cast(Any, ms2_predictor),
            glycan_composition_map = glycan_composition_map,
            logger = logger,
            progress_factory = progress_factory,
        )


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
        #intensity = intensity.pow(0.5)

        is_pep_frag = intensity.index.get_level_values("fragment_number") >= 0
        is_gly_frag = intensity.index.get_level_values("fragment_type") == "Y"
        is_glyb_frag = intensity.index.get_level_values("fragment_type") == "B"
        if sum(is_pep_frag) == 0: print(intensity)
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
                                intensity.iloc[:, 0].loc[is_gly_frag | is_glyb_frag].values,
                            ),
                            cast(
                                np.ndarray,
                                intensity.iloc[:, i + 1].loc[is_gly_frag | is_glyb_frag].values,
                            ),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "glyY": [
                        spectral_angle(
                            cast(
                                np.ndarray,
                                intensity.iloc[:, 0].loc[is_gly_frag].values,
                            ),
                            cast(
                                np.ndarray,
                                intensity.iloc[:, i + 1].loc[is_gly_frag].values,
                            ),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                    "glyB": [
                        spectral_angle(
                            cast(
                                np.ndarray,
                                intensity.iloc[:, 0].loc[is_glyb_frag].values,
                            ),
                            cast(
                                np.ndarray,
                                intensity.iloc[:, i + 1].loc[is_glyb_frag].values,
                            ),
                        )
                        for i in range(len(ref_spec_id))
                    ],
                }
            ),
            # "jac": pd.DataFrame.from_dict(
            #     {
            #         "total": [
            #             jaccard(
            #                 cast(np.ndarray, intensity.iloc[:, 0]),
            #                 cast(np.ndarray, intensity.iloc[:, i + 1]),
            #             )
            #             for i in range(len(ref_spec_id))
            #         ],
            #         "pep": [
            #             jaccard(
            #                 cast(np.ndarray, intensity.iloc[:, 0].loc[is_pep_frag]),
            #                 cast(np.ndarray, intensity.iloc[:, i + 1].loc[is_pep_frag]),
            #             )
            #             for i in range(len(ref_spec_id))
            #         ],
            #         "gly": [
            #             jaccard(
            #                 cast(np.ndarray, intensity.iloc[:, 0].loc[is_gly_frag]),
            #                 cast(np.ndarray, intensity.iloc[:, i + 1].loc[is_gly_frag]),
            #             )
            #             for i in range(len(ref_spec_id))
            #         ],
            #         "glyb": [
            #             jaccard(
            #                 cast(np.ndarray, intensity.iloc[:, 0].loc[is_glyb_frag]),
            #                 cast(np.ndarray, intensity.iloc[:, i + 1].loc[is_glyb_frag]),
            #             )
            #             for i in range(len(ref_spec_id))
            #         ],
            #     }
            # ),
        }

        combined_score = pd.DataFrame.from_dict({
            "score": 1 - scores["sa"]["glyY"] + 1 - scores["sa"]["glyB"]
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
        scores.insert(2, "delta_score", scores["score"].diff(-1))
        return scores


