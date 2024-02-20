import functools
import itertools
import os
import random
import time
from typing import cast

import pandas as pd


from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.gpep.gpmass import GlycoPeptideMassCalculator
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection

from deepglyco.deeplib.gpep.ms2b.data import GlycoPeptideBranchMS2OutputConverter
from deepglyco.deeplib.gpep.ms2b.prediction import GlycoPeptideBranchMS2Predictor
from deepglyco.chem.gpep.nglycans import NGlycan
from deepglyco.chem.gpep.sort import GlycanComparer
from deepglyco.gpscore.annotation import GlycoPeptideMS2SpectrumAnnotatorByComposition
from deepglyco.gpscore.scorer import load_glycan_composition_map
from deepglyco.gpscore.scorerb import GlycoPeptideStructBranchScorer
from deepglyco.specio.mgf import MgfReader

from deepglyco.specio.mzml import MzmlReader
from deepglyco.speclib.gpep.filter import GlycoPeptideSpectrumFilter
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.speclib.gpep.parser.strucgp import StrucGPReportParser
from deepglyco.speclib.gpep.parser.pglyco import pGlycoReportParser
from deepglyco.util.di import Context
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def register_dependencies(**kwargs):
    ctx = Context()
    ctx.register(
        "elements",
        ElementCollection.load,
        element_file=kwargs.get("element_file", None),
    )
    ctx.register(
        "amino_acids",
        AminoAcidCollection.load,
        amino_acid_file=kwargs.get("amino_acid_file", None),
    )
    ctx.register(
        "modifications",
        ModificationCollection.load,
        modification_file=kwargs.get("modification_file", None),
    )
    ctx.register(
        "monosaccharides",
        MonosaccharideCollection.load,
        monosaccharide_file=kwargs.get("monosaccharide_file", None),
    )
    ctx.register(
        "neutral_loss_types",
        NeutralLossTypeCollection.load,
        neutral_loss_file=kwargs.get("neutral_loss_file", None),
    )
    ctx.register(
        "peptide_fragment_types",
        PeptideFragmentTypeCollection.load,
        peptide_fragment_file=kwargs.get("peptide_fragment_file", None),
    )
    ctx.register(
        "glycan_fragment_types",
        GlycanFragmentTypeCollection.load,
        glycan_fragment_file=kwargs.get("glycan_fragment_file", None),
    )
    ctx.register(
        "glycan_composition_map",
        load_glycan_composition_map,
        glycan_file=kwargs.get("glycan_file", None),
    )
    ctx.register("glycan_comparer", GlycanComparer)
    ctx.register("sequence_parser", ModifiedSequenceParser)
    ctx.register("mass_calculator", GlycoPeptideMassCalculator)
    ctx.register("annotator", GlycoPeptideMS2SpectrumAnnotatorByComposition)
    ctx.register("spectrum_filter", GlycoPeptideSpectrumFilter)
    ctx.register(
        "report_parser",
        get_report_parser_factory(report_type=kwargs.get("report_type", "strucgp")),
        configs=kwargs.get("config_file", None),
    )

    ctx.register("converter", GlycoPeptideBranchMS2OutputConverter)
    ctx.register(
        "ms2_predictor",
        GlycoPeptideBranchMS2Predictor,
    )

    ctx.register("progress_factory", TqdmProgressFactory)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )

    return ctx


def get_report_parser_factory(report_type: str):
    if report_type.lower() == "strucgp":
        return StrucGPReportParser
    elif report_type.lower() == "pglyco":
        return pGlycoReportParser
    else:
        raise ValueError(f"unknown report type {report_type}")


from collections import Counter
from deepglyco.chem.gpep.glycans import GlycanNode

def glycan_properties(glycan_struct: pd.Series):
    def _glycan_properties(glycan_struct: str):
        ng = NGlycan(GlycanNode.from_str(glycan_struct))

        if ng.is_high_mannose:
            glycan_type = "high_mannose"
        elif ng.is_complex:
            glycan_type = "complex"
        elif ng.is_hybrid:
            glycan_type = "hybrid"
        else:
            glycan_type = "unknown"

        core_fucoses = ng.core_fucoses
        if core_fucoses is not None:
            core_fucoses = str(core_fucoses)
        else:
            core_fucoses = ""

        bisection = ng.bisection
        if bisection is not None:
            bisection = str(bisection)
        else:
            bisection = ""

        branches = ng.branches
        if branches is not None:
            branches = tuple(str(g) for g in itertools.chain.from_iterable(branches))

        return pd.Series(
            [glycan_type, core_fucoses, bisection, branches],
            index=["glycan_type", "core_fucoses", "bisection", "branches"],
        )

    unique_glycan_struct = glycan_struct.drop_duplicates()
    glycan_properties = unique_glycan_struct.apply(_glycan_properties)
    glycan_properties.set_index(unique_glycan_struct, inplace=True)
    glycan_properties = glycan_properties.loc[glycan_struct]
    glycan_properties.set_index(result.index, inplace=True)
    return glycan_properties


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Rescore GPSMs using predicted spectra.'
    )
    parser.add_argument(
        '--model', required=True,
        help='pretrained model file'
    )
    parser.add_argument(
        '--gpsm', nargs='+', required=True,
        help='input GPSM files'
    )
    parser.add_argument(
        '--spec', nargs='+', required=True,
        help='input spectra files'
    )
    parser.add_argument(
        '--gdb', required=True,
        help='glycan structure space file'
    )
    parser.add_argument(
        '--out',
        help='output result file'
    )
    parser.add_argument(
        '--pred',
        help='temp predict spectral library file'
    )
    parser.add_argument(
        '--type', choices=["strucgp", "pglyco"], default="strucgp",
        help='input GPSM file type'
    )
    parser.add_argument(
        '--config',
        help='config file'
    )

    args = parser.parse_args()

    pretrained_file = args.model
    report_type = args.type
    config_file = args.config
    if config_file is None:
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "deepglyco",
            "gpscore",
            "gpscore_pglyco_branch.yaml" if report_type == "pglyco" else "gpscore_strucgp_branch.yaml",
        )

    glycan_file = args.gdb
    report_files = args.gpsm
    spectra_files = args.spec
    assert report_files is not None and len(report_files) > 0
    assert spectra_files is not None and len(spectra_files) > 0

    result_file = args.out
    if result_file is None:
        result_file = f"gpscore_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(0, 99999):05d}_result.csv"

    pred_speclib_file = args.pred
    if pred_speclib_file is None:
        pred_speclib_file = os.path.splitext(result_file)[0]
        if pred_speclib_file.endswith(".result"):
            pred_speclib_file = os.path.splitext(pred_speclib_file)[0]
        pred_speclib_file += "_prediction.speclib.h5"

    ctx = register_dependencies(
        report_type=report_type,
        config_file=config_file,
        glycan_file=glycan_file,
    )

    scorer = ctx.build(GlycoPeptideStructBranchScorer)
    logger = ctx.get("logger")

    report_list = []
    for report_file in report_files:
        logger.info(f"Loading {report_file}")
        if os.path.splitext(report_file)[1].lower() == '.xlsx':
            report = pd.read_excel(report_file)
        elif os.path.splitext(report_file)[1].lower() == '.csv':
            report = pd.read_csv(report_file)
        else:
            report = pd.read_csv(report_file, sep='\t')
        report_list.append(report)

    report = pd.concat(report_list, ignore_index=True, copy=False)
    glycopeptides = scorer.prepare_glycopeptides(report)

    if not os.path.exists(pred_speclib_file):
        speclib = scorer.predict_speclib(
            glycopeptides=glycopeptides,
            pretrained_file=pretrained_file,
            cache_file=pred_speclib_file,
        )
    else:
        speclib = GlycoPeptideSpectralLibraryHdf(
            file_name=pred_speclib_file,
            is_read_only=True,
        )
        scorer.load_speclib(speclib)

    result_list = []
    for spectra_file in spectra_files:
        cache_spectra_file = f"{os.path.splitext(spectra_file)[0]}.spec.h5"
        if not os.path.exists(cache_spectra_file):
            logger.info(f"Loading {spectra_file}")
            if os.path.splitext(spectra_file)[1].lower() == '.mgf':
                reader_cls = MgfReader
            else:
                reader_cls = MzmlReader
            with reader_cls(spectra_file) as spectra:
                spectra = scorer.extract_spectral_fragments(
                    report=report,
                    spectra=spectra,
                    cache_file=cache_spectra_file,
                )
        else:
            logger.info(f"Loading {cache_spectra_file}")
            spectra = GlycoPeptideSpectralLibraryHdf(
                file_name=cache_spectra_file,
                is_read_only=True,
            )

        result = scorer.score(spectra)
        result_list.append(result)

    result = pd.concat(result_list, copy=False, ignore_index=True)

    result.to_csv(result_file, index=False)

    result = pd.concat(
        (result, glycan_properties(result["glycan_struct"])),
        axis=1,
        copy=False,
    )
    result = pd.concat(
        (
            result,
            glycan_properties(result["original_glycan_struct"])
            .rename(columns=lambda x: f"original_{x}"),
        ),
        axis=1,
        copy=False,
    )
    for column in ["glycan_type", "core_fucoses", "bisection"]:
        result["match_" + column] = result[column] == result["original_" + column]
    result["match_branches"] = result[["branches", "original_branches"]].apply(
        lambda x: Counter(x[0]) == Counter(x[1]), axis=1
    )

    # from glycoglyph import GlycoGlyphGlycanFormatter

    # gfmt = GlycoGlyphGlycanFormatter()
    # result["cfg_name"] = result["glycan_struct"].apply(gfmt.glycoglyph_name)
    # result["original_cfg_name"] = result["original_glycan_struct"].apply(
    #     gfmt.glycoglyph_name
    # )

    result.to_csv(result_file, index=False)
