import os

import pandas as pd

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.gpep.gpmass import GlycoPeptideMassCalculator
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModificationCollection
from deepglyco.specio.mzml import MzmlReader
from deepglyco.speclib.gpep.filter import GlycoPeptideSpectrumFilter
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.speclib.gpep.parser.annotation import GlycoPeptideMS2SpectrumAnnotator
from deepglyco.speclib.gpep.parser.strucgp import StrucGPReportParser
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
    ctx.register("mass_calculator", GlycoPeptideMassCalculator)
    ctx.register("annotator", GlycoPeptideMS2SpectrumAnnotator)
    ctx.register("spectrum_filter", GlycoPeptideSpectrumFilter)

    ctx.register("progress_factory", TqdmProgressFactory)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )

    return ctx


def get_report_parser(ctx: Context, report_type: str, **kwargs):
    if report_type.lower() == "strucgp":
        parser = ctx.build(StrucGPReportParser, configs=kwargs.get("config_file", None))
    else:
        raise ValueError(f"unknown report type {report_type}")

    return parser


def import_gpep_speclib(
    dest_file, report_file, spectra_files, report_type="StrucGP", **kwargs
):
    ctx = register_dependencies(
        log_file=rf"{os.path.splitext(dest_file)[0]}.log",
        **kwargs,
    )

    parser = get_report_parser(ctx, report_type, **kwargs)

    logger = ctx.get("logger")
    progress_factory = ctx.get("progress_factory")

    if logger:
        logger.info(f"Use configs: {parser.get_configs()}")

    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=not os.path.exists(dest_file),
    )

    if logger:
        logger.info(f"Loading {report_file}")
    if os.path.splitext(report_file)[1].lower() == '.xlsx':
        report = pd.read_excel(report_file)
    elif os.path.splitext(report_file)[1].lower() == '.csv':
        report = pd.read_csv(report_file)
    else:
        report = pd.read_csv(report_file, sep='\t')

    for spectra_file in spectra_files:
        if logger:
            logger.info(f"Loading {spectra_file}")

        with MzmlReader(spectra_file) as spectra:
            gpspectra = parser.parse_psm_report(report, spectra)
            if progress_factory:
                gpspectra = progress_factory(gpspectra)

            speclib.import_spectra(gpspectra)

        if logger:
            logger.info(
                f"Total: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
                f"{speclib.num_glycopeptides} glycopeptides"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import glycopeptide spectral library from StrucGP report."
    )
    parser.add_argument(
        "--in", dest="report_file", required=True, help="input StrucGP .xlsx report"
    )
    parser.add_argument(
        "--spectra",
        nargs="+",
        dest="spectra_files",
        required=True,
        help="input .mzML spectra file",
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output speclib file"
    )
    parser.add_argument("--config", dest="config_file", help="config file")

    args = parser.parse_args()

    import_gpep_speclib(**vars(args))
