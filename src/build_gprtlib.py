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


def import_gpep_rtlib(dest_file, report_file, report_type="StrucGP", **kwargs):
    ctx = register_dependencies(
        log_file=rf"{os.path.splitext(dest_file)[0]}.log",
        **kwargs,
    )

    parser = get_report_parser(ctx, report_type, **kwargs)

    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=not os.path.exists(dest_file),
    )

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {parser.get_configs()}")

    if logger:
        logger.info(f"Loading {report_file}")
    report = pd.read_excel(report_file)

    evidences = parser.parse_evidence_report(report)

    rt = (
        evidences[
            [
                "modified_sequence",
                "glycan_struct",
                "glycan_position",
                "run_name",
                "retention_time",
                "score",
            ]
        ]
        .groupby(
            ["modified_sequence", "glycan_struct", "glycan_position", "run_name"],
            group_keys=False,
            as_index=False,
            sort=False,
            dropna=False,
        )
        .apply(lambda x: x.iloc[x["score"].argmax()])
    )

    speclib.import_retention_time(rt)

    if logger:
        logger.info(
            f"Total: {speclib.num_retention_time} entries of {speclib.num_glycopeptides} glycopeptides"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import glycopeptide retention time library from StrucGP report."
    )
    parser.add_argument(
        "--in", dest="report_file", required=True, help="input StrucGP .xlsx report"
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output rtlib file"
    )
    parser.add_argument("--config", help="config file")

    args = parser.parse_args()

    import_gpep_rtlib(**vars(args))
