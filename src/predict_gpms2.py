import os
import pandas as pd

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.gpep.gpmass import GlycoPeptideMassCalculator
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection
from deepglyco.deeplib.gpep.ms2.data import GlycoPeptideMS2OutputConverter
from deepglyco.deeplib.gpep.ms2.prediction import GlycoPeptideMS2Predictor
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
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
    ctx.register("sequence_parser", ModifiedSequenceParser)
    ctx.register("mass_calculator", GlycoPeptideMassCalculator)

    ctx.register("converter", GlycoPeptideMS2OutputConverter)

    ctx.register("progress_factory", TqdmProgressFactory)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )
    return ctx


def load_glycopeptides(glycopeptide_file):
    if glycopeptide_file.endswith(".csv"):
        glycopeptides = pd.read_csv(glycopeptide_file, low_memory=False)
    elif glycopeptide_file.endswith(".tsv"):
        glycopeptides = pd.read_csv(glycopeptide_file, sep="\t", low_memory=False)
    elif glycopeptide_file.endswith(".speclib.h5"):
        speclib = GlycoPeptideSpectralLibraryHdf(
            file_name=glycopeptide_file,
            is_read_only=True,
        )
        glycopeptides = speclib.get_spectrum_data()
        # glycopeptides = speclib.get_precursor_data()
        assert glycopeptides is not None
    else:
        raise NotImplementedError
    return glycopeptides


def predict_gpep_ms2(pretrained_file, glycopeptide_file, out_file, **kwargs):
    ctx = register_dependencies(**kwargs)

    predictor = ctx.build(GlycoPeptideMS2Predictor)
    predictor.load_model(pretrained_file)

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {predictor.get_configs()}")

    glycopeptides = load_glycopeptides(glycopeptide_file)

    if logger:
        logger.info(f"Source: {len(glycopeptides)} precursor entries")

    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=out_file,
        is_read_only=False,
        is_new_file=not os.path.exists(out_file),
    )

    speclib.import_spectra(predictor.predict(glycopeptides))

    if logger:
        logger.info(
            f"Total: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
            f"{speclib.num_glycopeptides} glycopeptides"
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Predict glycopeptide MS2 spectra.")
    parser.add_argument("--in", required=True, help="input glycopeptide file")
    parser.add_argument("--model", required=True, help="pretrained model file")
    parser.add_argument("--out", help="output spectral library file")

    args = parser.parse_args()
    glycopeptide_file = getattr(args, "in")
    out_file = args.out
    pretrained_file = args.model

    predict_gpep_ms2(pretrained_file, glycopeptide_file, out_file)
