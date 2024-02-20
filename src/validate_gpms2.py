import os

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModificationCollection, ModifiedSequenceParser
from deepglyco.deeplib.gpep.ms2.data import (
    GlycoPeptideMS2DataConverter,
    GlycoPeptideMS2Dataset,
)
from deepglyco.deeplib.gpep.ms2.training import GlycoPeptideMS2Trainer
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

    ctx.register("progress_factory", TqdmProgressFactory)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )
    return ctx


def validate_gpep_ms2model(
    pretrained_file: str, speclib_file: str, out_file: str, **kwargs
):
    ctx = register_dependencies(**kwargs)

    trainer = ctx.build(GlycoPeptideMS2Trainer, configs=kwargs.get("config_file", None))
    trainer.load_model(pretrained_file)

    if isinstance(trainer.metrics_fn, dict):
        def ratio_error(batch, pred):
            return pred.fragment_intensity_ratio - batch.fragment_intensity_ratio

        trainer.metrics_fn.update({
            "ratio_error": ratio_error
        }) # type: ignore

        def pred_ratio(batch, pred):
            return pred.fragment_intensity_ratio
        
        trainer.metrics_fn.update({
            "ratio_pred": pred_ratio
        }) # type: ignore

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {trainer.get_configs()}")

    ctx.register(
        "converter",
        GlycoPeptideMS2DataConverter,
        configs=trainer.get_config("data", typed=dict),
    )

    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=speclib_file,
        is_read_only=True,
    )

    if logger:
        logger.info(
            f"Source: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
            f"{speclib.num_glycopeptides} glycopeptides"
        )

    spectrum_info = speclib.get_spectrum_data(columns=["precursor_id"])
    assert spectrum_info is not None

    dataset = ctx.build(GlycoPeptideMS2Dataset, speclib=speclib)
    dataset.load(cache=speclib_file.replace(".h5", ".cache.pt"))

    trainer.load_data(dataset, val_ratio=1.0, seed=0)

    result = trainer.validate()
    metrics = pd.DataFrame.from_dict(
        {k: v.numpy() for k, v in result["metrics_all"].items()}
    )
    if isinstance(trainer.val_loader.dataset, Subset):
        metrics.insert(0, "index", trainer.val_loader.dataset.indices)
        metrics = metrics.merge(
            spectrum_info,
            right_index=True,
            left_on="index",
            copy=False,
        )

    metrics.to_csv(out_file, index=False)

    if logger:
        logger.info(
            f"Total: {len(metrics)} spectral comparisons of "
            f"{len(metrics['precursor_id'].unique())} precursors, "
            f"{len(metrics['glycopeptide_id'].unique())} glycopeptides"
        )
        logger.info(
            f"loss: {result['loss']}, metrics: "
            + " ".join([f"{k}={v}" for k, v in result["metrics"].items()])
        )

    return metrics


import matplotlib.pyplot as plt


def plot_sa_loss(metrics: pd.DataFrame, ax=None):
    data = metrics[[c for c in metrics.columns if c.startswith("sa_")]]

    if ax is None:
        ax = plt.axes()

    ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    ax.boxplot(
        data,
        widths=0.05,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        boxprops={"facecolor": "w"},
        medianprops={"color": "k", "lw": 2},
    )

    ax.set_xticks(
        np.arange(1, 4),
        labels=data.columns.str.removeprefix("sa_").str.capitalize().tolist(),
    )
    ax.set_xlim(0.25, 3 + 0.75)
    ax.set_ylabel("Spectral Angle Loss")
    ax.set_ylim(1.05, -0.05)

    def sa_to_dp(x):
        return np.where(
            x >= 0,
            np.cos(x * np.pi / 2),
            2 - np.cos(x * np.pi / 2),
        )

    def dp_to_sa(x):
        x1 = np.where(x > 1, 2 - x, x)
        y = np.arccos(x1) * 2 / np.pi
        return np.where(x > 1, -y, y)

    secy = ax.secondary_yaxis("right", functions=(sa_to_dp, dp_to_sa))
    secy.set_ylabel("Dot Product")
    secy.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])

    for i, m in enumerate(data.median()):
        ax.annotate(
            text=f"SA={m:.3f}", xy=(i + 1, m), xytext=(i + 0.75, m), rotation=90
        )
        ax.annotate(
            text=f"DP={sa_to_dp(m):.3f}",
            xy=(i + 1, m),
            xytext=(i + 1.25, m),
            rotation=90,
        )

    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a model for glycopeptide MS2.")
    parser.add_argument("--model", required=True, help="pretrained model file")
    parser.add_argument("--in", required=True, help="input data files")
    parser.add_argument("--out", help="output predicted spectral similarities")
    parser.add_argument("--config", help="config file")

    args = parser.parse_args()

    pretrained_file = args.model
    speclib_file = getattr(args, "in")
    out_file = args.out
    if out_file is None:
        out_file = os.path.splitext(speclib_file)[0]
        if out_file.endswith(".speclib"):
            out_file = os.path.splitext(out_file)[0]
        out_file += "_prediction.ms2score.csv"

    metrics = validate_gpep_ms2model(
        pretrained_file,
        speclib_file,
        out_file,
    )

    fig = plot_sa_loss(metrics).get_figure()
    fig.savefig(os.path.splitext(out_file)[0] + ".svg")
