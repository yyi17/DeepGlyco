import os

import numpy as np
import pandas as pd

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.mods import ModificationCollection, ModifiedSequenceParser
from deepglyco.deeplib.gpep.rt.data import GlycoPeptideRTOutputConverter
from deepglyco.deeplib.gpep.rt.prediction import GlycoPeptideRTPredictor
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
    ctx.register("sequence_parser", ModifiedSequenceParser)

    ctx.register("converter", GlycoPeptideRTOutputConverter)

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
    elif glycopeptide_file.endswith("lib.h5"):
        rtlib = GlycoPeptideSpectralLibraryHdf(
            file_name=glycopeptide_file,
            is_read_only=True,
        )
        glycopeptides = rtlib.get_retention_time_data()
        if glycopeptides is not None:
            glycopeptides.insert(0, "index", glycopeptides.index)
        else:
            glycopeptides = rtlib.get_glycopeptide_data()
        assert glycopeptides is not None
    else:
        raise NotImplementedError
    return glycopeptides


def predict_gpep_rt(pretrained_file, glycopeptide_file, out_file, ctx=None, **kwargs):
    if ctx is None:
        ctx = register_dependencies(**kwargs)

    predictor = ctx.build(
        GlycoPeptideRTPredictor, configs=kwargs.get("config_file", None)
    )
    predictor.load_model(pretrained_file)

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {predictor.get_configs()}")

    glycopeptides = load_glycopeptides(glycopeptide_file)

    if logger:
        logger.info(f"Source: {len(glycopeptides)} glycopeptide entries")

    result = predictor.predict(glycopeptides)

    if out_file.endswith("lib.h5"):
        rtlib = GlycoPeptideSpectralLibraryHdf(
            file_name=out_file,
            is_read_only=False,
            is_new_file=not os.path.exists(out_file),
        )
        rtlib.import_retention_time(
            result[[c for c in result.columns if not c.endswith("_id")]]
        )
        if logger:
            logger.info(
                f"Total: {rtlib.num_retention_time} entries of "
                f"{rtlib.num_glycopeptides} glycopeptides"
            )
    else:
        if "retention_time" in glycopeptides.columns:
            result["original_retention_time"] = glycopeptides["retention_time"]

        if out_file.endswith(".csv"):
            result.to_csv(out_file, index=False)
        elif out_file.endswith(".tsv"):
            result.to_csv(out_file, sep="\t", index=False)
        else:
            raise NotImplementedError

    return result


def score_predicted_rt(result: pd.DataFrame):
    result = result.rename(columns={"original_retention_time": "target_retention_time"})
    result["prediction_error"] = (
        result["retention_time"] - result["target_retention_time"]
    )

    metrics = {
        "pcc": np.corrcoef(
            result["retention_time"],
            result["target_retention_time"],
        )[0, 1],
        "error": result["prediction_error"].quantile([0.025, 0.25, 0.5, 0.75, 0.975]),
    }
    return result, metrics


def validate_gpep_rtmodel(
    pretrained_file: str, rtlib_file: str, out_file: str, **kwargs
):
    ctx = register_dependencies(**kwargs)

    result = predict_gpep_rt(pretrained_file, rtlib_file, out_file, ctx=ctx, **kwargs)

    result, metrics = score_predicted_rt(result)

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Total: {len(result)} RT comparisons")
        logger.info(f"metrics: " + " ".join([f"{k}={v}" for k, v in metrics.items()]))

    result.to_csv(out_file, index=False)

    return result, metrics


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_rt_corr(result: pd.DataFrame, metrics: dict, ax=None):
    data = result[["target_retention_time", "retention_time"]].values
    density = gaussian_kde(data.T)(data.T)
    idx = density.argsort()
    data, density = data[idx, :], density[idx]

    if ax is None:
        ax = plt.axes()

    ax.scatter(
        x=data[:, 0],
        y=data[:, 1],
        c=density,
        alpha=0.25,
    )
    ax.set_xlabel("Target RT")
    ax.set_ylabel("Predicted RT")

    ax.annotate(
        text=f"r={metrics['pcc']:.3f}",
        xy=(
            ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1,
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
        ),
    )

    return ax


def plot_rt_error(result: pd.DataFrame, metrics: dict, ax=None):
    data = result[["prediction_error"]].values

    if ax is None:
        ax = plt.axes()

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
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Prediction Error")

    ax.annotate(
        text=f"IQR={metrics['error'].loc[0.75] - metrics['error'].loc[0.25]:.1f}",
        xy=(1, 0),
        xytext=(0.75, metrics["error"].loc[0.5]),
        rotation=90,
    )
    ax.annotate(
        text=f"R95%={metrics['error'].loc[0.975] - metrics['error'].loc[0.025]:.1f}",
        xy=(1, metrics["error"].loc[0.75]),
        xytext=(1.25, 0),
        rotation=90,
    )

    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict and validate glycopeptide retention time."
    )
    validate_group = parser.add_mutually_exclusive_group()
    validate_group.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="validate predicted results",
    )
    validate_group.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--in", required=True, help="input glycopeptide file")
    parser.add_argument("--model", required=True, help="pretrained model file")
    parser.add_argument("--out", help="output spectral library file")

    args = parser.parse_args()
    glycopeptide_file = getattr(args, "in")
    out_file = args.out
    pretrained_file = args.model
    validate = args.validate

    if not validate:
        predict_gpep_rt(
            pretrained_file,
            glycopeptide_file,
            out_file,
        )
    else:
        result, metrics = validate_gpep_rtmodel(
            pretrained_file,
            glycopeptide_file,
            out_file,
        )

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 5, (1, 4))
        plot_rt_corr(result, metrics, ax=ax)
        ax = plt.subplot(1, 5, 5)
        plot_rt_error(result, metrics, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.splitext(out_file)[0] + ".svg")
