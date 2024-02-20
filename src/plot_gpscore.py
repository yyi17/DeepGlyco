# %%
import functools
import itertools
import os
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


# %%
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


# %%
def get_intensity(
    scorer: GlycoPeptideStructBranchScorer,
    cache_speclib: GlycoPeptideSpectralLibraryHdf,
    run_name: str,
    scan_number: int,
):
    spectrum_info = cache_speclib.get_spectrum_info()
    assert spectrum_info is not None
    spec_id = (
        spectrum_info["run_name"].eq(run_name)
        & spectrum_info["scan_number"].eq(scan_number)
    ).argmax()
    spec_data = cache_speclib.get_fragments_by_spectrum_id([int(spec_id)])[0]
    assert spec_data is not None

    ref_spec_id = result.loc[
        result["run_name"].eq(run_name) & result["scan_number"].eq(scan_number),
        "ref_index",
    ].tolist()
    spectrum_fragments = scorer.speclib.get_fragments_by_spectrum_id(ref_spec_id)

    combined_charge = True
    budding_ratio = 0.00

    fragment_keys = spec_data.columns.difference(["mz", "intensity"]).to_list()
    if combined_charge:
        fragment_keys.remove("fragment_charge")

    spectrum_fragments.insert(0, spec_data)

    for spec_data in spectrum_fragments:
        spec_data.insert(
            0,
            "mass",
            spec_data["mz"] * spec_data["fragment_charge"]
            - spec_data["fragment_charge"],
        )
        spec_data.sort_values("mass", inplace=True)

    intensity_list = []
    for i, frag in enumerate(spectrum_fragments):
        intensity = frag.groupby(by=fragment_keys, sort=False)["intensity"].sum()

        if budding_ratio and i > 0:
            intensity = intensity.max() * budding_ratio + intensity * (
                1 - budding_ratio
            )

        intensity_list.append(intensity)

    intensity = pd.concat(intensity_list, axis=1)
    intensity.fillna(0.0, inplace=True)
    intensity = intensity.loc[intensity.iloc[:, 1:].sum(axis=1) > 0, :]
    intensity.columns = ["query"] + ref_spec_id

    return spectrum_fragments, intensity


# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_spectrum(spec, ax=None):
    if ax is None:
        ax = plt.axes()
        ax.set_xlabel(r"$m/z$")
        ax.set_ylabel(r"Intensity")
        ax.set_yscale("function", functions=(np.sqrt, lambda x: np.power(x, 2)))  # type: ignore
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.tick_params(axis="both", labelsize=8)

    ax.vlines(x=spec.mz, ymin=0, ymax=spec.intensity, colors=["grey"])

    return ax


def plot_annotated_spectrum(spec, ax=None):
    if ax is None:
        ax = plt.axes()
        ax.set_xlabel(r"$m/z$")
        ax.set_ylabel(r"Intensity")
        ax.set_yscale("function", functions=(np.sqrt, lambda x: np.power(x, 2)))  # type: ignore
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    pep_frag = spec.loc[spec.fragment_number >= 0]
    ax.vlines(x=pep_frag.mz, ymin=0, ymax=pep_frag.intensity, colors=["#339dff"])

    gly_frag = spec.loc[spec.fragment_type == "Y"]
    ax.vlines(x=gly_frag.mz, ymin=0, ymax=gly_frag.intensity, colors=["#c277ca"])

    glyb_frag = spec.loc[spec.fragment_type == "B"]
    ax.vlines(x=glyb_frag.mz, ymin=0, ymax=glyb_frag.intensity, colors=["#ff3355"])

    return ax


# %%
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


def plot_intensity(data):
    n = data.shape[1]

    data = data.loc[data.sum(axis=1) > 0]

    gly_frag = data.loc[data.index.get_level_values("fragment_type") == "Y"]
    gly_cmap = LinearSegmentedColormap.from_list(
        "gly", ["#d5a1db", "#c277ca", "#6f2e76"]
    )

    glyb_frag = data.loc[data.index.get_level_values("fragment_type") == "B"]
    glyb_cmap = LinearSegmentedColormap.from_list(
        "glyb", ["#ff5a76", "#ff3355", "#950019"]
    )

    fig, axes = plt.subplots(
        n,
        2,
        sharex="col",
        gridspec_kw={"width_ratios": [len(glyb_frag) + 1, len(gly_frag) + 1]},
    )
    for i in range(n):
        ax = axes[i, 0]
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.set_yticks([])
        ax.margins(x=0.5 / len(glyb_frag))
        ax.bar(
            x=glyb_frag.index.get_level_values("fragment_glycan"),
            height=glyb_frag.iloc[:, i] / glyb_frag.iloc[:, i].max(),
            color=glyb_cmap(
                np.sqrt(glyb_frag.iloc[:, i]) / np.sqrt(gly_frag.iloc[:, i].max())
            ),
        )

        ax = axes[i, 1]
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.set_yticks([])
        ax.margins(x=0.5 / len(gly_frag))
        ax.bar(
            x=gly_frag.index.get_level_values("fragment_glycan").map(
                lambda x: "Y-" + x if x != "" else "Y0"
            ),
            height=gly_frag.iloc[:, i] / gly_frag.iloc[:, i].max(),
            color=gly_cmap(
                np.sqrt(gly_frag.iloc[:, i]) / np.sqrt(gly_frag.iloc[:, i].max())
            ),
        )

    fig.subplots_adjust(hspace=0, wspace=0)

    return fig


# %%
report_type = "strucgp"
config_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "deepglyco",
    "gpscore",
    "gpscore_pglyco_branch.yaml",
)

glycan_file = "glycan.gdb"
pretrained_file = "gpms2b_model.pt"
pred_speclib_file = r"prediction.speclib.h5"
result_file = "result.csv"


# %%
ctx = register_dependencies(
    report_type=report_type,
    config_file=config_file,
    glycan_file=glycan_file,
)

scorer = ctx.build(GlycoPeptideStructBranchScorer)
logger = ctx.get("logger")

result = pd.read_csv(result_file)

speclib = GlycoPeptideSpectralLibraryHdf(
    file_name=pred_speclib_file,
    is_read_only=True,
)
scorer.load_speclib(speclib)


# %%
run_name = "Run"
scan_number = 1
scan_number_low = None
candidate_indexes = [1, 2, 3]

spectra_file = f"{run_name}.mzML"
cache_spectra_file = f"{run_name}.spec.h5"

cache_speclib = GlycoPeptideSpectralLibraryHdf(file_name=cache_spectra_file)

spectrum_fragments, intensity = get_intensity(
    scorer, cache_speclib, run_name, scan_number
)

fig = plot_intensity(
    intensity.iloc[:, [0] + candidate_indexes]
    if candidate_indexes is not None
    else intensity
)

fig.set_size_inches(16 / 2.54, 8 / 2.54)
fig.savefig(
    f"plots/gpscore/{run_name}.{scan_number}.intensity.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()
plt.close()

# %%
raw_spec_high = None
raw_spec_low = None

with MzmlReader(spectra_file) as spectra:
    for spec in spectra:
        if spec.scan_number == scan_number:
            raw_spec_high = pd.DataFrame.from_dict(
                {"mz": spec.mz, "intensity": spec.intensity / 2}
            )
        if scan_number_low and spec.scan_number == scan_number_low:
            raw_spec_low = pd.DataFrame.from_dict(
                {"mz": spec.mz, "intensity": spec.intensity / 2}
            )

assert raw_spec_high is not None

ax = plot_raw_spectrum(raw_spec_high)
if raw_spec_low is not None:
    ax = plot_raw_spectrum(raw_spec_low, ax=ax)

ax = plot_annotated_spectrum(spectrum_fragments[0], ax=ax)

ax.figure.set_size_inches(16 / 2.54, 5 / 2.54)
ax.figure.savefig(
    f"plots/gpscore/{run_name}.{scan_number}.spectrum.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()
plt.close()

# %%
