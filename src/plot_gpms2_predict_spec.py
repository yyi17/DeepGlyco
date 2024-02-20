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

from deepglyco.deeplib.gpep.ms2.data import GlycoPeptideMS2OutputConverter
from deepglyco.deeplib.gpep.ms2.prediction import GlycoPeptideMS2Predictor

# from deepglyco.deeplib.gpep.ms2b.data import (
#     GlycoPeptideBranchMS2OutputConverter as GlycoPeptideMS2OutputConverter,
# )
# from deepglyco.deeplib.gpep.ms2b.prediction import (
#     GlycoPeptideBranchMS2Predictor as GlycoPeptideMS2Predictor,
# )

from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
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


# %%
import numpy as np
import matplotlib.pyplot as plt

from adjustText import adjust_text


def plot_annotated_spectrum_mirror(spec1, spec2, ax=None):
    if ax is None:
        ax = plt.axes()
        ax.set_xlabel(r"$m/z$")
        ax.set_ylabel(r"Relative Intensity (%)")

    mirror = False
    for spec in [spec1, spec2]:
        intensity = spec.intensity / spec.intensity.max() * 100
        if mirror:
            intensity = -intensity
        else:
            mirror = True

        glyb_frag = spec.fragment_type == "B"
        ax.vlines(
            x=spec.mz[glyb_frag], ymin=0, ymax=intensity[glyb_frag], colors=["#ff3355"]
        )

        gly_frag = spec.fragment_type == "Y"
        ax.vlines(
            x=spec.mz[gly_frag], ymin=0, ymax=intensity[gly_frag], colors=["#c277ca"]
        )

        pep_frag = spec.fragment_number > 0
        ax.vlines(
            x=spec.mz[pep_frag], ymin=0, ymax=intensity[pep_frag], colors=["#339dff"]
        )

    df = pd.merge(
        spec1,
        spec2,
        on=[
            "fragment_charge",
            "fragment_glycan",
            "fragment_number",
            "fragment_type",
            "loss_type",
        ],
        how="outer",
        suffixes=("_1", "_2"),
    )
    df["intensity_1"] = df["intensity_1"].fillna(0)
    df["intensity_2"] = df["intensity_2"].fillna(0)
    df["intensity_1"] /= spec1.intensity.max() / 100
    df["intensity_2"] /= spec2.intensity.max() / 100

    text_list_1 = []
    text_list_2 = []
    for _, r in df.iterrows():
        if r["intensity_1"] < 1 and r["intensity_2"] < 10:
            continue

        if r["fragment_type"] == "Y":
            label = r["fragment_type"]
            if r["fragment_glycan"] == "":
                label += "0"
            else:
                label += "-" + r["fragment_glycan"]
        elif r["fragment_type"] == "B":
            label = r["fragment_glycan"]
        elif r["fragment_number"] > 0:
            label = f"{r['fragment_type']}{r['fragment_number']}"
            if r["fragment_glycan"] == "$":
                label += r["fragment_glycan"]
            elif r["fragment_glycan"] != "":
                label += "-" + r["fragment_glycan"]
        else:
            continue
        if r["fragment_charge"] > 1:
            label += f" {r['fragment_charge']}+"

        if r["intensity_1"] > r["intensity_2"]:
            if r["mz_1"] > 2500:
                continue
            text = ax.text(
                x=r["mz_1"],
                y=r["intensity_1"],
                s=label,
                rotation=90,
                va="bottom" if r["intensity_1"] < 25 else "top",
                ha="center",
                fontsize=5.5,
            )
            text_list_1.append(text)
        else:
            if r["mz_2"] > 2500:
                continue
            text = ax.text(
                x=r["mz_2"],
                y=-r["intensity_2"],
                s=label,
                rotation=90,
                va="bottom" if r["intensity_2"] > 25 else "top",
                ha="center",
                fontsize=5.5,
            )
            text_list_2.append(text)

    hline = ax.axhline(y=0, c="k", linewidth=0.75)

    adjust_text(
        text_list_1,
        ax=ax,
        add_objects=[hline],
        autoalign=False, # type: ignore
        only_move={"points": "y", "text": "y", "objects": "y"},
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "grey"},
        avoid_points=False,
        avoid_self=False,
        va="bottom",
    )
    adjust_text(
        text_list_2,
        ax=ax,
        add_objects=[hline],
        autoalign=False, # type: ignore
        only_move={"points": "y", "text": "y", "objects": "y"},
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "grey"},
        avoid_points=False,
        avoid_self=False,
        va="top",
    )

    ax.xaxis.label.set_fontsize(8.5)
    ax.tick_params(labelsize=7)
    ax.yaxis.label.set_fontsize(8.5)
    ax.set_yscale(
        "function", # type: ignore
        functions=(
            lambda x: np.sqrt(np.abs(x)) * np.sign(x),
            lambda x: np.power(x, 2) * np.sign(x),
        ),
    )
    ax.yaxis.set_major_formatter(lambda x, pos: f"{abs(x):.0f}")

    return ax


# %%
def load_speclib(speclib: GlycoPeptideSpectralLibraryHdf):
    spectrum_info = speclib.get_spectrum_data(columns=["precursor_id"])
    assert spectrum_info is not None

    spectrum_info.reset_index(drop=False, inplace=True)
    spectrum_info.set_index(
        [
            "modified_sequence",
            "glycan_position",
            "glycan_struct",
            "precursor_charge",
        ],
        inplace=True,
    )
    spectrum_info.sort_index(inplace=True)
    return spectrum_info


# %%
ctx = register_dependencies()

pretrained_file = r"gpms2b_model.pt"
speclib_file = r"gpms2b_consensus.speclib.h5"

sequence = "VPGNVTAVLGETLK"
glycan_struct = "(N(N(H(H(N(H(A))))(H(N(H(A)))))))"
glycan_position = 4
precursor_charge = 3


predictor = ctx.build(GlycoPeptideMS2Predictor)
predictor.load_model(pretrained_file)

input = pd.DataFrame(
    dict(
        modified_sequence=sequence,
        glycan_position=glycan_position,
        glycan_struct=glycan_struct,
        precursor_charge=precursor_charge,
    ),
    index=[0],
)
pred = list(predictor.predict(input))[0]
pred = pd.DataFrame.from_dict(
    dict(
        mz=pred.mz,
        intensity=pred.intensity,
        **dict(pred.frangment_annotations()),
    )
)

speclib = GlycoPeptideSpectralLibraryHdf(
    file_name=speclib_file,
    is_read_only=True,
)
spectrum_info = load_speclib(speclib)
spec = speclib.get_fragments_by_spectrum_id(
    [cast(int, spectrum_info.loc[tuple(input.iloc[0]), "index"])]
)[0]


ax = plot_annotated_spectrum_mirror(pred, spec)
# ax.yaxis.set_ticks_position('right')
# ax.yaxis.set_label_position('right')
ax.set_xlim(100, 2550)
ax.figure.set_size_inches(11.8 / 2.54, 8 / 2.54)
ax.figure.savefig(
    f"plots/{sequence}_{glycan_position}_{glycan_struct}_{precursor_charge}.spec.svg",
    transparent=True,
    bbox_inches="tight",
)

plt.show(ax.figure)
plt.close(ax.figure)

# %%
