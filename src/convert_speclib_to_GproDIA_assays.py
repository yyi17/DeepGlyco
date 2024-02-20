from typing import Optional

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModificationCollection, ModifiedSequenceParser
from deepglyco.speclib.gpep.abs import GlycoPeptideSpectralLibraryBase
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.speclib.gpep.spec import GlycoPeptideMS2SpectrumProto
from deepglyco.util.di import Context
from deepglyco.util.io.pickle import save_pickle
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def register_dependencies(**kwargs):
    ctx = Context()
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

# %%
import pandas as pd

class GproDIAAssayConverter:
    def __init__(self, sequence_parser: ModifiedSequenceParser):
        self.sequence_parser = sequence_parser

    def to_assays(self, speclib: GlycoPeptideSpectralLibraryBase, glycopeptides: Optional[pd.DataFrame] = None):
        if glycopeptides is not None:
            glycopeptides = glycopeptides.set_index(
                ["modified_sequence", "glycan_struct", "glycan_position", "precursor_charge"]
            )

        if glycopeptides is None or "retention_time" not in glycopeptides.columns:
            retention_time = speclib.get_retention_time_data()
            if retention_time is not None:
                retention_time = retention_time.set_index(
                    ["modified_sequence", "glycan_struct", "glycan_position", "precursor_charge"]
                )
        else:
            retention_time = None

        for spec in speclib.iter_spectra():
            assay = self.to_assay(spec)

            if glycopeptides is not None:
                row = glycopeptides.loc[
                    (
                        spec.modified_sequence,
                        spec.glycan_struct,
                        spec.glycan_position,
                        spec.precursor_charge,
                    ),
                    :,
                ]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                if "protein" in row and "protein_site" in row:
                    assay["metadata"] = {
                        "protein": row["protein"],
                        "proteinSite": row["protein_site"],
                    }
                if "retention_time" in row:
                    assay["rt"] = float(row["retention_time"])
            if retention_time is not None:
                row = retention_time.loc[
                    (
                        spec.modified_sequence,
                        spec.glycan_struct,
                        spec.glycan_position,
                        spec.precursor_charge,
                    ),
                    :,
                ]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                assay["rt"] = float(row["retention_time"])

            yield assay

    def to_assay(self, spec: GlycoPeptideMS2SpectrumProto):
        parsed_sequence = self.sequence_parser.parse_modified_sequence(
            spec.modified_sequence
        )

        sequence = [aa for aa, mod in parsed_sequence]
        sequence[spec.glycan_position - 1] = "J"
        sequence = "".join(sequence)

        modification = []
        for i, (aa, mod) in enumerate(parsed_sequence):
            if mod == "":
                continue
            mod_info = self.sequence_parser.modifications[mod]
            if "[" in mod or "<" in mod:
                site = "N-term"
                position = None
            elif "]" in mod or ">" in mod:
                site = "C-term"
                position = None
            else:
                site = aa
                position = i + 1
            modification.append(
                {
                    "name": mod_info.name.split(" ")[0],
                    "site": site,
                    "position": position,
                }
            )

        assert spec.precursor_mz is not None

        assay = {
            "peptideSequence": sequence,
            "glycanStruct": spec.glycan_struct,
            "glycanSite": int(spec.glycan_position),
            "modification": modification or None,
            "precursorCharge": int(spec.precursor_charge),
            "precursorMZ": float(spec.precursor_mz),
        }

        fragments = {
            "fragmentMZ": spec.mz.tolist(),
            "fragmentIntensity": spec.intensity.tolist(),
            "fragmentType": [
                ft + ("-" + fg if fg == "N(1)" else fg)
                if fg and (ft == "b" or ft == "y")
                else ft
                for ft, fg in zip(
                    spec.fragment_type,
                    spec.fragment_glycan,
                )
            ],
            "fragmentNumber": spec.fragment_number.tolist(),
            "fragmentCharge": spec.fragment_charge.tolist(),
            "fragmentLossType": spec.loss_type.tolist(),
            "fragmentGlycan": [
                None
                if ft == "b" or ft == "y"
                else ft + fg
                if fg == "$"
                else ft + "-" + fg
                if fg
                else ft + "0"
                for ft, fg in zip(
                    spec.fragment_type,
                    spec.fragment_glycan,
                )
            ],
        }
        fragments["fragmentAnnotation"] = [
            (fg if fg else ft + str(fn)) + ("-" + fl if fl else "") + "^+" + str(fc)
            for ft, fn, fl, fg, fc in zip(
                fragments["fragmentType"],
                fragments["fragmentNumber"],
                fragments["fragmentLossType"],
                fragments["fragmentGlycan"],
                fragments["fragmentCharge"],
            )
        ]

        assay["fragments"] = fragments

        return assay


if __name__ == "__main__":
    # %%
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert spectral library to GproDIA assays.'
    )
    parser.add_argument(
        '--in', required=True,
        help='input speclib file'
    )
    parser.add_argument(
        '--glycopeptide', required=True,
        help='input glycopeptide file'
    )
    parser.add_argument(
        '--out',
        help='output assay file'
    )

    args = parser.parse_args()
    speclib_file = getattr(args, 'in')
    out_file = args.out
    glycopeptide_file = args.glycopeptide

    # %%
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(filename)s: [%(levelname)s] %(message)s"
    )

    # %%
    import os

    if globals().get("out_file", None) is None:
        out_file = os.path.splitext(speclib_file)[0]
        if out_file.endswith(".speclib"):
            out_file = out_file[: -len(".speclib")]
        out_file += ".assay.pickle"

    # %%
    ctx = register_dependencies()

    converter = ctx.build(GproDIAAssayConverter)

    # %%
    logging.info("loading glycopeptides: " + glycopeptide_file)

    glycopeptides = pd.read_csv(glycopeptide_file)


    # %%
    logging.info("loading speclib: " + speclib_file)

    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=speclib_file,
        is_read_only=True,
    )

    logging.info("converting speclib to assays")
    assays = converter.to_assays(speclib, glycopeptides)

    progress_factory = ctx.get("progress_factory")
    if progress_factory:
        assays = progress_factory(assays, total=speclib.num_spectra)

    assays = list(assays)

    logging.info("assays converted: {0} glycopeptide precursors".format(len(assays)))

    # %%
    logging.info("saving assays: {0}".format(out_file))

    save_pickle(assays, out_file)

    logging.info(
        "assays saved: {0}, {1} glycopeptide precursors".format(out_file, len(assays))
    )

