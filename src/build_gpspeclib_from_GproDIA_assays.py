import numpy as np

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import (
    ModPosition,
    ModificationCollection,
    ModifiedSequenceParser,
    modified_sequence_to_str,
)
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.speclib.gpep.spec import GlycoPeptideMS2Spectrum
from deepglyco.util.di import Context
from deepglyco.util.io.pickle import load_pickle
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


class GproDIAAssayToSpecConverter:
    def __init__(self, sequence_parser: ModifiedSequenceParser):
        self.sequence_parser = sequence_parser

    def assay_to_spec(self, assay: dict):
        sequence = assay["peptideSequence"].replace("J", "N")
        mod_list = [""] * len(sequence)
        modification = assay.get("modification", None)
        if modification is not None:
            for mod in modification:
                pos = mod.get("position", None)
                site = mod.get("site", None)
                if isinstance(pos, int):
                    aa = sequence[pos - 1]
                else:
                    aa = ""
                if site == "N-term":
                    position = ModPosition.any_n_term
                    pos = 1
                elif site == "C-term":
                    position = ModPosition.any_c_term
                    pos = len(sequence)
                else:
                    position = ModPosition.anywhere
                if not isinstance(pos, int):
                    raise ValueError(
                        f"modification position not determined: {mod['name']}"
                    )
                mod_sym = self.sequence_parser.modifications.search_by_name(
                    name=mod["name"], amino_acid=aa, position=position
                )
                if len(mod_sym) == 0:
                    raise ValueError(f"unknown modification: {mod['name']} at {site}")
                mod_list[pos - 1] = mod_sym[0]

        modified_sequence = modified_sequence_to_str(list(zip(sequence, mod_list)))

        fragments = assay["fragments"]
        fragment_type = np.array(
            [
                ft[:-5]
                if ft.endswith("-N(1)")
                else (ft[:-1] if ft.endswith("$") else ft)
                for ft in fragments["fragmentType"]
            ],
            dtype=np.unicode_,
        )
        fragment_number = np.array(
            [-1 if fn is None else int(fn) for fn in fragments["fragmentNumber"]],
            dtype=np.int_,
        )
        loss_type = np.array(
            [
                ""
                if lt is None or lt == "none" or lt == "None" or lt == "noloss"
                else lt
                for lt in fragments["fragmentLossType"]
            ],
            dtype=np.unicode_,
        )
        fragment_glycan = np.array(
            [
                ("N(1)" if ft.endswith("-N(1)") else ("$" if ft.endswith("$") else ""))
                if fg is None or fg == "" or fg == "none" or fg == "None" or fg == "Y0"
                else (fg[2:] if fg.startswith("Y-") else ("$" if fg == "Y$" else fg))
                for ft, fg in zip(
                    fragments["fragmentType"], fragments["fragmentGlycan"]
                )
            ],
            dtype=np.unicode_,
        )

        return GlycoPeptideMS2Spectrum(
            modified_sequence=modified_sequence,
            glycan_struct=assay["glycanStruct"],
            glycan_position=int(assay["glycanSite"]),
            precursor_charge=int(assay["precursorCharge"]),
            mz=np.array(fragments["fragmentMZ"], dtype=np.float_),
            intensity=np.array(fragments["fragmentIntensity"], dtype=np.float_),
            fragment_charge=np.array(fragments["fragmentCharge"], dtype=np.int_),
            fragment_type=fragment_type,
            fragment_number=fragment_number,
            loss_type=loss_type,
            fragment_glycan=fragment_glycan,
            precursor_mz=float(assay["precursorMZ"]),
        )


if __name__ == "__main__":
    # %%
    import argparse

    parser = argparse.ArgumentParser(
        description="Build spectral library from GproDIA assays."
    )
    parser.add_argument("--in", required=True, help="input assay file")
    parser.add_argument("--out", help="output spectral library file")
    
    args = parser.parse_args()
    assay_file = getattr(args, "in")
    out_file = args.out

    # %%
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s: [%(levelname)s] %(message)s",
    )

    # %%
    import os

    if globals().get("out_file", None) is None:
        out_file = os.path.splitext(assay_file)[0]
        if out_file.endswith(".assay"):
            out_file = out_file[: -len(".assay")]
        out_file += ".speclib.h5"

    # %%
    ctx = register_dependencies()

    converter = ctx.build(GproDIAAssayToSpecConverter)

    # %%
    logging.info("loading assays: " + assay_file)

    assays = load_pickle(assay_file)

    logging.info("assays loaded: {0} glycopeptide precursors".format(len(assays)))

    # %%
    speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=out_file,
        is_new_file=True,
        is_read_only=False,
    )

    logging.info(f"import assays to speclib {out_file}")
    spectra = (converter.assay_to_spec(assay) for assay in assays)

    progress_factory = ctx.get("progress_factory")
    if progress_factory:
        spectra = progress_factory(spectra, total=len(assays))

    speclib.import_spectra(spectra)

    logging.info(
        f"Total: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
        f"{speclib.num_glycopeptides} glycopeptides"
    )
