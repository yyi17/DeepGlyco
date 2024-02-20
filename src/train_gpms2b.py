import random
import time
from typing import Any, Mapping

import torch

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection
from deepglyco.deeplib.gpep.ms2b.data import (
    GlycoPeptideBranchMS2DataConverter,
    GlycoPeptideBranchMS2Dataset,
)
from deepglyco.deeplib.gpep.ms2b.model import GlycoPeptideBranchMS2TreeLSTM
from deepglyco.deeplib.gpep.ms2b.training import GlycoPeptideBranchMS2Trainer
from deepglyco.deeplib.util.writer import TensorBoardSummaryWriter
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.util.collections.dict import chain_item_typed
from deepglyco.util.di import Context
from deepglyco.util.io.path import list_files
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
    ctx.register(
        "summary_writer",
        TensorBoardSummaryWriter,
        log_dir=kwargs.get("summary_dir", None),
    )
    return ctx


def load_gpep_ms2_dataset(ctx: Context, speclib_files: list[str]):
    datasets: list[GlycoPeptideBranchMS2Dataset] = []
    for speclib_file in speclib_files:
        speclib = GlycoPeptideSpectralLibraryHdf(
            file_name=speclib_file,
            is_read_only=True,
        )
        dataset = ctx.build(GlycoPeptideBranchMS2Dataset, speclib=speclib)
        dataset.load(cache=speclib_file.replace(".h5", ".cache.pt"))
        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(datasets)
    return dataset


def load_pretrained_model(trainer: GlycoPeptideBranchMS2Trainer, model_file: str):
    trainer.build_model()
    pretrained_model = torch.load(model_file)

    model_type = chain_item_typed(pretrained_model, str, "config", "model", "type")
    if model_type == trainer.get_config("model", "type", typed=str):
        trainer.model.load_state_dict(pretrained_model["state"])
        trainer.model.pep_input.requires_grad_(False)
        trainer.model.gly_input.requires_grad_(False)
        trainer.model.pep_lstm_1.requires_grad_(False)
        trainer.model.gly_lstm_1.requires_grad_(False)

    elif model_type == "PeptideRTBiLSTM" and isinstance(
        trainer.model, GlycoPeptideBranchMS2TreeLSTM
    ):
        load_pretrained_model_pepms2(
            trainer.model,
            pretrained_model["state"],
        )
        trainer.model.pep_input.requires_grad_(False)
        trainer.model.pep_lstm_1.requires_grad_(False)

    elif model_type == "GlycoPeptideMS2TreeLSTM" and isinstance(
        trainer.model, GlycoPeptideBranchMS2TreeLSTM
    ):
        _, unexpected = trainer.model.load_state_dict(
            pretrained_model["state"], strict=False
        )
        if any(unexpected):
            raise ValueError(f"unexpected model parameters: {unexpected}")
        trainer.model.pep_input.requires_grad_(False)
        trainer.model.gly_input.requires_grad_(False)
        trainer.model.pep_lstm_1.requires_grad_(False)
        trainer.model.gly_lstm_1.requires_grad_(False)
        trainer.model.pep_gly_transform.requires_grad_(False)
        trainer.model.gly_pep_transform.requires_grad_(False)
        trainer.model.pep_lstm_2.requires_grad_(False)
        trainer.model.gly_lstm_2.requires_grad_(False)
        trainer.model.pep_output.requires_grad_(False)
        trainer.model.gly_output.cleavage.requires_grad_(False)
        trainer.model.gly_output.fragment.requires_grad_(False)
        trainer.model.gly_output.output.requires_grad_(False)
        trainer.model.ratio_output.requires_grad_(False)


def load_pretrained_model_pepms2(
    model: GlycoPeptideBranchMS2TreeLSTM, state_dict: Mapping[str, Any]
):
    states = {}
    for k, v in state_dict.items():
        if k.startswith("pep_input."):
            states[k] = v
        elif k.startswith("lstm_1.") or k.startswith("lstm_2."):
            states["pep_" + k] = v
        elif k.startswith("output.output_main."):
            states["pep_output.output_main." + k[len("output.output_main.") :]] = v

    _, unexpected = model.load_state_dict(states, strict=False)
    if any(unexpected):
        raise ValueError(f"unexpected model parameters: {unexpected}")


import pandas as pd


def train_gpep_ms2model(speclib_files, num_epochs: int, **kwargs):
    ctx = register_dependencies(**kwargs)

    trainer = ctx.build(
        GlycoPeptideBranchMS2Trainer, configs=kwargs.get("config_file", None)
    )

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {trainer.get_configs()}")

    ctx.register(
        "converter",
        GlycoPeptideBranchMS2DataConverter,
        configs=trainer.get_config("data", typed=dict),
    )

    dataset = load_gpep_ms2_dataset(ctx, speclib_files)

    indices = trainer.load_data(dataset, val_ratio=0.2, holdout_ratio=0.2)
    pd.DataFrame.from_dict(
        {
            "index": [*indices[0], *indices[1]],
            "train": [True] * len(indices[0]) + [False] * len(indices[1]),
        }
    ).to_csv(rf"{kwargs.get('summary_dir', '')}\indices.csv", index=False)

    pretrained_model = kwargs.get("pretrained_model", None)
    if pretrained_model:
        load_pretrained_model(trainer, pretrained_model)
    else:
        trainer.build_model()

    if logger:
        logger.info(trainer.model)

    summary = trainer.train(
        num_epochs,
        patience=kwargs.get("patience", 0),
        checkpoint=kwargs.get("checkpoint", None),
        # callback=trainer_callback
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for glycopeptide MS2 with B ions."
    )
    parser.add_argument("--in", nargs="+", help="input training data files")
    parser.add_argument("--wkdir", help="working dir")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--pretrained", help="pretrained model file")
    parser.add_argument(
        "--epochs",
        type=int,
        default=470,
        help="number of epochs (default: %(default)s)",
    )

    args = parser.parse_args()

    train_dir = args.wkdir
    if train_dir is None:
        train_dir = (
            f"deepgpms2b_{time.strftime('%Y%m%d%H%M%S')}"
            + f"_{random.randint(0, 99999):05d}"
        )

    config_file = args.config
    pretrained_file = args.pretrained
    speclib_files = getattr(args, "in")
    assert speclib_files is not None and len(speclib_files) > 0
    num_epochs = args.epochs
    checkpoint = train_dir + r"\checkpoints\epoch_{epoch}.pt"

    summary = train_gpep_ms2model(
        speclib_files,
        num_epochs,
        checkpoint=checkpoint,
        pretrained_model=pretrained_file,
        summary_dir=train_dir,
    )

    print(summary)
