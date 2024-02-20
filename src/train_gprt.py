import random
import time
from typing import Any, Mapping

import torch

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.mods import ModificationCollection, ModifiedSequenceParser
from deepglyco.deeplib.gpep.rt.data import (
    GlycoPeptideRTDataConverter,
    GlycoPeptideRTDataset,
)
from deepglyco.deeplib.gpep.rt.model import GlycoPeptideRTTreeLSTM
from deepglyco.deeplib.gpep.rt.training import GlycoPeptideRTTrainer
from deepglyco.deeplib.util.writer import TensorBoardSummaryWriter
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.util.collections.dict import chain_item_typed
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


def load_gpep_rt_dataset(ctx: Context, rtlib_files: list[str]):
    datasets: list[GlycoPeptideRTDataset] = []

    for rtlib_file in rtlib_files:
        rtlib = GlycoPeptideSpectralLibraryHdf(
            file_name=rtlib_file,
            is_read_only=True,
        )

        dataset = ctx.build(GlycoPeptideRTDataset, rtlib=rtlib)
        dataset.load(cache=rtlib_file.replace(".h5", ".cache.pt"))
        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(datasets)
    return dataset


def load_pretrained_model(trainer: GlycoPeptideRTTrainer, model_file):
    trainer.build_model()
    pretrained_model = torch.load(model_file)

    model_type = chain_item_typed(pretrained_model, str, "config", "model", "type")
    if model_type == trainer.get_config("model", "type", typed=str):
        trainer.model.load_state_dict(pretrained_model["state"])
        trainer.model.pep_lstm_1.requires_grad_(False)
        trainer.model.gly_lstm_1.requires_grad_(False)

    elif model_type == "PeptideRTBiLSTM" and isinstance(trainer.model, GlycoPeptideRTTreeLSTM):
        load_pretrained_model_peprt(
            trainer.model,
            pretrained_model["state"],
        )
        trainer.model.pep_lstm_1.requires_grad_(False)


def load_pretrained_model_peprt(model: GlycoPeptideRTTreeLSTM, state_dict: Mapping[str, Any]):
    states = {}
    for k, v in state_dict.items():
        if k.startswith("pep_input."):
            states[k] = v
        elif k.startswith("lstm_1.") or k.startswith("lstm_2."):
            states["pep_" + k] = v
        elif k.startswith("output."):
            states["output.pep_" + k[len("output.") :]] = v

    _, unexpected = model.load_state_dict(states, strict=False)
    if any(unexpected):
        raise ValueError(f"unexpected model parameters: {unexpected}")


import pandas as pd


def train_gpep_rtmodel(rtlib_files, num_epochs, **kwargs):
    ctx = register_dependencies(**kwargs)

    trainer = ctx.build(GlycoPeptideRTTrainer, configs=kwargs.get("config_file", None))

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {trainer.get_configs()}")

    ctx.register(
        "converter",
        GlycoPeptideRTDataConverter,
        configs=trainer.get_config("data", typed=dict),
    )

    dataset = load_gpep_rt_dataset(ctx, rtlib_files)

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

    parser = argparse.ArgumentParser(description="Train a model for glycopeptide iRT.")
    parser.add_argument("--in", nargs="+", help="input training data files")
    parser.add_argument("--wkdir", help="working dir")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--pretrained", help="pretrained model file")
    parser.add_argument(
        "--epochs",
        type=int,
        default=320,
        help="number of epochs (default: %(default)s)",
    )

    args = parser.parse_args()

    train_dir = args.wkdir
    if train_dir is None:
        train_dir = (
            f"deepgprt_{time.strftime('%Y%m%d%H%M%S')}"
            + f"_{random.randint(0, 99999):05d}"
        )

    config_file = args.config
    pretrained_file = args.pretrained
    rtlib_files = getattr(args, "in")
    assert rtlib_files is not None and len(rtlib_files) > 0
    num_epochs = args.epochs
    checkpoint = train_dir + r"\checkpoints\epoch_{epoch}.pt"

    summary = train_gpep_rtmodel(
        rtlib_files,
        num_epochs,
        checkpoint=checkpoint,
        pretrained_model=pretrained_file,
        summary_dir=train_dir,
        config_file=config_file,
    )

    print(summary)
