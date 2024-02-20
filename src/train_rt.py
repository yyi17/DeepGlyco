import random
import time

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.mods import ModificationCollection, ModifiedSequenceParser
from deepglyco.deeplib.pep.rt.data import PeptideRTDataConverter, PeptideRTDataset
from deepglyco.deeplib.pep.rt.training import PeptideRTTrainer
from deepglyco.deeplib.util.writer import TensorBoardSummaryWriter
from deepglyco.speclib.pep.hdf import PeptideSpectralLibraryHdf
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


def load_pep_rt_dataset(ctx: Context, rtlib_files: list[str]):
    datasets: list[PeptideRTDataset] = []
    for rtlib_file in rtlib_files:
        rtlib = PeptideSpectralLibraryHdf(
            file_name=rtlib_file,
            is_read_only=True,
        )
        dataset = ctx.build(PeptideRTDataset, rtlib=rtlib)
        dataset.load(cache=rtlib_file.replace(".h5", ".cache.pt"))
        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(datasets)
    return dataset


import pandas as pd
from torch.utils.data import Subset


def train_pep_rtmodel(rtlib_files, num_epochs: int, **kwargs):
    ctx = register_dependencies(**kwargs)

    trainer: PeptideRTTrainer = ctx.build(
        PeptideRTTrainer, configs=kwargs.get("config_file", None)
    )

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {trainer.get_configs()}")

    ctx.register(
        "converter",
        PeptideRTDataConverter,
        configs=trainer.get_config("data", typed=dict),
    )

    dataset = load_pep_rt_dataset(ctx, rtlib_files)

    indices = trainer.load_data(dataset)
    pd.DataFrame.from_dict(
        {
            "index": [*indices[0], *indices[1]],
            "train": [True] * len(indices[0]) + [False] * len(indices[1]),
        }
    ).to_csv(rf"{kwargs.get('summary_dir', '')}\indices.csv", index=False)

    pretrained_model = kwargs.get("pretrained_model", None)
    if pretrained_model:
        trainer.load_model(pretrained_model)
    else:
        trainer.build_model()

    if logger:
        logger.info(trainer.model)

    summary = trainer.train(
        num_epochs,
        patience=kwargs.get("patience", 0),
        checkpoint=kwargs.get("checkpoint", None),
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for peptide iRT.")
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
            f"deeprt_{time.strftime('%Y%m%d%H%M%S')}"
            + f"_{random.randint(0, 99999):05d}"
        )

    config_file = args.config
    pretrained_file = args.pretrained
    rtlib_files = getattr(args, "in")
    assert rtlib_files is not None and len(rtlib_files) > 0
    num_epochs = args.epochs
    checkpoint = train_dir + r"\checkpoints\epoch_{epoch}.pt"

    summary = train_pep_rtmodel(
        rtlib_files,
        num_epochs,
        checkpoint=checkpoint,
        config_file=config_file,
        pretrained_model=pretrained_file,
        summary_dir=train_dir,
    )

    print(summary)
