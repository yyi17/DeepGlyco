import argparse
import tqdm
import os
import pandas as pd

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.mods import ModificationCollection
from deepglyco.speclib.pep.hdf import PeptideSpectralLibraryHdf
from deepglyco.speclib.pep.parser.maxquant import MaxQuantReportParser
from deepglyco.util.io.zip import zip_content
from deepglyco.util.di import Context
from deepglyco.util.log import get_logger


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

    ctx.register("progress_factory", lambda: tqdm.tqdm)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )

    return ctx


def get_report_parser(ctx: Context, report_type: str, **kwargs):
    if report_type.lower() == "maxquant":
        parser = ctx.build(
            MaxQuantReportParser, configs=kwargs.get("config_file", None)
        )
    else:
        raise ValueError(f"unknown report type {report_type}")

    return parser


def import_pep_rtlib(dest_file, report_file, report_type="MaxQuant", **kwargs):
    ctx = register_dependencies(
        log_file=rf"{os.path.splitext(dest_file)[0]}.log",
        **kwargs,
    )

    parser = get_report_parser(ctx, report_type, **kwargs)

    logger = ctx.get("logger")
    if logger:
        logger.info(f"Use configs: {parser.get_configs()}")

    speclib = PeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=not os.path.exists(dest_file),
    )

    def import_rt(report: pd.DataFrame):
        evidences = parser.parse_evidence_report(report)

        rt = (
            evidences[["modified_sequence", "run_name", "retention_time", "score"]]
            .groupby(
                ["modified_sequence", "run_name"],
                group_keys=False,
                as_index=False,
                sort=False,
                dropna=False,
            )
            .apply(lambda x: x.iloc[x["score"].argmax()])
        )

        speclib.import_retention_time(rt)

    with zip_content(report_file) as zip:
        chunksize = kwargs.get("chunksize", None)
        if chunksize is not None:
            for i, report in enumerate(
                pd.read_csv(zip, sep="\t", low_memory=False, chunksize=chunksize)
            ):
                if logger:
                    logger.info(f"Loading chunk {i} of {report_file}")
                import_rt(report)
                if logger:
                    logger.info(
                        f"Total: {speclib.num_retention_time} entries of {speclib.num_peptides} peptides"
                    )

        else:
            if logger:
                logger.info(f"Loading {report_file}")
            report = pd.read_csv(zip, sep="\t", low_memory=False)

            import_rt(report)
            if logger:
                logger.info(
                    f"Total: {speclib.num_retention_time} entries of {speclib.num_peptides} peptides"
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Import peptide retention time library from MaxQuant report."
    )
    parser.add_argument(
        "--in",
        dest="report_file",
        required=True,
        help="input MaxQuant evidence.txt report",
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output rtlib file"
    )
    parser.add_argument("--config", help="config file")
    parser.add_argument("--chunksize", type=int)

    args = parser.parse_args()

    import_pep_rtlib(**vars(args))
