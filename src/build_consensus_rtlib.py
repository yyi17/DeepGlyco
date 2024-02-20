import os
from typing import Any, cast

import numpy as np
import pandas as pd


np.seterr(all="raise")

from deepglyco.speclib.common.rtcalibrate import RetentionTimeCalibrator
from deepglyco.speclib.common.rtcombine import NonRedundantRetentionTimeConsensus
from deepglyco.speclib.pep.hdf import PeptideSpectralLibraryHdf
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def calibrate_rtlib(source_file, reference_file, logger):
    rt_calibrator = RetentionTimeCalibrator(logger=logger)

    speclib = PeptideSpectralLibraryHdf(
        file_name=source_file,
        is_read_only=True,
    )

    if logger:
        logger.info(f"Loading {reference_file}")
    reference = pd.read_csv(reference_file)

    rt_calibrator.load_reference(reference)
    if logger:
        logger.info(f"Reference: {len(reference)} entries")
        logger.info(
            f"Source: {speclib.num_retention_time} entries of {speclib.num_peptides} peptides"
        )

    data = speclib.get_retention_time_data()
    assert data is not None
    rt, anchors, r2 = rt_calibrator.calibrate_rt(data)

    if logger:
        logger.info(f"Anchors: {len(anchors)} entries")
        logger.info(f"Calibrated: {len(rt)} entries")
    return rt, anchors, r2


import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def plot_rt_calibration(
    rt_data: pd.DataFrame, anchors: pd.DataFrame, r2: pd.Series, pdf_path: str
):
    COLUMN_RUN_NAME = "run_name"
    COLUMN_RT = "retention_time"
    COLUMN_CALIBRATED_RT = "calibrated_retention_time"
    COLUMN_RT_REF = f"{COLUMN_RT}_reference"
    max_data_points = 10000
    progress_factory = TqdmProgressFactory()

    with PdfPages(pdf_path) as pdf:
        groups = rt_data.groupby(COLUMN_RUN_NAME)

        if len(groups) > 1:
            for idx, (run, data) in progress_factory(
                enumerate(groups),
                total=len(groups),
                desc="Generating plots",
            ):
                run_anchors = anchors.loc[anchors[COLUMN_RUN_NAME] == run]
                if len(data) > max_data_points:
                    data = data.iloc[
                        np.random.randint(
                            low=0,
                            high=len(data),
                            size=max_data_points,
                        )
                    ]
                if len(run_anchors) > max_data_points:
                    run_anchors = run_anchors.iloc[
                        np.random.randint(
                            low=0,
                            high=len(run_anchors),
                            size=max_data_points,
                        )
                    ]

                if idx % 6 == 0:
                    plt.figure(figsize=(10, 15))
                    plt.subplots_adjust(hspace=0.75)

                plt.subplot(321 + idx % 6)

                plt.scatter(
                    x=data[COLUMN_RT],
                    y=data[COLUMN_CALIBRATED_RT],
                    s=1,
                    marker=cast(Any, "."),
                )
                plt.scatter(
                    x=run_anchors[COLUMN_RT],
                    y=run_anchors[COLUMN_RT_REF],
                    s=np.where(run_anchors["inlier"], 4, 2),
                    color=np.where(run_anchors["inlier"], "red", "gray"),
                    alpha=0.5,
                    marker=cast(Any, "D"),
                )
                plt.annotate(
                    text=f"R2={r2.loc[run]:.3f}",
                    xy=(data[COLUMN_RT].max(), data[COLUMN_CALIBRATED_RT].min()),
                    ha="right",
                    va="bottom",
                )
                plt.xlabel("Raw RT")
                plt.ylabel("Calibrated RT")
                plt.title(cast(Any, (run)))

                if idx % 6 == 5 or idx == len(groups) - 1:
                    pdf.savefig()
                    plt.close()

        else:
            if len(rt_data) > max_data_points:
                rt_data = rt_data.iloc[
                    np.random.randint(
                        low=0,
                        high=len(rt_data),
                        size=max_data_points,
                    )
                ]
            if len(anchors) > max_data_points:
                anchors = anchors.iloc[
                    np.random.randint(
                        low=0,
                        high=len(anchors),
                        size=max_data_points,
                    )
                ]

            plt.figure(figsize=(10, 10))

            plt.scatter(
                x=rt_data[COLUMN_RT],
                y=rt_data[COLUMN_CALIBRATED_RT],
                s=1,
                marker=cast(Any, "."),
            )
            plt.scatter(
                x=anchors[COLUMN_RT],
                y=anchors[COLUMN_RT_REF],
                s=np.where(anchors["inlier"], 4, 2),
                color=np.where(anchors["inlier"], "red", "gray"),
                marker=cast(Any, "D"),
            )
            plt.annotate(
                text=f"R2={r2.iloc[0]:.3f}",
                xy=(rt_data[COLUMN_RT].max(), rt_data[COLUMN_CALIBRATED_RT].min()),
                ha="right",
                va="bottom",
            )
            plt.xlabel("Raw RT")
            plt.ylabel("Calibrated RT")
            plt.title(rt_data["run"][0])

            pdf.savefig()
            plt.close()


def build_consesus_rtlib(source_file, reference_file, dest_file):
    log_file = rf"{os.path.splitext(dest_file)[0]}.log"
    logger = get_logger(file=log_file)

    rt, anchors, r2 = calibrate_rtlib(source_file, reference_file, logger=logger)

    pdf_file = (
        rf"{os.path.splitext(dest_file)[0].replace('.rtlib', '')}_rtcalibration.pdf"
    )
    plot_rt_calibration(rt, anchors, r2, pdf_file)
    rt.to_csv(rf"{os.path.splitext(pdf_file)[0]}.csv", index=False)
    logger.info(f"RT calibration plots generated: {pdf_file}")

    nr_rt = NonRedundantRetentionTimeConsensus()
    rt = rt.drop(columns=["retention_time"]).rename(
        columns={"calibrated_retention_time": "retention_time"}
    )
    rt = nr_rt.combine_retention_time(rt)

    nr_speclib = PeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=True,
    )

    nr_speclib.import_retention_time(rt)
    logger.info(
        f"Total: {nr_speclib.num_retention_time} entries of {nr_speclib.num_peptides} peptides"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build consensus peptide retention time library."
    )
    parser.add_argument(
        "--in", dest="source_file", required=True, help="input rtlib file"
    )
    parser.add_argument(
        "--reference", dest="reference_file", required=True, help="input reference file"
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output rtlib file"
    )

    args = parser.parse_args()

    build_consesus_rtlib(**vars(args))
