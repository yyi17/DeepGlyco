import os

import numpy as np
import pandas as pd

from deepglyco.specio.mzml import MzmlReader
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def stat_speclib(speclib_file, logger, progres_factory):
    speclib = GlycoPeptideSpectralLibraryHdf(file_name=speclib_file)
    speclib_peak_stats = pd.DataFrame.from_records([
        {
            "run_name": spec.run_name,
            "scan_number": spec.scan_number,
            "num_peaks": spec.num_peaks,
            "num_peaks_peptide": np.isin(spec.fragment_type, ['b', 'y']).sum(),
            "num_peaks_glycan_Y": np.sum(spec.fragment_type == 'Y'),
            "num_peaks_glycan_B": np.sum(spec.fragment_type == 'B'),
            "sum_intensity": spec.intensity.sum(),
            "sum_intensity_peptide": spec.intensity[np.isin(spec.fragment_type, ['b', 'y'])].sum(),
            "sum_intensity_glycan_Y": spec.intensity[spec.fragment_type == 'Y'].sum(),
            "sum_intensity_glycan_B": spec.intensity[spec.fragment_type == 'B'].sum(),
        }
        for spec in progres_factory(speclib.iter_spectra())
    ])
    return speclib_peak_stats


def stat_spectra(report_files, spectra_files, logger, progres_factory):
    report = []
    for report_file in report_files:
        if report_file.endswith(".xlsx"):
            report.append(pd.read_excel(report_file))
        else:
            report.append(pd.read_csv(report_file))

    report = pd.concat(report, copy=False)

    spectra_peak_stats = []
    for spectra_file in spectra_files:
        logger.info(spectra_file)
        with MzmlReader(spectra_file) as spectra:
            spectra_peak_stats.append(pd.DataFrame.from_records([
                {
                    "run_name": spec.run_name,
                    "scan_number": spec.scan_number,
                    "num_peaks": spec.num_peaks,
                    "sum_intensity": spec.intensity.sum(),
                }
                for spec in progres_factory(spectra)
                if spec.ms_level == 2
            ]))

    spectra_peak_stats = pd.concat(spectra_peak_stats, copy=False)

    if any(report["MS2Scan"] != report["LowEnergy_MS2Scan"]):
        spectra_peak_stats_high = pd.merge(
            report[["FileName", "MS2Scan"]],
            spectra_peak_stats,
            left_on=["FileName", "MS2Scan"],
            right_on=["run_name", "scan_number"],
            how="left",
            copy=False,
        )
        spectra_peak_stats_low = pd.merge(
            report[["FileName", "LowEnergy_MS2Scan"]],
            spectra_peak_stats,
            left_on=["FileName", "LowEnergy_MS2Scan"],
            right_on=["run_name", "scan_number"],
            how="left",
            copy=False,
        )
        assert all(spectra_peak_stats_high["FileName"] == spectra_peak_stats_low["FileName"])

        spectra_peak_stats = pd.concat(
            [
                spectra_peak_stats_high.loc[:, ["run_name", "scan_number"]],
                spectra_peak_stats_high.loc[:, ["num_peaks"]]
                + spectra_peak_stats_low.loc[:, ["num_peaks"]],
                (spectra_peak_stats_high.loc[:, ["sum_intensity"]]
                + spectra_peak_stats_low.loc[:, ["sum_intensity"]]) / 2,
            ],
            axis=1,
            copy=False,
        )
    else:
        spectra_peak_stats = pd.merge(
            report[["FileName", "MS2Scan"]],
            spectra_peak_stats,
            left_on=["FileName", "MS2Scan"],
            right_on=["run_name", "scan_number"],
            how="left",
            copy=False,
        )
        spectra_peak_stats.drop(columns=["FileName", "MS2Scan"], inplace=True)

    spectra_peak_stats = spectra_peak_stats.loc[~spectra_peak_stats["num_peaks"].isnull()]

    return spectra_peak_stats




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count peak coverage of library glycopeptide spectra."
    )
    parser.add_argument(
        "--speclib", dest="speclib_file", required=True, help="input speclib file"
    )
    parser.add_argument(
        "--report", dest="report_files", nargs="+", required=True, help="input StrucGP .xlsx report"
    )
    parser.add_argument(
        "--spectra",
        nargs="+",
        dest="spectra_files",
        required=True,
        help="input .mzML spectra file",
    )
    parser.add_argument(
        "--out", dest="dest_file", help="output csv file"
    )
    args = parser.parse_args()

    speclib_file = args.speclib_file
    report_files = args.report_files
    spectra_files = args.spectra_files
    out_file = args.dest_file
    # speclib_file = r"..\data\gpms2\PXD025859_MouseBrain_StrucGP.speclib.h5"
    # report_file = r"D:\PXD025859\MouseBrain_result.xlsx"
    # spectra_files = [r"D:\PXD025859\mzML\ShenJ_MouseBrain_C18_HILIC_IGP_CE20_33_Run1.mzML.gz"]

    if out_file is None:
        out_file = os.path.splitext(speclib_file)[0] + ".peakcoverage.csv"

    logger = get_logger()
    progres_factory = TqdmProgressFactory()

    speclib_peak_stats = stat_speclib(speclib_file, logger=logger, progres_factory=progres_factory)
    spectra_peak_stats = stat_spectra(report_files, spectra_files, logger=logger, progres_factory=progres_factory)


    peak_stats = pd.merge(
        speclib_peak_stats,
        spectra_peak_stats,
        on=["run_name", "scan_number"],
        suffixes=["", "_full"],
        how="inner",
        copy=False,
    )

    peak_stats["coverage_peaks"] = peak_stats["num_peaks"] / peak_stats["num_peaks_full"]
    peak_stats["coverage_peaks_peptide"] = peak_stats["num_peaks_peptide"] / peak_stats["num_peaks_full"]
    peak_stats["coverage_peaks_glycan_Y"] = peak_stats["num_peaks_glycan_Y"] / peak_stats["num_peaks_full"]
    peak_stats["coverage_peaks_glycan_B"] = peak_stats["num_peaks_glycan_B"] / peak_stats["num_peaks_full"]
    peak_stats["fraction_intensity"] = peak_stats["sum_intensity"] / peak_stats["sum_intensity_full"]
    peak_stats["fraction_intensity_peptide"] = peak_stats["sum_intensity_peptide"] / peak_stats["sum_intensity_full"]
    peak_stats["fraction_intensity_glycan_Y"] = peak_stats["sum_intensity_glycan_Y"] / peak_stats["sum_intensity_full"]
    peak_stats["fraction_intensity_glycan_B"] = peak_stats["sum_intensity_glycan_B"] / peak_stats["sum_intensity_full"]

    peak_stats.to_csv(out_file, index=False)
