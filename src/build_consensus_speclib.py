import os

import numpy as np

np.seterr(all="raise")

from deepglyco.speclib.common.combine import NonRedundantSpectraConsensus
from deepglyco.speclib.pep.filter import PeptideSpectrumFilter
from deepglyco.speclib.pep.hdf import PeptideSpectralLibraryHdf
from deepglyco.speclib.pep.spec import PeptideMS2Spectrum
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def build_consesus_speclib(source_file, dest_file, **filter_args):
    log_file = rf"{os.path.splitext(dest_file)[0]}.log"
    logger = get_logger(file=log_file)
    progress_factory = TqdmProgressFactory()

    nr_spec = NonRedundantSpectraConsensus()
    filter = PeptideSpectrumFilter[PeptideMS2Spectrum]()

    if logger:
        logger.info(f"Use configs: {nr_spec.get_configs()}")
        logger.info(f"Use configs: {filter_args}")

    speclib = PeptideSpectralLibraryHdf(
        file_name=source_file,
        is_read_only=True,
    )
    nr_speclib = PeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=True,
    )

    if logger:
        logger.info(
            f"Source: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
            f"{speclib.num_peptides} peptides"
        )

    nr_speclib.import_spectra(
        progress_factory(
            filter.filter_spectra(nr_spec.nonredundant_spectra(speclib), **filter_args)
        )
    )

    if logger:
        logger.info(
            f"Total: {nr_speclib.num_spectra} spectra of {nr_speclib.num_precursors} precursors, "
            f"{nr_speclib.num_peptides} peptides"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build consensus peptide spectral library."
    )
    parser.add_argument(
        "--in", dest="source_file", required=True, help="input speclib file"
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output speclib file"
    )
    parser.add_argument("--min_num_fragments", type=eval, default=None)

    args = parser.parse_args()

    build_consesus_speclib(**vars(args))
