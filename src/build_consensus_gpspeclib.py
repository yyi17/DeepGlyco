import os
import numpy as np


np.seterr(all="raise")


from deepglyco.speclib.common.combine import NonRedundantSpectraConsensus
from deepglyco.speclib.gpep.filter import GlycoPeptideSpectrumFilter
from deepglyco.speclib.gpep.hdf import GlycoPeptideSpectralLibraryHdf
from deepglyco.speclib.gpep.inmemory import GlycoPeptideSpectralLibraryInMemory
from deepglyco.speclib.gpep.spec import GlycoPeptideMS2Spectrum
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def build_consesus_gpspeclib(source_files, dest_file, **filter_args):
    log_file = rf"{os.path.splitext(dest_file)[0]}.log"
    logger = get_logger(file=log_file)
    progress_factory = TqdmProgressFactory()

    nr_spec = NonRedundantSpectraConsensus()
    filter = GlycoPeptideSpectrumFilter[GlycoPeptideMS2Spectrum]()

    if logger:
        logger.info(f"Use configs: {nr_spec.get_configs()}")
        logger.info(f"Use configs: {filter_args}")

    if len(source_files) == 1:
        speclib = GlycoPeptideSpectralLibraryHdf(
            file_name=source_files[0],
            is_read_only=True,
        )
    else:
        speclib = GlycoPeptideSpectralLibraryInMemory()
        for source_file in source_files:
            speclib_ = GlycoPeptideSpectralLibraryHdf(
                file_name=source_file,
                is_read_only=True,
            )
            if logger:
                logger.info(
                    f"Source {source_file}: {speclib_.num_spectra} spectra of {speclib_.num_precursors} precursors, "
                    f"{speclib_.num_glycopeptides} glycopeptides"
                )
            speclib.import_spectra(progress_factory(speclib_.iter_spectra(), total=speclib_.num_spectra))

    nr_speclib = GlycoPeptideSpectralLibraryHdf(
        file_name=dest_file,
        is_overwritable=True,
        is_new_file=True,
    )

    if logger:
        logger.info(
            f"Source: {speclib.num_spectra} spectra of {speclib.num_precursors} precursors, "
            f"{speclib.num_glycopeptides} glycopeptides"
        )

    nr_speclib.import_spectra(
        progress_factory(
            filter.filter_spectra(nr_spec.nonredundant_spectra(speclib), **filter_args)
        )
    )

    if logger:
        logger.info(
            f"Total: {nr_speclib.num_spectra} spectra of {nr_speclib.num_precursors} precursors, "
            f"{nr_speclib.num_glycopeptides} glycopeptides"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build consensus glycopeptide spectral library."
    )
    parser.add_argument(
        "--in", dest="source_files", nargs="+", required=True, help="input speclib file"
    )
    parser.add_argument(
        "--out", dest="dest_file", required=True, help="output speclib file"
    )
    parser.add_argument("--min_num_fragments", type=eval, default=None)
    parser.add_argument("--min_num_peptide_fragments", type=int, default=5)
    parser.add_argument("--min_num_glycan_fragments", type=int, default=5)

    args = parser.parse_args()

    build_consesus_gpspeclib(**vars(args))
