__all__ = ["GlycoPeptideReportParserBase", "GlycoPeptideReportSpectrumParserBase"]

import abc
from typing import Iterable, Optional, Union

import pandas as pd

from ....specio.spec import MassSpectrum
from ...pep.parser.abs import PeptideReportParserBase
from ..filter import GlycoPeptideSpectrumFilter
from ..spec import GlycoPeptideMS2Spectrum
from .annotation import GlycoPeptideMS2SpectrumAnnotator


class GlycoPeptideReportParserBase(PeptideReportParserBase):
    def __init__(self, config: Union[str, dict]):
        super().__init__(config)

    def parse_psm_report(
        self, report: pd.DataFrame
    ) -> Iterable[GlycoPeptideMS2Spectrum]:
        report = self.filter_report(report)
        for i, row in report.iterrows():
            yield self.parse_psm_report_row(row)

    @abc.abstractmethod
    def parse_psm_report_row(self, row: pd.Series) -> GlycoPeptideMS2Spectrum:
        pass


class GlycoPeptideReportSpectrumParserBase(GlycoPeptideReportParserBase):
    def __init__(
        self,
        annotator: GlycoPeptideMS2SpectrumAnnotator,
        spectrum_filter: GlycoPeptideSpectrumFilter,
        configs: Union[str, dict],
    ):
        self.annotator = annotator
        self.spectrum_filter = spectrum_filter
        super().__init__(configs)

    def parse_psm_report(
        self, report: pd.DataFrame, spectra: Iterable[MassSpectrum]
    ) -> Iterable[GlycoPeptideMS2Spectrum]:
        report = self.filter_report(report)

        def _parse_psm_report():
            for spec in spectra:
                if spec.ms_level != 2:
                    continue
                row = self.find_psm_report_row(report, spec)
                if row is not None:
                    yield self.parse_psm_report_row(row, spec)

        result = _parse_psm_report()

        filter_config = self.get_config("spectrum_filter", required=False, typed=dict)
        if filter_config is not None:
            result = self.spectrum_filter.filter_spectra(result, **filter_config)
        return result

    @abc.abstractmethod
    def find_psm_report_row(
        self, report: pd.DataFrame, spectrum: MassSpectrum
    ) -> Optional[pd.Series]:
        pass

    @abc.abstractmethod
    def get_psm_report_row_info(self, row: pd.Series) -> dict:
        pass

    def parse_psm_report_row(
        self, row: pd.Series, spectrum: MassSpectrum
    ) -> GlycoPeptideMS2Spectrum:
        kwargs = self.get_psm_report_row_info(row)
        kwargs.update(
            {
                "run_name": spectrum.run_name,
                "scan_number": spectrum.scan_number,
            }
        )
        return self.annotator.annotate(spectrum, **kwargs)
