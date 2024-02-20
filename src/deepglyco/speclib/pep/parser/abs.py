__all__ = ["PeptideReportParserBase"]

import abc
from typing import Iterable, Union

import pandas as pd

from ....util.config import Configurable

from ..spec import PeptideMS2Spectrum
from ....util.table import filter_dataframe


class PeptideReportParserBase(abc.ABC, Configurable):
    def __init__(self, configs: Union[str, dict]):
       super().__init__(configs)

    def filter_report(self, report: pd.DataFrame) -> pd.DataFrame:
        config = self.get_config("report_filter", required=False, typed=dict)
        if not isinstance(config, dict):
            return report
        return filter_dataframe(report, config)

    def parse_psm_report(self, report: pd.DataFrame) -> Iterable[PeptideMS2Spectrum]:
        report = self.filter_report(report)
        for i, row in report.iterrows():
            yield self.parse_psm_report_row(row)

    @abc.abstractmethod
    def parse_psm_report_row(self, row: pd.Series) -> PeptideMS2Spectrum:
        pass
