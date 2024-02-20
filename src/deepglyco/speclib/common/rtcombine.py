import os
from typing import Optional, Union, cast
import numpy as np
import pandas as pd

from ...util.config import Configurable
from .rtlib import RetentionTimeLibraryBase


COLUMN_RUN_NAME = "run_name"
COLUMN_RT = "retention_time"
COLUMN_SCORE = "score"


class NonRedundantRetentionTimeConsensus(Configurable):
    def __init__(
        self,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "consensus.yaml"
            )
        super().__init__(configs)

    def nonredundant_retention_time(
        self, rtlib: RetentionTimeLibraryBase
    ) -> Optional[pd.DataFrame]:
        retention_time = rtlib.get_retention_time()
        if retention_time is None:
            return None
        return self.combine_retention_time(retention_time)

    def combine_retention_time(self, data: pd.DataFrame):
        max_replicates_combined = self.get_config("max_replicates_combined", typed=int)
        min_replicates_combined = self.get_config("min_replicates_combined", typed=int)
        discard_min_replicates = self.get_config("discard_min_replicates", required=False, typed=bool)

        use_score_weight = self.get_config("use_score_weight", typed=bool)

        rt_difference_threshold = self.get_config(
            "rt_difference_threshold", typed=float, allow_convert=True
        )

        def combine_replicates(group: pd.DataFrame):
            if len(group) < min_replicates_combined:
                if discard_min_replicates:
                    return group.head(0)
                else:
                    return group.iloc[[cast(int, group[COLUMN_SCORE].argmax())]]

            if len(group) == 1:
                return group
            if len(group) > max_replicates_combined:
                group = group.sort_values(by=COLUMN_SCORE, ascending=False).head(
                    max_replicates_combined
                )

            num_trials = 0
            while num_trials < 100:
                num_trials += 1

                if use_score_weight:
                    score_weight = (
                        group[COLUMN_SCORE]
                        .divide(group[COLUMN_SCORE].sum(skipna=True))
                        .values.astype(np.float_)
                    )
                    rt = group[COLUMN_RT].multiply(score_weight).sum()
                else:
                    rt = group[COLUMN_RT].mean()

                delta = group[COLUMN_RT].subtract(rt).abs()
                if not delta.gt(rt_difference_threshold).any():
                    break
                group = group.drop(index=group.index[delta.argmax()])
                if discard_min_replicates and len(group) < min_replicates_combined:
                    return group.head(0)

            r = group.iloc[[0]].copy()
            r[COLUMN_RT] = rt # type: ignore
            return r

        keys = data.columns.difference(
            [COLUMN_RUN_NAME, COLUMN_RT, COLUMN_SCORE]
        ).tolist()
        data = data.groupby(
            keys,
            group_keys=False,
            as_index=False,
            sort=False,
        ).apply(combine_replicates)
        return data
