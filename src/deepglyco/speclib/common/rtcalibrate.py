from logging import Logger
from typing import Literal, Optional

import numpy as np
import pandas as pd

COLUMN_RUN_NAME = "run_name"
COLUMN_RT = "retention_time"
COLUMN_CALIBRATED_RT = "calibrated_retention_time"
COLUMN_ORIGINAL_RT = "original_retention_time"


class RetentionTimeCalibrator:
    def __init__(
        self,
        method: Literal["lowess"] = "lowess",
        method_args: Optional[dict] = None,
        min_anchors: int = 5,
        bound_factor: float = 0.1,
        logger: Optional[Logger] = None,
    ):
        self.method = method
        self.method_args = method_args
        self.min_anchors = min_anchors
        self.bound_factor = bound_factor
        self.logger = logger
        self.build_calibrate_fn()

    def build_calibrate_fn(self):
        # if self.method == "linear":

        #     def linear(x, y, new_x):
        #         coef = np.polyfit(x, y, 1)
        #         return coef[0] * new_x + coef[1]

        #     self.calibrate_fn = linear

        if self.method == "lowess":
            from ...util.regression import isotonic_lowess_RANSAC

            def lowess(x, y, new_x):
                lowess_args = self.method_args or {}

                model = isotonic_lowess_RANSAC(x, y, **lowess_args)
                new_y = model.predict(np.expand_dims(new_x, axis=1))
                inlier_mask = model.inlier_mask_
                score = model.score(np.expand_dims(x[inlier_mask], axis=1), y[inlier_mask])
                return new_y, score, inlier_mask

            self.calibrate_fn = lowess
        else:
            raise ValueError(f"invalid model: {self.method}")

    def load_reference(self, reference: pd.DataFrame):
        reference[COLUMN_RT]
        self.reference = reference


    def calibrate_rt(self, data: pd.DataFrame):
        anchor_list = []
        r2_dict = {}

        def calculate_rt(data: pd.DataFrame):
            keys = self.reference.columns.difference([COLUMN_RT, COLUMN_CALIBRATED_RT])
            keys = data.columns.intersection(keys).tolist()

            reference = self.reference
            if COLUMN_RUN_NAME in data.columns:
                run_name = data[COLUMN_RUN_NAME].iloc[0]
                if COLUMN_RUN_NAME in reference.columns and COLUMN_RUN_NAME in data.columns:
                    reference = reference.loc[reference[COLUMN_RUN_NAME] == run_name]
            else:
                run_name = "All"

            min_rt = reference[COLUMN_RT].min()
            max_rt = reference[COLUMN_RT].max()
            bound_margin = (max_rt - min_rt) * self.bound_factor
            min_rt -= bound_margin
            max_rt += bound_margin
            reference = reference.drop_duplicates(subset=keys)

            COLUMN_SUFFIX_REF = "_reference"
            COLUMN_RT_REF = f"{COLUMN_RT}{COLUMN_SUFFIX_REF}"
            merged_data = data.merge(
                reference,
                on=keys,
                suffixes=("", COLUMN_SUFFIX_REF),
                copy=False,
            )

            if COLUMN_ORIGINAL_RT in reference.columns:
                merged_data = pd.concat(
                    (
                        merged_data[keys + [COLUMN_RT, COLUMN_RT_REF]],
                        reference[keys + [COLUMN_ORIGINAL_RT, COLUMN_RT]].rename(columns={
                            COLUMN_RT: COLUMN_RT_REF,
                            COLUMN_ORIGINAL_RT: COLUMN_RT
                        })
                    ),
                    ignore_index=True,
                    copy=False,
                )
                merged_data = merged_data.drop_duplicates(subset=keys)

            if len(merged_data) < self.min_anchors:
                if self.logger:
                    self.logger.warn(
                        f"Too few anchors ({len(merged_data)} < {self.min_anchors}). Skipping run {run_name}"
                    )
                return pd.Series(np.array([], dtype=np.float_))

            index = np.argsort(merged_data[COLUMN_RT_REF])
            y = merged_data[COLUMN_RT_REF][index].values
            x = merged_data[COLUMN_RT][index].values
            y_new, r2, inlier_mask = self.calibrate_fn(x, y, data[COLUMN_RT].values)
            y_new = np.clip(y_new, a_min=min_rt, a_max=max_rt)
            merged_data["inlier"] = np.zeros(len(merged_data), dtype=np.bool_)
            merged_data["inlier"].iloc[index] = inlier_mask
            if self.logger:
                self.logger.info(
                    f"Run {run_name}: {sum(inlier_mask)} inlier anchors, R2={r2}"
                )

            if COLUMN_RUN_NAME in data.columns and COLUMN_RUN_NAME not in merged_data.columns:
                merged_data.insert(0, COLUMN_RUN_NAME, data[COLUMN_RUN_NAME].iloc[0])
            anchor_list.append(merged_data)
            r2_dict[run_name] = r2
            return pd.Series(y_new, index=data.index)

        assert COLUMN_RT in data.columns
        if COLUMN_RUN_NAME not in data.columns:
            rt_new = calculate_rt(data)
        else:
            rt_new = pd.Series(np.nan, index=data.index)
            rt_new = data.groupby(
                by=COLUMN_RUN_NAME,
                group_keys=False,
                as_index=False,
                sort=False,
            ).apply(calculate_rt)

        if COLUMN_CALIBRATED_RT in data.columns:
            data = data.drop(columns=[COLUMN_CALIBRATED_RT])
        data = pd.concat(
            (
                data.loc[rt_new.index],
                pd.DataFrame(rt_new, columns=[COLUMN_CALIBRATED_RT]),
            ),
            axis=1,
            copy=False,
        )

        anchors = pd.concat(anchor_list, copy=False, ignore_index=True)
        r2 = pd.Series(r2_dict, name="r2")
        return data, anchors, r2
