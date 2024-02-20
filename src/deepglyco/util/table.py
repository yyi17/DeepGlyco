__all__ = ["filter_dataframe", "merge_dataframes"]

import functools
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd


def filter_dataframe(data: pd.DataFrame, filters: Mapping[str, Mapping[str, Any]]):
    if filters is None:
        return data

    should_remove = np.zeros(data.shape[0], dtype=np.bool_)
    for column, operators in filters.items():
        if column not in data.columns:
            continue
        for op, value in operators.items():
            inverse = False
            op = op.lower()
            if op.startswith("!"):
                inverse = True
                op = op[1:]
            elif op.startswith("not"):
                inverse = True
                op = op[3:]

            if op in {"=", "==", "eq", "equal", "equals"}:
                should_keep = data[column].eq(value)
            elif op in {">", "gt", "greaterthan"}:
                should_keep = data[column].gt(value)
            elif op in {"<", "lt", "lessthan"}:
                should_keep = data[column].lt(value)
            elif op in {">=", "ge", "greaterthanorequal"}:
                should_keep = data[column].gt(value)
            elif op in {"<=", "le", "lessthanorequal"}:
                should_keep = data[column].lt(value)
            elif op in {"ne", "equal"}:
                should_keep = data[column].ne(value)
            elif op in {"startwith", "startswith"}:
                should_keep = data[column].str.startswith(value, na=False)
            elif op in {"endwith", "endswith"}:
                should_keep = data[column].str.endswith(value, na=False)
            elif op in {"contain", "contains"}:
                should_keep = data[column].str.contains(value, na=False)
            else:
                raise ValueError(f"unknown filter operator: {op}")

            if inverse:
                should_keep = ~should_keep

            should_remove = np.bitwise_or(
                should_remove, ~cast(np.ndarray, should_keep.values)
            )

    return data.loc[~should_remove]


def merge_dataframes(
    data: Iterable[pd.DataFrame],
    how: Literal["inner", "left", "right", "outer"],
    on: Sequence[str],
    copy: bool = True,
):
    return functools.reduce(
        lambda t1, t2: (
            None,
            pd.merge(
                t1[1],
                t2[1],
                how=how,
                suffixes=("", f"_{t2[0]}"),
                on=on,
                copy=copy,
            ),
        ),
        enumerate(data),
    )[1]
