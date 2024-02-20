import copy
import os
from typing import cast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
import statsmodels.api as sm
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score


def isotonic_lowess(x, y, lowess_frac, lowess_it=3):
    r = sm.nonparametric.lowess(y, x, frac=lowess_frac, it=lowess_it)

    # https://github.com/statsmodels/statsmodels/issues/2449
    if any(np.isnan(r[:, 1])):
        data = pd.DataFrame.from_dict({"x": x, "y": y}).groupby(x).mean()
        x = data["x"]
        y = data["y"]
        r = sm.nonparametric.lowess(y, x, frac=lowess_frac, it=lowess_it)

    x, y = r[:, 0], r[:, 1]
    iso_reg = IsotonicRegression()
    y = iso_reg.fit_transform(x, y)

    x, index = np.unique(x, return_index=True)
    y = y[index]

    interp = interp1d(x, y, bounds_error=False, fill_value=cast(float, "extrapolate"))
    return interp


class IsotonicLowessEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y):
        self.model = isotonic_lowess(
            np.squeeze(X, axis=1),
            y,
            **{k: v for k, v in self.params.items() if k.startswith("lowess_")},
        )
        return self

    def get_params(self, deep=False):
        r = self.params
        if deep:
            r = copy.deepcopy(r)
        return r

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def score(self, X, y):
        return r2_score(y, self.model(np.squeeze(X, axis=1)))

    def predict(self, X):
        return self.model(np.squeeze(X, axis=1))


def isotonic_lowess_RANSAC(x, y, **kwargs):
    X = np.expand_dims(x, axis=1)

    estimator = IsotonicLowessEstimator(**kwargs, lowess_frac=0.1, lowess_it=3)

    gsc = GridSearchCV(
        estimator=estimator,
        param_grid={'lowess_frac': [0.05, 0.1, 0.2, 0.25, 0.333, 0.5, 0.667], "lowess_it": [0, 1, 2, 3]},
        cv=KFold(4, shuffle=True, random_state=0),
        n_jobs=os.cpu_count()
    )
    gsc.fit(X, y)

    # print(gsc.best_estimator_.get_params())

    reg = RANSACRegressor(
        gsc.best_estimator_,
        min_samples=5,
        max_trials=10000,
        residual_threshold=50,
        random_state=0,
    )
    reg.fit(X, y)

    return reg
