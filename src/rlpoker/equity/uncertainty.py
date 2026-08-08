"""Uncertainty quantification for equity: conformal intervals + quantile RF.

Two families:

1. **Split-conformal** wrapping any point estimator. Given calibration residuals
   {|y_i - yhat_i|}, the (1-alpha) prediction interval is
   [yhat - q_{1-alpha}, yhat + q_{1-alpha}]
   where q is the empirical quantile of calibration residuals. Marginally
   calibrated by construction under exchangeability.

2. **Quantile Random Forest** (Meinshausen 2006-style, approximated with per-leaf
   sample quantiles from a standard `RandomForestRegressor`). Gives *conditional*
   intervals that vary by input.

Both output a coverage-vs-nominal calibration curve for the figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from ..config import CONFORMAL_ALPHA, QUANTILE_RF_PARAMS


@dataclass
class ConformalPredictor:
    """Split-conformal wrapper around a point estimator."""

    base_estimator: BaseEstimator
    q_hat: float
    alpha: float

    def predict(self, X) -> np.ndarray:
        return self.base_estimator.predict(X)

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yhat = self.predict(X)
        return yhat, yhat - self.q_hat, yhat + self.q_hat


def fit_split_conformal(
    estimator: BaseEstimator,
    X,
    y,
    sample_weight: Optional[np.ndarray] = None,
    alpha: float = CONFORMAL_ALPHA,
    calibration_frac: float = 0.2,
    random_state: int = 0,
) -> ConformalPredictor:
    """Fit `estimator` on a training slice, calibrate its residual quantile on the rest."""
    idx = np.arange(len(y))
    tr_idx, cal_idx = train_test_split(idx, test_size=calibration_frac, random_state=random_state)

    est = clone(estimator)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight)[tr_idx]
    est.fit(_iloc(X, tr_idx), np.asarray(y)[tr_idx], **fit_kwargs)

    yhat_cal = est.predict(_iloc(X, cal_idx))
    y_cal = np.asarray(y)[cal_idx]
    residuals = np.abs(y_cal - yhat_cal)
    # (1-alpha) empirical quantile with the standard finite-sample correction
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q_hat = float(np.sort(residuals)[k - 1])
    return ConformalPredictor(base_estimator=est, q_hat=q_hat, alpha=alpha)


def _iloc(X, idx):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return X[idx]


class QuantileRandomForest:
    """Per-leaf quantile Random Forest (Meinshausen 2006, discrete-tree variant).

    Trains a standard RandomForestRegressor, then at prediction time collects
    leaf-membership training targets from each tree and returns empirical
    quantiles conditional on the input's leaf assignment.
    """

    def __init__(self, random_state: int = 0):
        self.model = RandomForestRegressor(random_state=random_state, **QUANTILE_RF_PARAMS)
        self._leaf_y: list[dict[int, np.ndarray]] = []
        self.random_state = random_state

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None) -> "QuantileRandomForest":
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        # build a per-tree map from leaf_id to training y values
        leaves = self.model.apply(X)  # (n_train, n_trees)
        self._leaf_y = []
        for t in range(leaves.shape[1]):
            leaf_ids = leaves[:, t]
            mapping: dict[int, list[float]] = {}
            for lid, yi in zip(leaf_ids, y):
                mapping.setdefault(int(lid), []).append(float(yi))
            self._leaf_y.append({lid: np.array(v, dtype=np.float64) for lid, v in mapping.items()})
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_quantiles(self, X, quantiles=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Return an (n, len(quantiles)) array of predicted quantiles."""
        leaves = self.model.apply(X)  # (n, n_trees)
        n = leaves.shape[0]
        out = np.zeros((n, len(quantiles)), dtype=np.float64)
        for i in range(n):
            pooled: list[np.ndarray] = []
            for t in range(leaves.shape[1]):
                lid = int(leaves[i, t])
                arr = self._leaf_y[t].get(lid)
                if arr is not None and arr.size > 0:
                    pooled.append(arr)
            if not pooled:
                out[i] = np.nan
                continue
            all_y = np.concatenate(pooled)
            out[i] = np.quantile(all_y, quantiles)
        return out

    def predict_std(self, X) -> np.ndarray:
        """Approximate per-example std as (q95 - q05) / (2 * 1.645). Handy for shaping."""
        q = self.predict_quantiles(X, quantiles=(0.05, 0.95))
        return (q[:, 1] - q[:, 0]) / (2 * 1.645)


def coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals_calibration: np.ndarray,
    alphas=(0.5, 0.4, 0.3, 0.2, 0.1, 0.05),
) -> pd.DataFrame:
    """Sweep alpha: for each nominal (1-alpha), compute empirical coverage of the split-conformal band."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    n = len(residuals_calibration)
    sorted_res = np.sort(residuals_calibration)
    for a in alphas:
        k = int(np.ceil((n + 1) * (1 - a)))
        k = min(max(k, 1), n)
        q = float(sorted_res[k - 1])
        covered = np.abs(y_true - y_pred) <= q
        rows.append(
            {
                "alpha": float(a),
                "nominal_coverage": float(1 - a),
                "empirical_coverage": float(np.mean(covered)),
                "avg_width": 2 * q,
            }
        )
    return pd.DataFrame(rows)
