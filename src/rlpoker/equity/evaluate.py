"""Weighted / unweighted MAE, RMSE, R² for equity regression."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn import metrics


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is None:
        return {
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
            "r2": float(metrics.r2_score(y_true, y_pred)),
        }
    sw = np.asarray(sample_weight, dtype=np.float64)
    if sw.ndim != 1 or sw.shape[0] != y_true.shape[0]:
        raise ValueError("sample_weight must be 1-D and match y_true length.")
    if np.any(sw < 0):
        raise ValueError("sample_weight must be non-negative.")
    sw_sum = float(np.sum(sw))
    if sw_sum <= 0:
        raise ValueError("sum(sample_weight) must be positive.")
    resid = y_true - y_pred
    mae = float(np.sum(np.abs(resid) * sw) / sw_sum)
    rmse = float(np.sqrt(np.sum((resid**2) * sw) / sw_sum))
    r2 = float(metrics.r2_score(y_true, y_pred, sample_weight=sw))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_cv_metrics(fold_reports: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not fold_reports:
        return {}
    keys = fold_reports[0].keys()
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([float(fr[k]) for fr in fold_reports], dtype=np.float64)
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    return out
