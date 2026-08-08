"""Weighted absolute-error breakdowns by feature category, given a pre-fit model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def _weighted_mae(abs_err: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(abs_err * weights) / np.sum(weights))


def grouped_weighted_summary(
    abs_err: np.ndarray,
    weights: np.ndarray,
    group: pd.Series,
    stage: str,
    group_feature: str,
) -> pd.DataFrame:
    df = pd.DataFrame({"group_value": group.astype(str).values, "abs_err": abs_err, "w": weights})
    rows = []
    for val, sub in df.groupby("group_value", sort=False):
        rows.append(
            {
                "stage": stage,
                "group_feature": group_feature,
                "group_value": val,
                "n_test_rows": int(len(sub)),
                "weighted_mean_abs_error": _weighted_mae(sub["abs_err"].values, sub["w"].values),
                "unweighted_mean_abs_error": float(np.mean(sub["abs_err"].values)),
            }
        )
    return pd.DataFrame(rows)


def run_error_analysis(
    fitted_model: BaseEstimator,
    X_test,
    y_test: np.ndarray,
    weights_test: np.ndarray,
    meta_test: pd.DataFrame,
    stage: str,
    group_cols: list[str],
    out_path: Path,
) -> Path:
    pred = fitted_model.predict(X_test)
    ae = np.abs(np.asarray(y_test) - pred)
    parts: list[pd.DataFrame] = []
    for col in group_cols:
        if col not in meta_test.columns:
            continue
        parts.append(grouped_weighted_summary(ae, weights_test, meta_test[col], stage, col))
    if stage == "preflop" and {"players", "suited"}.issubset(meta_test.columns):
        combo = (
            meta_test["players"].astype(str)
            + "_players_"
            + meta_test["suited"].map({True: "suited", False: "offsuit", 1: "suited", 0: "offsuit"})
        )
        parts.append(grouped_weighted_summary(ae, weights_test, combo, stage, "players_x_suited"))
    if not parts:
        return out_path
    out = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


PREFLOP_GROUP_COLS = ["players", "suited"]
POSTFLOP_GROUP_COLS = [
    "players",
    "hand_class",
    "pocket_pair",
    "suited",
    "board_connectivity",
    "flush_potential",
    "straight_potential",
]
