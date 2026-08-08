"""Build (X, y, weights, groups) design matrices per street."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TARGET_EQUITY
from .features import add_preflop_rank_features, bool_to_int
from .preprocess import (
    prepare_postflop_with_weights,
    prepare_preflop,
    preflop_bucket_key_series,
)


def _postflop_columns_for_one_hot(dfp: pd.DataFrame) -> list[str]:
    exclude = {TARGET_EQUITY, "total_hands"}
    out: list[str] = []
    for c in dfp.columns:
        if c in exclude:
            continue
        s = dfp[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        out.append(c)
    return out


def build_xy_preflop(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    df = prepare_preflop(df)
    keys = preflop_bucket_key_series(df)
    _, groups = pd.factorize(keys)
    df = add_preflop_rank_features(df)
    df = bool_to_int(df, ["suited"])
    weights = df["total_hands"].to_numpy(dtype=np.float64)
    y = df[TARGET_EQUITY]
    feature_cols = [c for c in df.columns if c not in (TARGET_EQUITY, "total_hands")]
    X = df[feature_cols].copy()
    return X, y, weights, groups.astype(np.int64)


def build_xy_postflop_street(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    dfp = prepare_postflop_with_weights(df)
    cat_cols = _postflop_columns_for_one_hot(dfp)
    for c in cat_cols:
        dfp[c] = dfp[c].astype(str)
    if cat_cols:
        dfp = pd.get_dummies(dfp, columns=cat_cols, drop_first=False)
    bool_cols = [c for c in dfp.columns if dfp[c].dtype == bool]
    dfp = bool_to_int(dfp, bool_cols)
    weights = dfp["total_hands"].to_numpy(dtype=np.float64)
    y = dfp[TARGET_EQUITY]
    X = dfp.drop(columns=[TARGET_EQUITY, "total_hands"])
    return X, y, weights
