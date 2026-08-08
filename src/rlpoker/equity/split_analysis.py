"""Bucket-identity uniqueness diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .preprocess import preflop_bucket_key_series


def analyze_preflop_buckets(df_before_features: pd.DataFrame) -> dict:
    keys = preflop_bucket_key_series(df_before_features)
    n_rows = len(df_before_features)
    n_unique = keys.nunique()
    vc = keys.value_counts()
    max_size = int(vc.max()) if len(vc) else 0
    n_dup = int((vc > 1).sum())
    return {
        "n_rows": n_rows,
        "n_unique_bucket_keys": int(n_unique),
        "max_rows_per_bucket": max_size,
        "n_bucket_keys_with_duplicates": n_dup,
        "rows_are_unique_buckets": bool(n_rows == n_unique and max_size == 1),
    }


def analyze_postflop_street_buckets(df_with_features: pd.DataFrame, feature_cols: list[str]) -> dict:
    cols = [c for c in feature_cols if c in df_with_features.columns]
    if not cols:
        return {"n_rows": len(df_with_features), "n_unique_bucket_keys": 0, "max_rows_per_bucket": 0, "rows_are_unique_buckets": False}
    sub = df_with_features[cols].astype(str).apply(lambda row: "_".join(row.values), axis=1)
    n_rows = len(df_with_features)
    n_unique = sub.nunique()
    vc = sub.value_counts()
    max_size = int(vc.max()) if len(vc) else 0
    return {
        "n_rows": n_rows,
        "n_unique_bucket_keys": int(n_unique),
        "max_rows_per_bucket": max_size,
        "rows_are_unique_buckets": bool(n_rows == n_unique and max_size == 1),
    }


def write_split_analysis(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
