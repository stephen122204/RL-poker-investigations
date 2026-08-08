"""Feature-engineering helpers migrated from poker-ml."""
from __future__ import annotations

import pandas as pd


def add_preflop_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "r1" not in out.columns or "r2" not in out.columns:
        return out
    out["rank_high"] = out[["r1", "r2"]].max(axis=1)
    out["rank_low"] = out[["r1", "r2"]].min(axis=1)
    out["rank_gap"] = (out["r1"] - out["r2"]).abs()
    return out


def bool_to_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns and out[c].dtype == bool:
            out[c] = out[c].astype(int)
    return out
