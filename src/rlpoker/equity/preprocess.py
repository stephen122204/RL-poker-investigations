"""Cleaning, equity target, and structural-NaN handling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TARGET_EQUITY

STRUCTURAL_NA_TOKEN = "not_applicable"


def fill_structural_postflop_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("flush_potential", "straight_potential"):
        if c in out.columns:
            out[c] = out[c].fillna(STRUCTURAL_NA_TOKEN)
    return out


def add_equity_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wins" in out.columns and "total_hands" in out.columns:
        out[TARGET_EQUITY] = out["wins"].astype(np.float64) / out["total_hands"].astype(np.float64)
    elif "total_wins" in out.columns and "total_hands" in out.columns:
        out[TARGET_EQUITY] = out["total_wins"].astype(np.float64) / out["total_hands"].astype(np.float64)
    else:
        raise ValueError("Need (wins, total_hands) or (total_wins, total_hands) to compute equity.")
    return out


def drop_zero_total_hands(df: pd.DataFrame) -> pd.DataFrame:
    if "total_hands" not in df.columns:
        return df
    return df.loc[df["total_hands"] > 0].copy()


def select_feature_columns_preflop(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["suited", "players", "r1", "r2"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for preflop: {missing}")
    return df[cols].copy()


def postflop_feature_names(df: pd.DataFrame) -> list[str]:
    base = [
        "players",
        "hand_class",
        "pocket_pair",
        "suited",
        "overcards",
        "suit_texture",
        "rank_texture",
        "board_connectivity",
    ]
    optional = ["flush_potential", "straight_potential"]
    return [c for c in base + optional if c in df.columns]


def prepare_preflop(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_zero_total_hands(df)
    df = add_equity_target(df)
    feats = select_feature_columns_preflop(df)
    return pd.concat([feats, df[[TARGET_EQUITY]], df[["total_hands"]]], axis=1)


def prepare_postflop_with_weights(
    df: pd.DataFrame,
    drop_rows_with_nan: bool = False,
    handle_structural_nan: bool = True,
) -> pd.DataFrame:
    df = drop_zero_total_hands(df)
    df = add_equity_target(df)
    feat_names = postflop_feature_names(df)
    if not feat_names:
        raise ValueError("No postflop feature columns found.")
    cols = feat_names + [TARGET_EQUITY, "total_hands"]
    out = df[cols].copy()
    if handle_structural_nan:
        out = fill_structural_postflop_nan(out)
    if drop_rows_with_nan:
        out = out.dropna(subset=feat_names)
    return out


def preflop_bucket_key_series(df: pd.DataFrame) -> pd.Series:
    cols = ["suited", "players", "r1", "r2"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"preflop_bucket_key_series missing column: {c}")
    s = df["suited"].map({True: "1", False: "0", 1: "1", 0: "0"})
    return (
        s.astype(str)
        + "_"
        + df["players"].astype(str)
        + "_"
        + df["r1"].astype(str)
        + "_"
        + df["r2"].astype(str)
    )
