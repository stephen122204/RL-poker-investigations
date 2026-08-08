"""Consolidate per-street benchmark tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import METRICS_DIR, STAGES

STAGE_RESULT_FILES = {stage: f"{stage}_results.csv" for stage in STAGES}


def write_stage_comparison_csv(output_dir: Path = METRICS_DIR) -> Path:
    parts: list[pd.DataFrame] = []
    for stage, fname in STAGE_RESULT_FILES.items():
        p = output_dir / fname
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "stage" not in df.columns:
            df.insert(0, "stage", stage)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No stage result CSVs found under {output_dir}.")
    out = pd.concat(parts, ignore_index=True)
    out_path = output_dir / "stage_comparison.csv"
    out.to_csv(out_path, index=False)
    return out_path
