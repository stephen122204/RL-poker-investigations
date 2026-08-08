"""Download the Kaggle Monte Carlo bucket dataset and load per-street tables."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import (
    DATA_CSV_DIR,
    DATA_DOWNLOAD_ROOT,
    KAGGLE_DATASET_HANDLE,
    STAGE_TO_FILE,
)


def ensure_dataset_downloaded(
    output_root: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Return the directory containing the equity CSVs, downloading if needed."""
    root = output_root or DATA_DOWNLOAD_ROOT
    csv_dir = root / "data"
    preflop = csv_dir / STAGE_TO_FILE["preflop"]
    if preflop.is_file() and not force:
        return csv_dir

    try:
        import kagglehub
    except ImportError as e:
        raise ImportError("pip install kagglehub") from e

    # kagglehub 1.x downloads to a managed cache; copy or point directly.
    downloaded = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, force_download=force)
    downloaded_path = Path(downloaded)

    root.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    for src in downloaded_path.rglob("*.csv"):
        dst = csv_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    if not preflop.is_file():
        raise FileNotFoundError(f"Expected {preflop} after download; got {list(csv_dir.iterdir())}")
    return csv_dir


def load_equity_table(csv_dir: Path, stage: str = "preflop") -> pd.DataFrame:
    if stage not in STAGE_TO_FILE:
        raise ValueError(f"stage must be one of {list(STAGE_TO_FILE)}")
    path = csv_dir / STAGE_TO_FILE[stage]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_hand_distribution(csv_dir: Path) -> pd.DataFrame:
    path = csv_dir / "hand_dist.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
