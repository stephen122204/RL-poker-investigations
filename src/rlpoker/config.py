"""Central config: paths, seeds, hyperparameters for equity, CFR, and diffusion."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Data ------------------------------------------------------------------

KAGGLE_DATASET_HANDLE = "benjaminniesmertelny/texas-holdem-monte-carlo-data"
DATA_DOWNLOAD_ROOT = PROJECT_ROOT / "data" / "raw"
DATA_CSV_DIR = DATA_DOWNLOAD_ROOT / "data"

CSV_PREFLOP = "preflop_equity.csv"
CSV_FLOP = "flop_equity.csv"
CSV_TURN = "turn_equity.csv"
CSV_RIVER = "river_equity.csv"
CSV_HAND_DIST = "hand_dist.csv"

STAGE_TO_FILE = {
    "preflop": CSV_PREFLOP,
    "flop": CSV_FLOP,
    "turn": CSV_TURN,
    "river": CSV_RIVER,
}
STAGES = tuple(STAGE_TO_FILE)

TARGET_EQUITY = "equity"

# ---- Outputs ---------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

for _d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR, DATA_CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Reproducibility -------------------------------------------------------

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

# ---- Equity model hyperparameters -----------------------------------------

RF_PARAMS = dict(n_estimators=200, max_depth=12, n_jobs=-1)
RIDGE_PARAMS = dict(alpha=1.0)
LIGHTGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    reg_lambda=1.0,
    n_jobs=-1,
    verbosity=-1,
)
CATBOOST_PARAMS = dict(
    iterations=600,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    verbose=False,
    allow_writing_files=False,
)
MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=400,
    early_stopping=True,
    n_iter_no_change=20,
    learning_rate_init=1e-3,
)

# Conformal / quantile-RF uncertainty settings
CONFORMAL_ALPHA = 0.1  # 90% intervals
QUANTILE_RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1)

# ---- CFR / diffusion (used in M3+) ----------------------------------------

CFR_PARAMS = dict(
    n_iterations=500,
    traversals_per_iter=1000,
    advantage_hidden=(128, 128),
    policy_hidden=(128, 128),
    advantage_lr=1e-3,
    policy_lr=1e-3,
    advantage_batch=256,
    policy_batch=256,
    reservoir_size=200_000,
)
DIFFUSION_PARAMS = dict(
    n_diffusion_steps=20,
    beta_start=1e-4,
    beta_end=0.02,
    embed_dim=64,
    hidden=(128, 128),
    lr=1e-3,
    batch=256,
)
