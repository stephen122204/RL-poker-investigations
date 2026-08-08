"""Run the unified equity benchmark across streets and models.

Usage:
    python scripts/train_equity.py --stage all
    python scripts/train_equity.py --stage preflop --models baseline_mean ridge random_forest lightgbm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_equity.py` from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpoker.config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from rlpoker.equity.models import MODEL_REGISTRY
from rlpoker.equity.reporting import write_stage_comparison_csv
from rlpoker.equity.train import run_all_stages, run_side_artifacts, run_stage_benchmark
from rlpoker.figures.equity_plots import make_all_equity_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("preflop", "flop", "turn", "river", "all"), default="all")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--no-side-artifacts", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stages = ("preflop", "flop", "turn", "river") if args.stage == "all" else (args.stage,)

    results = {}
    for stage in stages:
        res = run_stage_benchmark(
            stage,
            models=args.models,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
        )
        if not args.no_side_artifacts:
            run_side_artifacts(res, random_state=args.random_state)
        results[stage] = res

    try:
        p = write_stage_comparison_csv()
        print(f"stage_comparison.csv -> {p}")
    except FileNotFoundError:
        pass

    if not args.no_figures:
        outs = make_all_equity_figures()
        print(f"wrote {len(outs)} figure files under outputs/figures/")


if __name__ == "__main__":
    main()
