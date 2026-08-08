"""Regenerate every figure from cached metrics — no training required.

Usage:
    python scripts/make_all_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpoker.figures.equity_plots import make_all_equity_figures
from rlpoker.figures.rl_plots import make_all_rl_figures


def main() -> None:
    outs = []
    outs += make_all_equity_figures()
    outs += make_all_rl_figures()
    print(f"wrote {len(outs)} figure files under outputs/figures/")


if __name__ == "__main__":
    main()
