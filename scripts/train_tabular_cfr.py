"""Tabular CFR reference curve (converges cleanly on Leduc / Kuhn)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_spiel.python.algorithms import cfr

from rlpoker.agents.tabular_cfr import TabularCFRAgent
from rlpoker.config import METRICS_DIR
from rlpoker.envs.games import make_game
from rlpoker.evaluation.exploitability import compute_exploitability


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=("leduc", "kuhn"), default="leduc")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = make_game(args.game)
    solver = cfr.CFRSolver(spec.game)
    curve: list[dict] = []
    for i in range(1, args.iters + 1):
        solver.evaluate_and_update_policy()
        if i % args.log_every == 0:
            agent = TabularCFRAgent(solver, spec.game)
            expl = compute_exploitability(spec.game, agent)
            curve.append({"iteration": i, "exploitability": expl, "agent": "tabular_cfr", "game": args.game})
            print(f"iter {i:5d}  exploitability={expl:.4f}", flush=True)

    out_json = METRICS_DIR / f"exploitability_curve_tabular_cfr_{args.game}.json"
    out_json.write_text(json.dumps(curve, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
