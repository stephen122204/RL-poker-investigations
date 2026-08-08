"""Round-robin tournament + optional exploitability computation.

Usage:
    python scripts/evaluate_agents.py --game leduc --agents random equity_threshold --n-hands 4000
    python scripts/evaluate_agents.py --game hulhe --agents random equity_threshold --n-hands 2000 --no-exploitability
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpoker.agents.equity_threshold import EquityThresholdAgent
from rlpoker.agents.random_agent import RandomAgent
from rlpoker.config import METRICS_DIR
from rlpoker.envs.games import make_game
from rlpoker.evaluation.exploitability import compute_exploitability
from rlpoker.evaluation.head_to_head import round_robin


def build_agent(kind: str, spec) -> object:
    if kind == "random":
        return RandomAgent(spec.num_actions)
    if kind == "equity_threshold":
        return EquityThresholdAgent(spec.num_actions, n_mc_samples=30)
    raise ValueError(f"Unknown agent kind '{kind}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=("leduc", "hulhe", "kuhn"), default="leduc")
    p.add_argument("--agents", nargs="+", default=["random", "equity_threshold"])
    p.add_argument("--n-hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-exploitability", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = make_game(args.game)
    agents = [build_agent(k, spec) for k in args.agents]
    df = round_robin(spec, agents, n_hands=args.n_hands, seed=args.seed)
    out = METRICS_DIR / f"head_to_head_{spec.name}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")

    if not args.no_exploitability and args.game in ("leduc", "kuhn"):
        rows = []
        for k, agent in zip(args.agents, agents):
            expl = compute_exploitability(spec.game, agent)
            rows.append({"game": spec.name, "agent": k, "exploitability": expl})
            print(f"exploitability({spec.name}, {k}) = {expl:.4f}")
        import pandas as pd

        out2 = METRICS_DIR / f"exploitability_{spec.name}.csv"
        pd.DataFrame(rows).to_csv(out2, index=False)
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
