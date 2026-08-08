"""Train Diffusion-CFR on Leduc (or Kuhn / HULHE).

Usage:
    python scripts/train_diffusion_cfr.py --game leduc --iters 30 --traversals 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlpoker.agents.diffusion_cfr import DiffusionCFRAgent
from rlpoker.config import METRICS_DIR
from rlpoker.envs.games import make_game
from rlpoker.evaluation.exploitability import compute_exploitability
from rlpoker.training.diffusion_cfr_solver import DiffusionCFRSolver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=("leduc", "kuhn", "hulhe"), default="leduc")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--traversals", type=int, default=40)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--n-diffusion-steps", type=int, default=20)
    p.add_argument("--inference-samples", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = make_game(args.game)

    solver = DiffusionCFRSolver(
        game=spec.game,
        policy_network_layers=(128, 128),
        advantage_hidden=(128, 128),
        n_diffusion_steps=args.n_diffusion_steps,
        inference_samples=args.inference_samples,
        num_iterations=args.log_every,
        num_traversals=args.traversals,
        learning_rate=args.lr,
        memory_capacity=1_000_000,
        device="cpu",
        seed=args.seed,
    )
    agent = DiffusionCFRAgent(solver)

    curve: list[dict] = []
    for chunk in range(0, args.iters, args.log_every):
        solver.solve()
        expl = compute_exploitability(spec.game, agent) if args.game in ("leduc", "kuhn") else float("nan")
        row = {"iteration": chunk + args.log_every, "exploitability": expl, "agent": "diffusion_cfr", "game": args.game}
        curve.append(row)
        print(f"iter {row['iteration']:5d}  exploitability={row['exploitability']:.4f}")

    suffix = f"_seed{args.seed}" if args.seed != 42 else ""
    out_json = METRICS_DIR / f"exploitability_curve_diffusion_cfr_{args.game}{suffix}.json"
    out_json.write_text(json.dumps(curve, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
