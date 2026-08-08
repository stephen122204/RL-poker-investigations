"""Estimate equity from an OpenSpiel state.

`monte_carlo_equity(state, player, n_samples)` rolls out remaining chance events
(opponent hole cards + missing board cards) and averages the terminal utility.
Player actions in the rollout copy the current bet sequence; we hand off decisions
downstream by simulating a check/call line to showdown. That approximates
"showdown equity" — the probability of winning if both players check to the river.

Works uniformly for Leduc / Kuhn / HULHE. For Leduc, sample counts <100 give
low-variance estimates because the card space is small.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyspiel


def _rollout_to_showdown(state: pyspiel.State, rng: np.random.Generator) -> pyspiel.State:
    """Play out remaining chance events + check/call all future betting decisions."""
    s = state.clone()
    while not s.is_terminal():
        if s.is_chance_node():
            outcomes = s.chance_outcomes()
            actions, probs = zip(*outcomes)
            idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
            s.apply_action(int(actions[idx]))
        else:
            legal = s.legal_actions()
            # Prefer Call (encoded as action_str containing "Call") to reach showdown;
            # if unavailable, take the smallest-index legal action (typically fold/check).
            chosen = None
            for a in legal:
                name = s.action_to_string(a).lower()
                if "call" in name or "check" in name:
                    chosen = a
                    break
            if chosen is None:
                chosen = legal[0]
            s.apply_action(int(chosen))
    return s


def monte_carlo_equity(
    state: pyspiel.State,
    player: int,
    n_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Return the average showdown outcome for `player` from `state`.

    We map utility to [0, 1] by treating a positive final return as a "win",
    zero as a "tie" (half-credit), and negative as a "loss". This matches how
    the paper's `equity = wins / total_hands` is defined per bucket.
    """
    if rng is None:
        rng = np.random.default_rng()
    wins = 0.0
    for _ in range(n_samples):
        terminal = _rollout_to_showdown(state, rng)
        r = float(terminal.returns()[player])
        if r > 0:
            wins += 1.0
        elif r == 0:
            wins += 0.5
    return wins / n_samples


def batch_monte_carlo_equity(
    states: list[pyspiel.State],
    player: int,
    n_samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([monte_carlo_equity(s, player, n_samples, rng) for s in states])
