"""Potential-based reward shaping using estimated equity.

Given a potential function `F(s)` on states, Ng, Harada & Russell (1999) proved
that shaping the reward with `F(s') - gamma F(s)` (gamma = discount factor)
preserves the optimal policy. Here gamma = 1 (poker is episodic + non-discounted from
the agent's perspective) and take `F(s) = ê(s)` (equity for the acting player).

Two equity sources are supported:

  - `MCEquityPotential(n_samples=...)`   — uses `shaping.equity_bridge.monte_carlo_equity`.
                                             Works on any OpenSpiel state.
  - `LookupEquityPotential(fn=...)`      — user-supplied callable, e.g. RF model.

Uncertainty-modulated shaping (novel small angle from the plan): the shaping
magnitude is scaled by `(1 − û(s))` where `û ∈ [0, 1]` is a normalized
uncertainty estimate; more confident equity gets more weight in the shaping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyspiel

from .equity_bridge import monte_carlo_equity


EquityFn = Callable[[pyspiel.State, int], float]
UncertaintyFn = Optional[Callable[[pyspiel.State, int], float]]


@dataclass
class MCEquityPotential:
    n_samples: int = 30
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, state: pyspiel.State, player: int) -> float:
        return monte_carlo_equity(state, player, self.n_samples, self._rng)


@dataclass
class LookupEquityPotential:
    fn: EquityFn

    def __call__(self, state: pyspiel.State, player: int) -> float:
        return float(self.fn(state, player))


def shaping_reward(
    state: pyspiel.State,
    next_state: pyspiel.State,
    player: int,
    potential: EquityFn,
    uncertainty: UncertaintyFn = None,
    coef: float = 1.0,
) -> float:
    """Return `coef * (1 - û(s)) * (F(s') - F(s))` for the acting player."""
    f_s = potential(state, player)
    f_next = potential(next_state, player) if not next_state.is_terminal() else 0.0
    delta = f_next - f_s
    scale = coef
    if uncertainty is not None:
        u = float(uncertainty(state, player))
        u = min(max(u, 0.0), 1.0)
        scale = coef * (1.0 - u)
    return scale * delta
