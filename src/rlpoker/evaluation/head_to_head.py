"""Head-to-head tournaments over N hands.

`play_hand(game, agents, rng)` returns the per-player utility of one dealt hand.
`round_robin(game, agents, n_hands)` runs a per-pair tournament and returns a
long-form dataframe: rows are (a, b, n_hands, mean_a_returns, ...).

Utilities are converted to mBB/100 using the game's `big_blind` field:
  mBB/100 = 100 * (u / big_blind) * 1000

SIGN CONVENTION (do not change silently — the paper cites this):
  `mean_a`, `a_mbb_per_100`, and `a_mbb_ci95` are ALL from agent A's perspective.
  A positive `a_mbb_per_100` means A won; a negative value means A lost.
  Consequently, the paper's head-to-head sanity check reads:
    "random_agent (a) vs equity_threshold (b) gives a_mbb_per_100 ~ -154,000",
  i.e. the equity-threshold agent won by ~154,000 mBB/100.
  If you reorder agents in the CLI or elsewhere, the sign flips accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from ..envs.games import GameSpec


@dataclass
class MatchResult:
    a_name: str
    b_name: str
    n_hands: int
    mean_a: float
    std_a: float
    a_mbb_per_100: float
    a_mbb_ci95: float


def play_hand(game: GameSpec, agents: list, seat_of_agent: list[int], rng: np.random.Generator) -> np.ndarray:
    """One hand; agents[i] sits at seat seat_of_agent[i]. Returns per-seat utility."""
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
            state.apply_action(int(actions[idx]))
        else:
            seat = state.current_player()
            agent_idx = seat_of_agent.index(seat)
            action = agents[agent_idx].select_action(state, rng)
            state.apply_action(int(action))
    return np.asarray(state.returns(), dtype=np.float64)


def duel(
    game: GameSpec,
    a,
    b,
    n_hands: int = 2000,
    seed: int = 0,
    alternate_seats: bool = True,
) -> MatchResult:
    rng = np.random.default_rng(seed)
    a_utils: list[float] = []
    for i in range(n_hands):
        seats = [0, 1] if (not alternate_seats or i % 2 == 0) else [1, 0]
        u = play_hand(game, [a, b], seats, rng)
        a_utils.append(float(u[seats[0]]))
    a_arr = np.array(a_utils)
    mean_a = float(a_arr.mean())
    std_a = float(a_arr.std(ddof=1)) if len(a_arr) > 1 else 0.0
    # mBB/100 = (mean utility per hand / big_blind) * 100 hands * 1000 (mBB per BB)
    mbb_per_100 = mean_a / game.big_blind * 100 * 1000
    mbb_ci95 = 1.96 * std_a / np.sqrt(len(a_arr)) / game.big_blind * 100 * 1000
    return MatchResult(
        a_name=getattr(a, "name", "a"),
        b_name=getattr(b, "name", "b"),
        n_hands=n_hands,
        mean_a=mean_a,
        std_a=std_a,
        a_mbb_per_100=float(mbb_per_100),
        a_mbb_ci95=float(mbb_ci95),
    )


def round_robin(
    game: GameSpec,
    agents: Iterable,
    n_hands: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    agents = list(agents)
    rows = []
    for i, (a, b) in enumerate(combinations(agents, 2)):
        r = duel(game, a, b, n_hands=n_hands, seed=seed + i)
        rows.append(vars(r))
    return pd.DataFrame(rows)
