"""OpenSpiel game factories.

Three games are supported:
  - `leduc`  — 2-player Leduc poker (6 cards). Small enough for exact exploitability.
  - `hulhe`  — 2-player Heads-Up Limit Hold'em (universal_poker, `betting=limit`).
  - `kuhn`   — 2-player Kuhn poker (3 cards). Sanity smoke tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyspiel


HULHE_PARAMS = {
    "betting": "limit",
    "numPlayers": 2,
    "numRounds": 4,
    "blind": "50 100",
    "raiseSize": "100 100 200 200",
    "firstPlayer": "2 1 1 1",
    "maxRaises": "3 4 4 4",
    "numSuits": 4,
    "numRanks": 13,
    "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
    "stack": "20000 20000",
}


@dataclass(frozen=True)
class GameSpec:
    name: str
    game: pyspiel.Game
    num_players: int
    num_actions: int
    info_state_size: int
    big_blind: int  # in chips, used for mBB/100 conversion

    def new_initial_state(self) -> pyspiel.State:
        return self.game.new_initial_state()


def make_leduc() -> GameSpec:
    g = pyspiel.load_game("leduc_poker")
    return GameSpec(
        name="leduc",
        game=g,
        num_players=g.num_players(),
        num_actions=g.num_distinct_actions(),
        info_state_size=g.information_state_tensor_size(),
        big_blind=1,  # Leduc "big blind" is 1 chip
    )


def make_kuhn() -> GameSpec:
    g = pyspiel.load_game("kuhn_poker")
    return GameSpec(
        name="kuhn",
        game=g,
        num_players=g.num_players(),
        num_actions=g.num_distinct_actions(),
        info_state_size=g.information_state_tensor_size(),
        big_blind=1,
    )


def make_hulhe() -> GameSpec:
    g = pyspiel.load_game("universal_poker", HULHE_PARAMS)
    return GameSpec(
        name="hulhe",
        game=g,
        num_players=g.num_players(),
        num_actions=g.num_distinct_actions(),
        info_state_size=g.information_state_tensor_size(),
        big_blind=100,  # from `blind: "50 100"`
    )


def make_game(name: str) -> GameSpec:
    name = name.lower()
    if name in ("leduc", "leduc_poker"):
        return make_leduc()
    if name in ("hulhe", "hulh", "hold_em", "universal_poker"):
        return make_hulhe()
    if name in ("kuhn", "kuhn_poker"):
        return make_kuhn()
    raise ValueError(f"Unknown game '{name}'. Use 'leduc', 'hulhe', or 'kuhn'.")
