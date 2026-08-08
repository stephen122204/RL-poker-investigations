"""Common agent interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pyspiel


class Agent(Protocol):
    name: str

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        """Return a probability vector over `num_distinct_actions()`, 0 on illegal actions."""
        ...

    def select_action(self, state: pyspiel.State, rng: np.random.Generator) -> int:
        """Sample a legal action from `action_probs`."""
        ...


class BaseAgent:
    """Default `select_action` implementation on top of `action_probs`."""

    name: str = "base"

    def action_probs(self, state: pyspiel.State) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def select_action(self, state: pyspiel.State, rng: np.random.Generator) -> int:
        probs = self.action_probs(state)
        return int(rng.choice(len(probs), p=probs))
