"""Uniform-random legal-action agent."""
from __future__ import annotations

import numpy as np
import pyspiel

from .base import BaseAgent


class RandomAgent(BaseAgent):
    name = "random_agent"

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        probs = np.zeros(self.num_actions, dtype=np.float64)
        legal = state.legal_actions()
        if not legal:
            return probs
        probs[legal] = 1.0 / len(legal)
        return probs
