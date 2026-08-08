"""Diffusion-CFR agent wrapper (HEADLINE)."""
from __future__ import annotations

import numpy as np
import pyspiel

from .base import BaseAgent


class DiffusionCFRAgent(BaseAgent):
    name = "diffusion_cfr"

    def __init__(self, solver) -> None:
        self.solver = solver

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        num_actions = state.get_game().num_distinct_actions()
        probs_dict = self.solver.action_probabilities(state)
        out = np.zeros(num_actions, dtype=np.float64)
        for a, p in probs_dict.items():
            out[int(a)] = float(p)
        legal = state.legal_actions()
        if not np.all(out >= 0) or out.sum() <= 0:
            out = np.zeros(num_actions, dtype=np.float64)
            for a in legal:
                out[a] = 1.0 / len(legal)
        return out
