"""Exact tabular CFR baseline (OpenSpiel `cfr.CFRSolver`) — converges to Nash.

On Leduc: exploitability falls to ~0.1 within a few hundred iterations. Gives us a
"ground truth" convergence curve for the figure; the neural methods (Deep CFR,
Diffusion-CFR) are then judged by how well they approximate this at similar
compute budgets rather than absolute Nash convergence.
"""
from __future__ import annotations

import numpy as np
import pyspiel

from .base import BaseAgent


class TabularCFRAgent(BaseAgent):
    name = "tabular_cfr"

    def __init__(self, solver, game: pyspiel.Game) -> None:
        self.solver = solver
        self.policy = solver.average_policy()
        self.num_actions = game.num_distinct_actions()

    def refresh(self) -> None:
        self.policy = self.solver.average_policy()

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        probs_dict = self.policy.action_probabilities(state)
        out = np.zeros(self.num_actions, dtype=np.float64)
        for a, p in probs_dict.items():
            out[int(a)] = float(p)
        legal = state.legal_actions()
        if out.sum() <= 0:
            out = np.zeros(self.num_actions, dtype=np.float64)
            for a in legal:
                out[a] = 1.0 / len(legal)
        return out
