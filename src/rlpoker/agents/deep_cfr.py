"""Deep CFR baseline — thin wrapper around OpenSpiel's PyTorch DeepCFRSolver.

The solver holds the advantage / policy networks and CFR memory. Our `Agent`
wrapper exposes `action_probs(state)` for evaluation.
"""
from __future__ import annotations

import numpy as np
import pyspiel

from .base import BaseAgent


class DeepCFRAgent(BaseAgent):
    name = "deep_cfr"

    def __init__(self, solver) -> None:
        """`solver` should be a fitted `open_spiel.python.pytorch.deep_cfr.DeepCFRSolver`."""
        self.solver = solver

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        num_actions = state.get_game().num_distinct_actions()
        probs_dict = self.solver.action_probabilities(state)
        out = np.zeros(num_actions, dtype=np.float64)
        for a, p in probs_dict.items():
            out[int(a)] = float(p)
        legal = state.legal_actions()
        # sanitize
        if not np.all(out >= 0) or out.sum() <= 0:
            out = np.zeros(num_actions, dtype=np.float64)
            for a in legal:
                out[a] = 1.0 / len(legal)
        return out


def build_and_train_deep_cfr(
    game: pyspiel.Game,
    num_iterations: int = 200,
    num_traversals: int = 100,
    advantage_layers=(128, 128),
    policy_layers=(128, 128),
    learning_rate: float = 1e-3,
    memory_capacity: int = 1_000_000,
    device: str = "cpu",
    seed: int = 42,
) -> "DeepCFRAgent":
    from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver

    solver = DeepCFRSolver(
        game=game,
        policy_network_layers=policy_layers,
        advantage_network_layers=advantage_layers,
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        learning_rate=learning_rate,
        memory_capacity=memory_capacity,
        device=device,
        seed=seed,
    )
    _adv_losses, _pol_loss = solver.solve()
    return DeepCFRAgent(solver)
