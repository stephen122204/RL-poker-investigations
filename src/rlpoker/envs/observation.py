"""Turn an OpenSpiel state into agent-ready tensors."""
from __future__ import annotations

import numpy as np
import pyspiel


def info_state_vec(state: pyspiel.State, player: int) -> np.ndarray:
    return np.asarray(state.information_state_tensor(player), dtype=np.float32)


def legal_actions_mask(state: pyspiel.State, num_actions: int) -> np.ndarray:
    mask = np.zeros(num_actions, dtype=np.float32)
    for a in state.legal_actions():
        mask[a] = 1.0
    return mask


def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.array(logits, dtype=np.float64) / max(temperature, 1e-8)
    z[mask <= 0] = -1e9
    z = z - np.max(z)
    e = np.exp(z)
    e = e * (mask > 0)
    s = e.sum()
    if s <= 0:
        # Fallback: uniform over legal actions.
        legal = mask > 0
        out = np.zeros_like(z)
        out[legal] = 1.0 / max(legal.sum(), 1)
        return out.astype(np.float32)
    return (e / s).astype(np.float32)


def regret_matching(regrets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classic regret-matching strategy from positive regrets, masked to legal actions."""
    r = np.maximum(np.asarray(regrets, dtype=np.float64), 0.0)
    r = r * (mask > 0)
    s = r.sum()
    if s <= 0:
        legal = mask > 0
        out = np.zeros_like(r)
        out[legal] = 1.0 / max(legal.sum(), 1)
        return out.astype(np.float32)
    return (r / s).astype(np.float32)
