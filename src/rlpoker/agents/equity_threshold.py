"""Threshold policy over Monte-Carlo showdown equity.

Encodes the "what does the paper's model actually enable in practice?" baseline:
fold when equity < fold_thr (or check if fold is not legal), call/check when
fold_thr <= equity < call_thr, and raise/bet if legal (else call) when
equity >= call_thr.

Uses `monte_carlo_equity` from `shaping.equity_bridge` so it works uniformly on
Leduc and HULHE. In M5 we swap the estimator for the paper's RF model on HULHE.
"""
from __future__ import annotations

import numpy as np
import pyspiel

from ..shaping.equity_bridge import monte_carlo_equity
from .base import BaseAgent


class EquityThresholdAgent(BaseAgent):
    name = "equity_threshold"

    def __init__(
        self,
        num_actions: int,
        fold_thr: float = 0.40,
        call_thr: float = 0.55,
        n_mc_samples: int = 50,
        seed: int = 0,
    ) -> None:
        self.num_actions = num_actions
        self.fold_thr = fold_thr
        self.call_thr = call_thr
        self.n_mc_samples = n_mc_samples
        self._rng = np.random.default_rng(seed)

    def _classify_actions(self, state: pyspiel.State) -> dict:
        buckets = {"fold": [], "call": [], "raise": []}
        for a in state.legal_actions():
            name = state.action_to_string(a).lower()
            if "fold" in name:
                buckets["fold"].append(a)
            elif "call" in name or "check" in name:
                buckets["call"].append(a)
            elif "raise" in name or "bet" in name:
                buckets["raise"].append(a)
            else:
                buckets["call"].append(a)  # unknown ~ passive default
        return buckets

    def action_probs(self, state: pyspiel.State) -> np.ndarray:
        legal = state.legal_actions()
        probs = np.zeros(self.num_actions, dtype=np.float64)
        if not legal:
            return probs
        buckets = self._classify_actions(state)
        eq = monte_carlo_equity(state, state.current_player(), n_samples=self.n_mc_samples, rng=self._rng)

        # Pick the class based on thresholds; fall back through the chain if unavailable.
        if eq < self.fold_thr and buckets["fold"]:
            probs[buckets["fold"][0]] = 1.0
        elif eq < self.call_thr:
            # call/check preferred; if none, then raise; if none, fold.
            if buckets["call"]:
                probs[buckets["call"][0]] = 1.0
            elif buckets["raise"]:
                probs[buckets["raise"][0]] = 1.0
            elif buckets["fold"]:
                probs[buckets["fold"][0]] = 1.0
        else:
            if buckets["raise"]:
                probs[buckets["raise"][0]] = 1.0
            elif buckets["call"]:
                probs[buckets["call"][0]] = 1.0
            elif buckets["fold"]:
                probs[buckets["fold"][0]] = 1.0

        # Safety: if nothing got selected (unusual), fall back uniform.
        if probs.sum() == 0:
            probs[legal] = 1.0 / len(legal)
        return probs
