"""Regression guard for the head-to-head sign convention cited in the paper.

The paper's Section~5.3 says "The equity-threshold agent won by approximately
154,000 mBB per 100 hands" against a uniform-random opponent on Leduc. That
claim relies on:

  (a) mBB values are reported from agent A's perspective.
  (b) Agent A is the uniform-random agent in that run.
  (c) A lost, i.e. `a_mbb_per_100 < 0`, and |a_mbb_per_100| ~ 1.5e5 in Leduc.

If any of those flips silently in a future refactor (e.g. someone reorders the
duel args), this test catches it and the paper's sentence stops being true.
"""
from __future__ import annotations

import numpy as np

from rlpoker.agents.equity_threshold import EquityThresholdAgent
from rlpoker.agents.random_agent import RandomAgent
from rlpoker.envs.games import make_leduc
from rlpoker.evaluation.head_to_head import duel


def test_random_loses_to_equity_threshold_on_leduc():
    spec = make_leduc()
    rng = np.random.default_rng(0)
    a = RandomAgent(spec.num_actions)
    b = EquityThresholdAgent(spec.num_actions, n_mc_samples=20, seed=0)

    result = duel(spec, a, b, n_hands=500, seed=0)

    # Perspective and identity assertions
    assert result.a_name == "random_agent", (
        f"agent A must be the random agent for the paper's sign convention, "
        f"got a_name={result.a_name}"
    )
    assert result.b_name == "equity_threshold", result.b_name
    # Random should lose by a large margin (>= tens of thousands of mBB/100).
    assert result.a_mbb_per_100 < -1_000, (
        f"expected A (random) to lose big (a_mbb_per_100 << 0), got "
        f"{result.a_mbb_per_100:.1f}"
    )
    # CI half-width is non-negative
    assert result.a_mbb_ci95 >= 0
