"""OpenSpiel wrapper shape + Random-agent legality smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from rlpoker.agents.random_agent import RandomAgent
from rlpoker.envs.games import make_hulhe, make_kuhn, make_leduc


@pytest.mark.parametrize("factory,name,n_actions", [(make_leduc, "leduc", 3), (make_kuhn, "kuhn", 2), (make_hulhe, "hulhe", 3)])
def test_game_shape(factory, name, n_actions):
    g = factory()
    assert g.name == name
    assert g.num_players == 2
    assert g.num_actions == n_actions
    s = g.new_initial_state()
    assert s is not None


def _play_one_hand(spec, agent, seed=0):
    rng = np.random.default_rng(seed)
    state = spec.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outs, probs = zip(*state.chance_outcomes())
            state.apply_action(int(outs[rng.choice(len(outs), p=np.asarray(probs, dtype=np.float64))]))
        else:
            a = agent.select_action(state, rng)
            legal = state.legal_actions()
            assert a in legal, f"illegal action {a}; legal={legal}"
            state.apply_action(int(a))
    return state.returns()


def test_random_agent_plays_leduc():
    spec = make_leduc()
    agent = RandomAgent(spec.num_actions)
    ret = _play_one_hand(spec, agent, seed=0)
    assert len(ret) == 2


def test_random_agent_plays_hulhe():
    spec = make_hulhe()
    agent = RandomAgent(spec.num_actions)
    ret = _play_one_hand(spec, agent, seed=0)
    assert len(ret) == 2


def test_leduc_exploitability_of_uniform_matches_openspiel():
    """Uniform random exploitability of Leduc is a known-ish number ~ 2.37."""
    from open_spiel.python.algorithms import exploitability as oss_exp
    from open_spiel.python.policy import UniformRandomPolicy

    from rlpoker.evaluation.exploitability import compute_exploitability

    spec = make_leduc()
    ours = compute_exploitability(spec.game, RandomAgent(spec.num_actions))
    theirs = oss_exp.exploitability(spec.game, UniformRandomPolicy(spec.game))
    assert abs(ours - theirs) < 1e-6, (ours, theirs)
