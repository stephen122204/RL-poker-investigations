"""Confirm the Diffusion-CFR advantage head is actually being invoked.

If DiffusionCFRSolver silently fell back to the parent Deep CFR MLP head, its
convergence curve would just be a copy of Deep CFR's and the paper's novel
contribution would be vacuous. This test verifies:

1. Every ``_advantage_networks[p]`` is a ``DiffusionAdvantageNet`` (not the
   parent's ``MLP``).
2. A forward call triggers the reverse-denoise sampling loop, not the parent's
   MLP forward.
3. The DDPM training loss (``training_loss``) is monotonically driven by the
   advantage-network parameters (i.e. gradient flows).
"""
from __future__ import annotations

import numpy as np
import torch

from rlpoker.envs.games import make_leduc
from rlpoker.training.diffusion_cfr_solver import DiffusionCFRSolver
from rlpoker.training.diffusion_net import DiffusionAdvantageNet


def test_advantage_head_is_diffusion_not_mlp():
    spec = make_leduc()
    solver = DiffusionCFRSolver(
        game=spec.game,
        policy_network_layers=(32, 32),
        advantage_hidden=(32, 32),
        n_diffusion_steps=4,
        inference_samples=1,
        num_iterations=1,
        num_traversals=2,
    )
    for p, net in enumerate(solver._advantage_networks):
        assert isinstance(net, DiffusionAdvantageNet), (
            f"Player {p} advantage network is {type(net).__name__}, not DiffusionAdvantageNet"
        )


def test_forward_call_triggers_diffusion_sampling():
    spec = make_leduc()
    net = DiffusionAdvantageNet(
        state_dim=spec.info_state_size,
        num_actions=spec.num_actions,
        hidden=(16, 16),
        n_diffusion_steps=3,
        inference_samples=1,
        seed=0,
    )

    calls = {"eps": 0}
    original_eps = net.eps
    def counting_eps(state, v_t, t):
        calls["eps"] += 1
        return original_eps(state, v_t, t)
    net.eps = counting_eps

    x = torch.zeros(2, spec.info_state_size)
    _ = net(x)
    assert calls["eps"] == 3, (
        f"forward() should call eps() T={net.schedule.n_steps} times per sample "
        f"(K=1 x T=3 = 3), got {calls['eps']}"
    )


def test_ddpm_loss_backprops_through_advantage_net():
    spec = make_leduc()
    net = DiffusionAdvantageNet(
        state_dim=spec.info_state_size, num_actions=spec.num_actions,
        hidden=(16, 16), n_diffusion_steps=3, inference_samples=1, seed=1,
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    state = torch.randn(32, spec.info_state_size)
    target = torch.randn(32, spec.num_actions) * 0.1
    weight = torch.ones(32)

    initial_params = [p.detach().clone() for p in net.parameters()]
    for _ in range(20):
        opt.zero_grad()
        loss = net.training_loss(state, target, weight)
        loss.backward()
        opt.step()
    final_params = [p.detach().clone() for p in net.parameters()]

    changed = sum(
        (not torch.allclose(a, b)) for a, b in zip(initial_params, final_params)
    )
    assert changed == len(initial_params), (
        f"only {changed}/{len(initial_params)} parameter tensors moved under DDPM loss - "
        "gradient may not be flowing through the diffusion net"
    )


def test_exploitability_uses_openspiel_best_response():
    """Regression check: our exploitability wrapper matches OpenSpiel's built-in
    exploitability of a uniform-random policy to numerical precision, which
    means we are using the same tabular best-response calculation."""
    from open_spiel.python.algorithms import exploitability as oss_exp
    from open_spiel.python.policy import UniformRandomPolicy

    from rlpoker.agents.random_agent import RandomAgent
    from rlpoker.evaluation.exploitability import compute_exploitability

    spec = make_leduc()
    ours = compute_exploitability(spec.game, RandomAgent(spec.num_actions))
    theirs = float(oss_exp.exploitability(spec.game, UniformRandomPolicy(spec.game)))
    assert abs(ours - theirs) < 1e-6, (
        f"our exploitability {ours} != OpenSpiel's {theirs}; the best-response "
        "computation is not the same underneath"
    )
