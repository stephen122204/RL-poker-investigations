"""Warm-start the Diffusion-CFR advantage network from the equity-threshold rule.

Sample many info-states by playing out random hands, compute the target
advantage vector implied by the equity-threshold policy (fold/call/raise as one-hot),
and behavioral-clone the diffusion advantage network on those (info_state, target)
pairs. The CFR loop then refines from this initialization.

Kept lightweight so the warm-start ablation is a single flag on the trainer.
"""
from __future__ import annotations

import numpy as np
import pyspiel
import torch

from ..agents.equity_threshold import EquityThresholdAgent


def collect_warm_start_data(
    game: pyspiel.Game,
    num_actions: int,
    n_states: int = 4000,
    fold_thr: float = 0.4,
    call_thr: float = 0.55,
    n_mc_samples: int = 20,
    seed: int = 0,
    advantage_magnitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (info_state_tensor, target_advantage_vec) arrays.

    Target advantages: place `advantage_magnitude` on the chosen action, 0 elsewhere.
    Regret matching over these deterministic advantages recovers the threshold policy.
    """
    rng = np.random.default_rng(seed)
    agent = EquityThresholdAgent(num_actions, fold_thr, call_thr, n_mc_samples, seed=seed)
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    while len(states) < n_states:
        s = game.new_initial_state()
        while not s.is_terminal():
            if s.is_chance_node():
                outs, probs = zip(*s.chance_outcomes())
                s.apply_action(int(outs[rng.choice(len(outs), p=np.asarray(probs, dtype=np.float64))]))
                continue
            player = s.current_player()
            probs = agent.action_probs(s)
            action = int(np.argmax(probs))
            info = np.asarray(s.information_state_tensor(player), dtype=np.float32)
            tgt = np.zeros(num_actions, dtype=np.float32)
            tgt[action] = advantage_magnitude
            states.append(info)
            targets.append(tgt)
            if len(states) >= n_states:
                break
            s.apply_action(action)
    return np.stack(states), np.stack(targets)


def behavioral_clone_diffusion(
    net,  # DiffusionAdvantageNet
    info_states: np.ndarray,
    targets: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Cross-entropy-free BC via the DDPM loss with iteration=1 sample weights."""
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(info_states)
    losses: list[float] = []
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            state = torch.as_tensor(info_states[b], device=device, dtype=torch.float32)
            adv = torch.as_tensor(targets[b], device=device, dtype=torch.float32)
            weight = torch.ones(len(b), device=device, dtype=torch.float32)
            opt.zero_grad()
            loss = net.training_loss(state, adv, weight)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses
