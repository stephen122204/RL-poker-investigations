"""Diffusion-CFR: OpenSpiel Deep CFR with a diffusion advantage head.

Subclasses `open_spiel.python.pytorch.deep_cfr.DeepCFRSolver` and swaps:
  - The advantage networks (`_advantage_networks[p]`) for `DiffusionAdvantageNet`
    instances that model p(advantage_vec | info_state) via epsilon-prediction.
  - `_learn_advantage_network` to use the DDPM loss.
  - `_reinitialize_advantage_network` to hand out a fresh diffusion net.

Everything else — CFR game-tree traversal, strategy memory, average policy
network, exploitability — is inherited unchanged. This keeps convergence
guarantees intact: only the advantage function class differs.
"""
from __future__ import annotations

import numpy as np
import torch

from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver

from .diffusion_net import DiffusionAdvantageNet


class DiffusionCFRSolver(DeepCFRSolver):
    def __init__(
        self,
        game,
        *,
        policy_network_layers=(128, 128),
        advantage_hidden=(128, 128),
        n_diffusion_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_embed_dim: int = 64,
        inference_samples: int = 3,
        num_iterations: int = 100,
        num_traversals: int = 40,
        learning_rate: float = 1e-3,
        batch_size_advantage: int | None = None,
        batch_size_strategy: int | None = None,
        memory_capacity: int = 1_000_000,
        policy_network_train_steps: int = 1,
        advantage_network_train_steps: int = 1,
        reinitialize_advantage_networks: bool = True,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self._diffusion_kwargs = dict(
            hidden=advantage_hidden,
            time_embed_dim=time_embed_dim,
            n_diffusion_steps=n_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            inference_samples=inference_samples,
            seed=seed,
        )
        super().__init__(
            game,
            policy_network_layers=policy_network_layers,
            advantage_network_layers=advantage_hidden,
            num_iterations=num_iterations,
            num_traversals=num_traversals,
            learning_rate=learning_rate,
            batch_size_advantage=batch_size_advantage,
            batch_size_strategy=batch_size_strategy,
            memory_capacity=memory_capacity,
            policy_network_train_steps=policy_network_train_steps,
            advantage_network_train_steps=advantage_network_train_steps,
            reinitialize_advantage_networks=reinitialize_advantage_networks,
            device=device,
            seed=seed,
        )
        # The parent init created MLP advantage networks; replace them.
        state_dim = game.information_state_tensor_shape()[0]
        num_actions = game.num_distinct_actions()
        self._advantage_networks = [
            DiffusionAdvantageNet(state_dim, num_actions, **self._diffusion_kwargs).to(self._device)
            for _ in range(self._num_players)
        ]
        self._optimizer_advantages = [None] * self._num_players
        for p in range(self._num_players):
            self._reinitialize_advantage_network(p)

    # ---- override advantage learning to use DDPM loss ----------------------

    def _learn_advantage_network(self, player):
        for _ in range(self._advantage_network_train_steps):
            if self._batch_size_advantage:
                if self._batch_size_advantage > len(self._advantage_memories[player]):
                    return None
                samples = self._advantage_memories[player].sample(self._batch_size_advantage)
            else:
                self._advantage_memories[player].shuffle()
                samples = self._advantage_memories[player].experience
            if len(samples.info_state) == 0:
                return None

            self._optimizer_advantages[player].zero_grad()
            # sqrt(k) weight, applied outside the squared norm: effective weight
            # is sqrt(k) (sublinear), intentionally unlike Deep CFR's linear k
            # (see the LOSS-WEIGHT CONTRACT in diffusion_net.py)
            iters = torch.as_tensor(samples.iteration, device=self._device, dtype=torch.float32).sqrt().squeeze(-1)
            state = torch.as_tensor(samples.info_state, device=self._device, dtype=torch.float32)
            adv = torch.as_tensor(samples.advantage, device=self._device, dtype=torch.float32)
            loss = self._advantage_networks[player].training_loss(state, adv, iters)
            loss.backward()
            self._optimizer_advantages[player].step()
        return float(loss.detach().cpu().item())

    def _reinitialize_advantage_network(self, player):
        state_dim = self._game.information_state_tensor_shape()[0]
        num_actions = self._game.num_distinct_actions()
        self._advantage_networks[player] = DiffusionAdvantageNet(
            state_dim, num_actions, **self._diffusion_kwargs
        ).to(self._device)
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate
        )
