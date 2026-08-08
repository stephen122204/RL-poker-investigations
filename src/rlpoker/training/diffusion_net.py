"""Conditional denoising diffusion network for CFR advantages.

Given info-state `s` (dim `d_s`) and target advantage vector `v` in R^A, learn
p(v | s) as a DDPM epsilon-predictor:

  v_t = sqrt(alpha_bar_t) v_0 + sqrt(1 - alpha_bar_t) epsilon,  epsilon ~ N(0, I)
  epsilon_theta(v_t, t, s) approximates epsilon

Training loss: MSE(epsilon_theta, epsilon), scaled by the CFR-style
sqrt(iteration) weight from Deep CFR (see `training_loss`).

Inference (see `sample`): reverse denoising, K DDPM steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard positional embedding for the diffusion step index."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


@dataclass
class DiffusionSchedule:
    n_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def betas(self, device) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.n_steps, device=device)

    def alphas_cumprod(self, device) -> torch.Tensor:
        return torch.cumprod(1.0 - self.betas(device), dim=0)


class DiffusionAdvantageNet(nn.Module):
    """epsilon-predictor MLP mapping (s, v_t, t) to predicted noise.

    Compatible with OpenSpiel's DeepCFRSolver internals: exposes a `.reset()`
    method and, when called as `net(state_tensor)`, returns a mean advantage
    vector via a `K`-sample reverse-denoise (see `predict_mean`)."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden=(128, 128),
        time_embed_dim: int = 64,
        n_diffusion_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        inference_samples: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.schedule = DiffusionSchedule(n_diffusion_steps, beta_start, beta_end)
        self.time_embed_dim = time_embed_dim
        self.inference_samples = inference_samples

        in_dim = state_dim + num_actions + time_embed_dim
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers += [nn.LayerNorm(in_dim), nn.Linear(in_dim, num_actions)]
        self.eps_net = nn.Sequential(*layers)

    # ---- epsilon-predictor forward -------------------------------------------

    def eps(self, state: torch.Tensor, v_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = _sinusoidal_time_embedding(t, self.time_embed_dim)
        x = torch.cat([state, v_t, te], dim=-1)
        return self.eps_net(x)

    # ---- Training-loss helper ----------------------------------------------

    def training_loss(
        self,
        state: torch.Tensor,
        v_target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # LOSS-WEIGHT CONTRACT (report-paper/template.tex, section 4.3):
        #   `weight` is applied OUTSIDE the squared error norm. The caller passes
        #   weight = sqrt(k_i) (k_i = CFR iteration the target was stored), so the
        #   effective per-example weight is sqrt(k_i), SUBLINEAR. OpenSpiel's
        #   DeepCFRSolver puts sqrt(k_i) INSIDE the MSE instead, making it LINEAR
        #   in k_i. Intentionally different; to match Deep CFR, replace
        #   `weight * per_sample` with `(weight ** 2) * per_sample` or pass k_i
        #   (not sqrt) from the solver.
        device = state.device
        n = state.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (n,), device=device)
        alpha_bar = self.schedule.alphas_cumprod(device)[t].unsqueeze(-1)
        eps = torch.randn_like(v_target)
        v_t = torch.sqrt(alpha_bar) * v_target + torch.sqrt(1.0 - alpha_bar) * eps
        pred = self.eps(state, v_t, t)
        per_sample = ((pred - eps) ** 2).mean(dim=-1)
        return (weight * per_sample).mean()

    # ---- Inference: reverse denoise -----------------------------------------

    @torch.inference_mode()
    def sample(self, state: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Reverse DDPM sampling; return an (n, num_actions) advantage tensor.

        Averages `num_samples` independent runs to reduce variance.
        """
        device = state.device
        K = num_samples if num_samples is not None else self.inference_samples
        n = state.shape[0]
        betas = self.schedule.betas(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        T = self.schedule.n_steps
        agg = torch.zeros((n, self.num_actions), device=device)
        for _ in range(K):
            v = torch.randn((n, self.num_actions), device=device)
            for i in reversed(range(T)):
                t_idx = torch.full((n,), i, device=device, dtype=torch.long)
                eps_pred = self.eps(state, v, t_idx)
                alpha_t = alphas[i]
                alpha_bar_t = alphas_bar[i]
                # DDPM posterior mean
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
                mean = coef1 * (v - coef2 * eps_pred)
                if i > 0:
                    beta_t = betas[i]
                    noise = torch.randn_like(v)
                    v = mean + torch.sqrt(beta_t) * noise
                else:
                    v = mean
            agg = agg + v
        return agg / K

    # ---- OpenSpiel compatibility -------------------------------------------

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Called by OpenSpiel's Deep CFR internals: expects mean advantage vector."""
        return self.sample(state)

    def reset(self):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.apply(fn=weight_reset)
