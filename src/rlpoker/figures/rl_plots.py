"""Figures for the RL side of the project.

  - `plot_exploitability_curves`: exploitability vs. CFR iteration for
    Deep CFR, Diffusion-CFR, and any other cached curves.
  - `plot_head_to_head_heatmap`: mBB/100 tournament matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIGURES_DIR, METRICS_DIR
from .style import PALETTE, apply_style, color_for, savefig


def _load_curves(metrics_dir: Path, game: str) -> dict[str, pd.DataFrame]:
    """Load per-agent curves; if multiple seed files exist, aggregate to
    mean/std across seeds by iteration.

    Recognizes both ``exploitability_curve_{agent}_{game}.json`` (single seed,
    treated as seed 42) and ``exploitability_curve_{agent}_{game}_seed{N}.json``.
    """
    grouped: dict[str, list[pd.DataFrame]] = {}
    # Match both `exploitability_curve_{agent}_{game}.json` (seed 42) and
    # `exploitability_curve_{agent}_{game}_seed{N}.json` (extra seeds).
    for p in metrics_dir.glob("exploitability_curve_*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not data:
            continue
        df = pd.DataFrame(data)
        # Filter to the requested game via the JSON payload, not the filename.
        if "game" in df.columns and df["game"].iloc[0] != game:
            continue
        if "agent" in df.columns:
            agent = df["agent"].iloc[0]
        else:
            continue
        grouped.setdefault(agent, []).append(df[["iteration", "exploitability"]].copy())
    # Also pick up the seedless file (seed=42 default) that may match multiple times.
    curves: dict[str, pd.DataFrame] = {}
    for agent, dfs in grouped.items():
        merged = None
        for i, d in enumerate(dfs):
            d = d.rename(columns={"exploitability": f"expl_{i}"})
            merged = d if merged is None else merged.merge(d, on="iteration", how="outer")
        merged = merged.sort_values("iteration").reset_index(drop=True)
        expl_cols = [c for c in merged.columns if c.startswith("expl_")]
        arr = merged[expl_cols].to_numpy(dtype=float)
        merged["mean"] = np.nanmean(arr, axis=1)
        merged["std"] = np.nanstd(arr, axis=1, ddof=0)
        merged["n_seeds"] = np.sum(~np.isnan(arr), axis=1)
        curves[agent] = merged
    return curves


PALETTE.update({"tabular_cfr": "#2ECC71"})


def plot_exploitability_curves(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    game: str = "leduc",
    out_stem: str = "fig_rl_exploitability",
) -> list[Path]:
    apply_style()
    curves = _load_curves(metrics_dir, game)
    if not curves:
        return []
    order = ["tabular_cfr", "deep_cfr", "diffusion_cfr"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for agent in order:
        if agent not in curves:
            continue
        df = curves[agent]
        c = color_for(agent)
        n = int(df["n_seeds"].max())
        label = f"{agent} (n={n} seeds)" if n > 1 else agent
        ax.plot(df["iteration"], df["mean"], marker="o", label=label, color=c, lw=1.6)
        if n > 1:
            ax.fill_between(
                df["iteration"], df["mean"] - df["std"], df["mean"] + df["std"],
                color=c, alpha=0.2, linewidth=0,
            )
    ax.set_xlabel("CFR iteration")
    ax.set_ylabel("Exploitability (game-utility units)")
    ax.set_title(f"Exploitability convergence in {game.capitalize()}")
    ax.legend()
    ax.set_yscale("log")
    outs = savefig(fig, figures_dir / f"{out_stem}_{game}")
    plt.close(fig)
    return outs


def plot_exploitability_two_panel(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    games: tuple[str, ...] = ("kuhn", "leduc"),
    out_stem: str = "fig_rl_exploitability_two_panel",
) -> list[Path]:
    apply_style()
    fig, axes = plt.subplots(1, len(games), figsize=(5.5 * len(games), 4.2), sharey=False)
    if len(games) == 1:
        axes = [axes]
    order = ["tabular_cfr", "deep_cfr", "diffusion_cfr"]
    plotted = False
    for ax, game in zip(axes, games):
        curves = _load_curves(metrics_dir, game)
        if not curves:
            ax.set_visible(False)
            continue
        plotted = True
        for agent in order:
            if agent not in curves:
                continue
            df = curves[agent]
            c = color_for(agent)
            n = int(df["n_seeds"].max())
            label = f"{agent} (n={n})" if n > 1 else agent
            ax.plot(df["iteration"], df["mean"], marker="o", label=label, color=c, lw=1.6)
            if n > 1:
                ax.fill_between(
                    df["iteration"], df["mean"] - df["std"], df["mean"] + df["std"],
                    color=c, alpha=0.2, linewidth=0,
                )
        ax.set_xlabel("CFR iteration")
        ax.set_ylabel("Exploitability")
        ax.set_title(game.capitalize())
        ax.set_yscale("log")
        ax.legend(fontsize=9)
    if not plotted:
        plt.close(fig)
        return []
    fig.suptitle("Exploitability convergence: tabular CFR vs. neural CFR variants", y=1.02)
    outs = savefig(fig, figures_dir / out_stem)
    plt.close(fig)
    return outs


def plot_head_to_head_heatmap(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    game: str = "leduc",
    out_stem: str = "fig_rl_head_to_head",
) -> list[Path]:
    apply_style()
    p = metrics_dir / f"head_to_head_{game}.csv"
    if not p.is_file():
        return []
    df = pd.read_csv(p)
    agents = sorted(set(df["a_name"]) | set(df["b_name"]))
    idx = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    mat = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = idx[row["a_name"]]
        j = idx[row["b_name"]]
        mat[i, j] = row["a_mbb_per_100"]
        mat[j, i] = -row["a_mbb_per_100"]

    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(agents)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        color="black" if abs(mat[i, j]) < vmax * 0.5 else "white", fontsize=9)
    ax.set_title(f"Head-to-head mBB/100 in {game.capitalize()} (row perspective)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    outs = savefig(fig, figures_dir / f"{out_stem}_{game}")
    plt.close(fig)
    return outs


def make_all_rl_figures(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    games: tuple[str, ...] = ("kuhn", "leduc", "hulhe"),
) -> list[Path]:
    outputs: list[Path] = []
    for g in games:
        outputs += plot_exploitability_curves(metrics_dir, figures_dir, game=g)
        outputs += plot_head_to_head_heatmap(metrics_dir, figures_dir, game=g)
    outputs += plot_exploitability_two_panel(metrics_dir, figures_dir, games=("kuhn", "leduc"))
    return outputs
