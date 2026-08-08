"""Figures for the supervised equity benchmark.

Reads cached metrics from `outputs/metrics/` and produces publication-ready plots
in `outputs/figures/`. Every function is idempotent and safe to re-run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIGURES_DIR, METRICS_DIR, STAGES
from .style import PALETTE, apply_style, color_for, savefig


def _load_stage_comparison(metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage in STAGES:
        p = metrics_dir / f"{stage}_results.csv"
        if p.is_file():
            df = pd.read_csv(p)
            if "stage" not in df.columns:
                df.insert(0, "stage", stage)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_results.csv found in {metrics_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_r2_by_stage_model(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    out_stem: str = "fig_equity_r2_by_stage_model",
) -> list[Path]:
    apply_style()
    df = _load_stage_comparison(metrics_dir)
    stages = [s for s in STAGES if s in df["stage"].unique()]
    models = list(df["model"].unique())

    fig, axes = plt.subplots(1, len(stages), figsize=(4.0 * len(stages), 3.6), sharey=True)
    if len(stages) == 1:
        axes = [axes]
    for ax, stage in zip(axes, stages):
        sub = df[df["stage"] == stage]
        sub = sub.set_index("model").reindex(models).reset_index()
        x = np.arange(len(models))
        colors = [color_for(m) for m in models]
        bars = ax.bar(x, sub["test_r2"].fillna(0.0), color=colors, edgecolor="white")
        for b, v in zip(bars, sub["test_r2"].fillna(0.0)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([sub.iloc[i]["display"] if "display" in sub.columns else models[i]
                            for i in range(len(models))], rotation=25, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(stage.capitalize())
        ax.set_ylabel("Held-out test R²")
    fig.suptitle("Held-out test R² by street and model", y=1.02)
    outs = savefig(fig, figures_dir / out_stem)
    plt.close(fig)
    return outs


def plot_mae_by_stage_model(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    out_stem: str = "fig_equity_mae_by_stage_model",
) -> list[Path]:
    apply_style()
    df = _load_stage_comparison(metrics_dir)
    stages = [s for s in STAGES if s in df["stage"].unique()]
    models = list(df["model"].unique())

    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 1.5, 4.0))
    width = 0.16
    x = np.arange(len(stages))
    for i, m in enumerate(models):
        vals = []
        for st in stages:
            sub = df[(df["stage"] == st) & (df["model"] == m)]
            vals.append(float(sub["test_mae"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width,
               color=color_for(m), edgecolor="white", label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in stages])
    ax.set_ylabel("Weighted test MAE")
    ax.set_title("Weighted test MAE by street and model (lower is better)")
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    outs = savefig(fig, figures_dir / out_stem)
    plt.close(fig)
    return outs


def plot_conformal_calibration(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    stage: str = "preflop",
    out_stem: str = "fig_equity_conformal_calibration",
) -> list[Path]:
    apply_style()
    frames: list[pd.DataFrame] = []
    for p in metrics_dir.glob(f"conformal_{stage}_*.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="ideal")
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("nominal_coverage")
        ax.plot(
            sub["nominal_coverage"], sub["empirical_coverage"],
            marker="o", label=model, color=color_for(model),
        )
    ax.set_xlabel("Nominal coverage 1 − α")
    ax.set_ylabel("Empirical coverage on held-out set")
    ax.set_title(f"Split-conformal calibration ({stage})")
    ax.legend()
    outs = savefig(fig, figures_dir / f"{out_stem}_{stage}")
    plt.close(fig)
    return outs


def plot_importance_side_by_side(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    stage: str = "preflop",
    model: str = "random_forest",
    out_stem: str = "fig_equity_importance",
) -> list[Path]:
    apply_style()
    p = metrics_dir / f"importance_{stage}_{model}.csv"
    if not p.is_file():
        return []
    df = pd.read_csv(p)

    metric_cols = [c for c in ("impurity_importance", "permutation_importance_mean", "shap_mean_abs") if c in df.columns]
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, max(3.2, 0.25 * len(df))))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, metric_cols):
        sub = df[["feature", col]].dropna().sort_values(col, ascending=True)
        ax.barh(sub["feature"], sub[col], color=color_for(model), edgecolor="white")
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("importance")
    fig.suptitle(f"Feature attribution — {stage} / {model}", y=1.02)
    outs = savefig(fig, figures_dir / f"{out_stem}_{stage}_{model}")
    plt.close(fig)
    return outs


def plot_error_by_group(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
    stage: str = "preflop",
    model: str = "random_forest",
    out_stem: str = "fig_equity_error_by_group",
) -> list[Path]:
    apply_style()
    p = metrics_dir / f"error_analysis_{stage}_{model}.csv"
    if not p.is_file():
        return []
    df = pd.read_csv(p)
    groups = list(df["group_feature"].unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(6.5, 1.4 * len(groups) + 1.0))
    if len(groups) == 1:
        axes = [axes]
    for ax, g in zip(axes, groups):
        sub = df[df["group_feature"] == g].sort_values("weighted_mean_abs_error", ascending=True)
        ax.barh(sub["group_value"].astype(str), sub["weighted_mean_abs_error"],
                color=color_for(model), edgecolor="white")
        ax.set_title(g)
        ax.set_xlabel("weighted MAE")
    fig.suptitle(f"Weighted absolute error by feature group ({stage}, {model})", y=1.005)
    outs = savefig(fig, figures_dir / f"{out_stem}_{stage}_{model}")
    plt.close(fig)
    return outs


def make_all_equity_figures(
    metrics_dir: Path = METRICS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    outputs: list[Path] = []
    outputs += plot_r2_by_stage_model(metrics_dir, figures_dir)
    outputs += plot_mae_by_stage_model(metrics_dir, figures_dir)
    for stage in STAGES:
        outputs += plot_conformal_calibration(metrics_dir, figures_dir, stage=stage)
        for m in ("random_forest", "lightgbm", "ridge"):
            outputs += plot_importance_side_by_side(metrics_dir, figures_dir, stage=stage, model=m)
            outputs += plot_error_by_group(metrics_dir, figures_dir, stage=stage, model=m)
    return outputs
