"""Matplotlib + seaborn defaults for publication figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "baseline_mean": "#9E9E9E",
    "ridge": "#4C9AC0",
    "random_forest": "#E76F51",
    "lightgbm": "#F4A261",
    "catboost": "#2A9D8F",
    "mlp": "#8E7CC3",
    "diffusion_cfr": "#D62728",
    "deep_cfr": "#1F77B4",
    "dqn": "#7F7F7F",
    "equity_threshold": "#8C564B",
    "random_agent": "#BDBDBD",
}


def apply_style() -> None:
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    except ImportError:
        pass
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def color_for(name: str) -> str:
    return PALETTE.get(name, "#333333")


def savefig(fig: plt.Figure, path, *, formats: tuple[str, ...] = ("png", "pdf")) -> list:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outs: list = []
    for ext in formats:
        p = path.with_suffix(f".{ext}")
        fig.savefig(p)
        outs.append(p)
    return outs
