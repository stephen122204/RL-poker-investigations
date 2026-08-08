"""Multi-model feature attribution: impurity, permutation, and SHAP.

- Impurity importances only exist for tree ensembles (RF, LightGBM, CatBoost).
- Permutation importance is model-agnostic.
- SHAP uses TreeExplainer for tree models and KernelExplainer as a fallback
  (kernel is slow — subsample X to keep runtimes reasonable).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from .models import ModelSpec, underlying_estimator


def impurity_importance_df(estimator: BaseEstimator, feature_names: list[str]) -> Optional[pd.DataFrame]:
    inner = underlying_estimator(estimator)
    imp = getattr(inner, "feature_importances_", None)
    if imp is None:
        return None
    return (
        pd.DataFrame({"feature": feature_names, "impurity_importance": np.asarray(imp)})
        .sort_values("impurity_importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_df(
    estimator: BaseEstimator,
    X_test,
    y_test,
    feature_names: list[str],
    random_state: int,
    n_repeats: int = 10,
) -> pd.DataFrame:
    r = permutation_importance(
        estimator, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance_mean": r.importances_mean,
            "permutation_importance_std": r.importances_std,
        }
    )


def shap_importance_df(
    estimator: BaseEstimator,
    spec: ModelSpec,
    X_bg,
    X_eval,
    feature_names: list[str],
    max_bg: int = 200,
    max_eval: int = 500,
    random_state: int = 0,
) -> Optional[pd.DataFrame]:
    """Global SHAP importance: mean(|shap value|) per feature.

    Uses TreeExplainer for RF / LightGBM / CatBoost, LinearExplainer for Ridge,
    and KernelExplainer for MLP (slow — capped by max_eval).
    """
    try:
        import shap
    except ImportError:
        return None
    inner = underlying_estimator(estimator)
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X_bg), size=min(max_bg, len(X_bg)), replace=False)
    ev_idx = rng.choice(len(X_eval), size=min(max_eval, len(X_eval)), replace=False)
    X_bg_s = _iloc(X_bg, bg_idx)
    X_ev_s = _iloc(X_eval, ev_idx)

    try:
        if spec.kind in ("tree_ensemble", "gbdt"):
            explainer = shap.TreeExplainer(inner)
            values = explainer.shap_values(X_ev_s)
        elif spec.kind == "linear":
            # Ridge lives inside a Pipeline; hand SHAP the scaler-transformed X.
            scaler = estimator.named_steps["scaler"] if hasattr(estimator, "named_steps") else None
            X_bg_t = scaler.transform(X_bg_s) if scaler is not None else X_bg_s
            X_ev_t = scaler.transform(X_ev_s) if scaler is not None else X_ev_s
            explainer = shap.LinearExplainer(inner, X_bg_t)
            values = explainer.shap_values(X_ev_t)
        else:
            # MLP + baseline via kernel (small subsample).
            f = estimator.predict
            explainer = shap.KernelExplainer(f, X_bg_s)
            values = explainer.shap_values(X_ev_s, nsamples=100)
    except Exception:
        return None

    values = np.asarray(values)
    if values.ndim == 3:  # multi-output edge case
        values = values.mean(axis=0)
    mean_abs = np.mean(np.abs(values), axis=0)
    return pd.DataFrame({"feature": feature_names, "shap_mean_abs": mean_abs}).sort_values(
        "shap_mean_abs", ascending=False
    ).reset_index(drop=True)


def build_full_importance_table(
    estimator: BaseEstimator,
    spec: ModelSpec,
    X_train,
    X_test,
    y_test,
    feature_names: list[str],
    random_state: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    imp = impurity_importance_df(estimator, feature_names)
    if imp is not None:
        parts.append(imp)
    perm = permutation_importance_df(
        estimator, X_test, y_test, feature_names, random_state=random_state, n_repeats=10
    )
    parts.append(perm)
    shap_df = shap_importance_df(
        estimator, spec, X_train, X_test, feature_names, random_state=random_state
    )
    if shap_df is not None:
        parts.append(shap_df)
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="feature", how="outer")
    # Choose a sort key: impurity if present, else SHAP, else permutation.
    sort_key = "impurity_importance" if "impurity_importance" in out.columns else (
        "shap_mean_abs" if "shap_mean_abs" in out.columns else "permutation_importance_mean"
    )
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


def save_importance_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _iloc(X, idx):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return X[idx]
