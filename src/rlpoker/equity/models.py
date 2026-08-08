"""Registry of equity regression models.

Each model is a fresh scikit-learn compatible estimator. Some models are wrapped
in a Pipeline that first standard-scales the features (Ridge, MLP); tree-based
models see raw features. All models support `sample_weight` at fit time except
MLPRegressor, which is trained unweighted (documented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    CATBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    MLP_PARAMS,
    RF_PARAMS,
    RIDGE_PARAMS,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display: str
    factory: Callable[[int], BaseEstimator]
    supports_sample_weight: bool
    scaled: bool
    kind: str  # "baseline" | "linear" | "tree_ensemble" | "gbdt" | "mlp"


def _make_baseline(_random_state: int) -> BaseEstimator:
    return DummyRegressor(strategy="mean")


def _make_ridge(_random_state: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", Ridge(**RIDGE_PARAMS)),
        ]
    )


def _make_random_forest(random_state: int) -> BaseEstimator:
    return RandomForestRegressor(random_state=random_state, **RF_PARAMS)


def _make_lightgbm(random_state: int) -> BaseEstimator:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError("pip install lightgbm") from e
    return LGBMRegressor(random_state=random_state, **LIGHTGBM_PARAMS)


def _make_catboost(random_state: int) -> BaseEstimator:
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError("pip install catboost") from e
    return CatBoostRegressor(random_state=random_state, **CATBOOST_PARAMS)


def _make_mlp(random_state: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", MLPRegressor(random_state=random_state, **MLP_PARAMS)),
        ]
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "baseline_mean": ModelSpec(
        name="baseline_mean",
        display="Mean baseline",
        factory=_make_baseline,
        supports_sample_weight=True,
        scaled=False,
        kind="baseline",
    ),
    "ridge": ModelSpec(
        name="ridge",
        display="Ridge",
        factory=_make_ridge,
        supports_sample_weight=True,
        scaled=True,
        kind="linear",
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display="Random Forest",
        factory=_make_random_forest,
        supports_sample_weight=True,
        scaled=False,
        kind="tree_ensemble",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        display="LightGBM",
        factory=_make_lightgbm,
        supports_sample_weight=True,
        scaled=False,
        kind="gbdt",
    ),
    "catboost": ModelSpec(
        name="catboost",
        display="CatBoost",
        factory=_make_catboost,
        supports_sample_weight=True,
        scaled=False,
        kind="gbdt",
    ),
    "mlp": ModelSpec(
        name="mlp",
        display="MLP",
        factory=_make_mlp,
        supports_sample_weight=False,  # sklearn MLPRegressor has no sample_weight
        scaled=True,
        kind="mlp",
    ),
}


PAPER_MODELS = ("baseline_mean", "ridge", "random_forest")
UPGRADED_MODELS = tuple(MODEL_REGISTRY)


def make_estimator(name: str, random_state: int) -> Tuple[BaseEstimator, ModelSpec]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    return spec.factory(random_state), spec


def fit_with_optional_weights(estimator: BaseEstimator, spec: ModelSpec, X, y, sample_weight):
    """Uniform fit interface across the registry.

    - Trees / linear: pass sample_weight when supported.
    - Pipelines: route via `est__sample_weight`.
    - MLP: unweighted (sklearn MLPRegressor limitation).
    """
    if not spec.supports_sample_weight or sample_weight is None:
        estimator.fit(X, y)
        return estimator
    if isinstance(estimator, Pipeline):
        estimator.fit(X, y, est__sample_weight=sample_weight)
    else:
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


def underlying_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Return the inner regressor if wrapped in a Pipeline."""
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["est"]
    return estimator
