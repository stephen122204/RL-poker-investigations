"""Unified equity benchmark: one code path for preflop, flop, turn, river.

Runs every model in the registry with a shared train/test split and K-fold CV,
saves per-model fitted estimators, and writes a per-stage results CSV/JSON.
Optional side artifacts: interpretability, error analysis, conformal calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from ..config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, METRICS_DIR, MODELS_DIR
from ..utils import get_logger, set_seed
from .data_loader import ensure_dataset_downloaded, load_equity_table
from .error_analysis import POSTFLOP_GROUP_COLS, PREFLOP_GROUP_COLS, run_error_analysis
from .evaluate import regression_report, summarize_cv_metrics
from .interpretability import build_full_importance_table, save_importance_csv
from .models import MODEL_REGISTRY, ModelSpec, fit_with_optional_weights, make_estimator
from .preprocess import prepare_postflop_with_weights, prepare_preflop
from .split_analysis import (
    analyze_postflop_street_buckets,
    analyze_preflop_buckets,
    write_split_analysis,
)
from .uncertainty import coverage_curve, fit_split_conformal
from .xy_builders import build_xy_postflop_street, build_xy_preflop


log = get_logger("rlpoker.equity.train")


@dataclass
class StageResult:
    stage: str
    per_model: list[dict]
    fitted_models: dict[str, BaseEstimator]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    w_train: np.ndarray
    w_test: np.ndarray
    feature_names: list[str]
    meta_test: pd.DataFrame


def load_stage_data(stage: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Return (X, y, weights, groups_or_None, meta_full) for a street."""
    csv_dir = ensure_dataset_downloaded()
    raw = load_equity_table(csv_dir, stage)
    if stage == "preflop":
        X, y, weights, groups = build_xy_preflop(raw)
        meta = prepare_preflop(raw).reset_index(drop=True)
        return X, y, weights, groups, meta
    X, y, weights = build_xy_postflop_street(raw)
    meta = prepare_postflop_with_weights(raw).reset_index(drop=True)
    return X, y, weights, None, meta


def cv_fold_metrics(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    groups_train: Optional[np.ndarray],
    cv_folds: int,
    random_state: int,
    use_grouped: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run K-fold CV; return (weighted_reports, unweighted_reports)."""
    if use_grouped and groups_train is not None:
        splitter = GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups=groups_train)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state).split(X_train)
    w_reports: list[dict] = []
    u_reports: list[dict] = []
    for tr_idx, va_idx in splitter:
        est, _ = make_estimator(spec.name, random_state=random_state)
        Xtr = X_train.iloc[tr_idx]
        Xva = X_train.iloc[va_idx]
        ytr = y_train[tr_idx]
        yva = y_train[va_idx]
        wtr = w_train[tr_idx]
        wva = w_train[va_idx]
        fit_with_optional_weights(est, spec, Xtr, ytr, wtr if spec.supports_sample_weight else None)
        yhat = est.predict(Xva)
        w_reports.append(regression_report(yva, yhat, sample_weight=wva))
        u_reports.append(regression_report(yva, yhat))
    return w_reports, u_reports


def run_stage_benchmark(
    stage: str,
    models: Iterable[str] = tuple(MODEL_REGISTRY),
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    metrics_dir: Path = METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    save_models: bool = True,
    grouped_cv: bool = False,
) -> StageResult:
    set_seed(random_state)
    log.info("=== %s ===", stage.upper())

    X, y, weights, groups, meta_full = load_stage_data(stage)
    y_arr = np.asarray(y, dtype=np.float64)
    idx = np.arange(len(y_arr))

    splitter_kwargs = dict(test_size=test_size, random_state=random_state)
    if stage == "preflop" and groups is not None:
        X_train, X_test, y_train, y_test, w_train, w_test, g_train, _, idx_train, idx_test = (
            train_test_split(X, y_arr, weights, groups, idx, **splitter_kwargs)
        )
    else:
        X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
            X, y_arr, weights, idx, **splitter_kwargs
        )
        g_train = None

    meta_test = meta_full.iloc[idx_test].reset_index(drop=True)
    feature_names = list(X.columns)

    per_model: list[dict] = []
    fitted: dict[str, BaseEstimator] = {}

    for name in models:
        if name not in MODEL_REGISTRY:
            log.warning("Skipping unknown model %s", name)
            continue
        spec = MODEL_REGISTRY[name]
        log.info("[%s] CV + test for %s", stage, spec.display)
        try:
            w_reports, u_reports = cv_fold_metrics(
                spec,
                X_train,
                y_train,
                w_train,
                g_train,
                cv_folds=cv_folds,
                random_state=random_state,
                use_grouped=grouped_cv,
            )
        except Exception as e:
            log.warning("[%s] %s failed CV: %s", stage, spec.display, e)
            continue

        est, _ = make_estimator(spec.name, random_state=random_state)
        try:
            fit_with_optional_weights(est, spec, X_train, y_train, w_train if spec.supports_sample_weight else None)
        except Exception as e:
            log.warning("[%s] %s failed final fit: %s", stage, spec.display, e)
            continue
        yhat = est.predict(X_test)
        test_w = regression_report(y_test, yhat, sample_weight=w_test)
        test_u = regression_report(y_test, yhat)

        fitted[spec.name] = est
        if save_models:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, models_dir / f"{stage}_{spec.name}.joblib")

        w_summary = summarize_cv_metrics(w_reports)
        u_summary = summarize_cv_metrics(u_reports)
        per_model.append(
            {
                "stage": stage,
                "model": spec.name,
                "display": spec.display,
                "kind": spec.kind,
                "cv_folds": cv_folds,
                "cv_mae_mean": w_summary.get("mae", {}).get("mean"),
                "cv_mae_std": w_summary.get("mae", {}).get("std"),
                "cv_rmse_mean": w_summary.get("rmse", {}).get("mean"),
                "cv_rmse_std": w_summary.get("rmse", {}).get("std"),
                "cv_r2_mean": w_summary.get("r2", {}).get("mean"),
                "cv_r2_std": w_summary.get("r2", {}).get("std"),
                "test_mae": test_w["mae"],
                "test_rmse": test_w["rmse"],
                "test_r2": test_w["r2"],
                "test_mae_unweighted": test_u["mae"],
                "test_rmse_unweighted": test_u["rmse"],
                "test_r2_unweighted": test_u["r2"],
                "cv_weighted_folds": w_reports,
                "cv_unweighted_folds": u_reports,
            }
        )

    df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if not isinstance(v, list)}
            for r in per_model
        ]
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / f"{stage}_results.csv", index=False)
    with open(metrics_dir / f"{stage}_results.json", "w") as f:
        json.dump(per_model, f, indent=2, default=float)
    log.info("[%s] wrote %s / %s", stage, f"{stage}_results.csv", f"{stage}_results.json")

    return StageResult(
        stage=stage,
        per_model=per_model,
        fitted_models=fitted,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        w_train=w_train,
        w_test=w_test,
        feature_names=feature_names,
        meta_test=meta_test,
    )


def run_side_artifacts(
    result: StageResult,
    metrics_dir: Path = METRICS_DIR,
    interpret_models: Iterable[str] = ("random_forest", "lightgbm", "ridge"),
    conformal_models: Iterable[str] = ("random_forest", "lightgbm"),
    run_error: bool = True,
    run_split: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """After benchmarking a stage, write interpretability + error + conformal artifacts."""
    stage = result.stage
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if run_split:
        if stage == "preflop":
            info = analyze_preflop_buckets(result.meta_test)
        else:
            info = analyze_postflop_street_buckets(result.meta_test, list(result.meta_test.columns))
        write_split_analysis(metrics_dir / f"{stage}_split_analysis.json", info)

    if run_error:
        group_cols = PREFLOP_GROUP_COLS if stage == "preflop" else POSTFLOP_GROUP_COLS
        for name, est in result.fitted_models.items():
            out = metrics_dir / f"error_analysis_{stage}_{name}.csv"
            run_error_analysis(
                est,
                result.X_test,
                result.y_test,
                result.w_test,
                result.meta_test,
                stage,
                group_cols,
                out,
            )

    for name in interpret_models:
        if name not in result.fitted_models:
            continue
        est = result.fitted_models[name]
        spec = MODEL_REGISTRY[name]
        try:
            imp = build_full_importance_table(
                est, spec, result.X_train, result.X_test, result.y_test,
                result.feature_names, random_state=random_state,
            )
            save_importance_csv(imp, metrics_dir / f"importance_{stage}_{name}.csv")
        except Exception as e:
            log.warning("importance %s/%s failed: %s", stage, name, e)

    for name in conformal_models:
        if name not in result.fitted_models:
            continue
        spec = MODEL_REGISTRY[name]
        try:
            base, _ = make_estimator(spec.name, random_state=random_state)
            cp = fit_split_conformal(
                base, result.X_train, result.y_train, result.w_train, random_state=random_state
            )
            yhat_test = cp.predict(result.X_test)
            # Estimate empirical coverage over a sweep of alpha using held-out residuals.
            cal_residuals = np.abs(result.y_train - cp.base_estimator.predict(result.X_train))
            cov = coverage_curve(result.y_test, yhat_test, cal_residuals)
            cov.insert(0, "model", name)
            cov.insert(0, "stage", stage)
            cov.to_csv(metrics_dir / f"conformal_{stage}_{name}.csv", index=False)
        except Exception as e:
            log.warning("conformal %s/%s failed: %s", stage, name, e)


def run_all_stages(
    stages: Iterable[str] = ("preflop", "flop", "turn", "river"),
    models: Iterable[str] = tuple(MODEL_REGISTRY),
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    do_side_artifacts: bool = True,
) -> dict[str, StageResult]:
    out: dict[str, StageResult] = {}
    for stage in stages:
        res = run_stage_benchmark(
            stage,
            models=models,
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
        )
        if do_side_artifacts:
            run_side_artifacts(res, random_state=random_state)
        out[stage] = res
    return out
