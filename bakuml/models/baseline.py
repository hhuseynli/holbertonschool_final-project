"""Tier-1 model: gradient-boosted trees on the leakage-safe feature matrix.

Methodology
-----------
The baseline is an XGBoost regressor predicting the cell-month median price
(``config.TARGET_COL``) from the feature matrix built by ``bakuml.features``.
This module deliberately never imports the feature code: it consumes a plain
DataFrame plus an explicit ``feature_cols`` list, so models and features can
be developed (and unit-tested) independently against the DESIGN.md contract.

Evaluation happens along both axes on which the study could leak:

* **Temporal** - ``evaluate_walk_forward`` retrains on an expanding window of
  past months and scores strictly-future months
  (``bakuml.validation.temporal``).  Every walk-forward split is also scored
  with a *naive persistence* benchmark - "next month's price = last month's
  price" (``lag_own_1m``).  Persistence is notoriously hard to beat in asset
  markets; reporting the model only relative to it keeps the headline
  numbers honest.
* **Spatial** - ``evaluate_spatial_cv`` holds out whole contiguous blocks of
  H3 cells (``bakuml.validation.spatial_cv``) across *all* months, measuring
  how well the model transfers to neighbourhoods it has never seen.

Explainability uses TreeSHAP: XGBoost computes exact per-feature Shapley
contributions natively (``Booster.predict(pred_contribs=True)``), so no
external SHAP dependency is required.  ``shap_summary`` reports the mean
absolute contribution per feature - the standard "global importance" view.

Metrics: ``mae`` (AZN/m2), ``mape`` (percent), ``r2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from bakuml import config
from bakuml.validation.spatial_cv import spatial_block_folds
from bakuml.validation.temporal import walk_forward_splits

# Sane defaults per DESIGN.md: ~600 trees, lr 0.05, depth 6,
# subsample/colsample 0.8, no early stopping. Tests override with
# smaller values through the optional ``params`` argument.
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}

# Cap on rows used for the SHAP summary (deterministic subsample beyond it).
_SHAP_MAX_ROWS = 10_000


@dataclass
class TrainedBaseline:
    """A fitted Tier-1 model plus the metadata needed to reuse it safely."""

    model: XGBRegressor
    feature_cols: list[str]
    train_months: list[str]


def _make_model(seed: int, params: dict | None) -> XGBRegressor:
    merged = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return XGBRegressor(random_state=seed, **merged)


def _metric_dict(y_true, y_pred) -> dict[str, float]:
    """mae / mape(%) / r2 over the rows where both arrays are finite."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size == 0:
        return {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    err = np.abs(y_true - y_pred)
    nonzero = y_true != 0
    r2 = (
        float(r2_score(y_true, y_pred))
        if y_true.size >= 2 and float(np.var(y_true)) > 0
        else float("nan")
    )
    return {
        "mae": float(err.mean()),
        "mape": float((err[nonzero] / np.abs(y_true[nonzero])).mean() * 100.0)
        if nonzero.any()
        else float("nan"),
        "r2": r2,
    }


def train_baseline(
    fm: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = config.TARGET_COL,
    seed: int = 0,
    params: dict | None = None,
) -> TrainedBaseline:
    """Fit an XGBoost regressor on every row of ``fm`` with a valid target."""
    valid = fm[fm[target].notna()]
    if valid.empty:
        raise ValueError(f"no rows with a non-null target {target!r}")
    model = _make_model(seed, params)
    model.fit(valid[feature_cols], valid[target])
    train_months = (
        sorted(valid["month"].unique().tolist()) if "month" in valid.columns else []
    )
    return TrainedBaseline(model=model, feature_cols=list(feature_cols),
                           train_months=train_months)


def predict(tb: TrainedBaseline, fm: pd.DataFrame) -> np.ndarray:
    """Predict the target for every row of ``fm`` (row-aligned ndarray)."""
    return np.asarray(tb.model.predict(fm[tb.feature_cols]), dtype=float)


def _naive_metrics(frame: pd.DataFrame, target: str) -> dict[str, float]:
    """Persistence benchmark: prediction = last month's own median price."""
    return _metric_dict(frame[target], frame["lag_own_1m"])


def evaluate_walk_forward(
    fm: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = config.TARGET_COL,
    seed: int = 0,
    params: dict | None = None,
) -> dict:
    """Expanding-window walk-forward evaluation vs naive persistence.

    Returns ``{"splits": [...], "aggregate": {...}, "naive": {...},
    "n_splits": int}`` where each split entry carries per-split
    ``mae/mape/r2`` for the model and (nested) for the persistence
    benchmark; ``aggregate``/``naive`` pool all test rows across splits.
    """
    months = sorted(fm["month"].unique().tolist())
    splits = walk_forward_splits(months)

    split_reports: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_naive: list[np.ndarray] = []
    for train_months, test_months in splits:
        train = fm[fm["month"].isin(train_months) & fm[target].notna()]
        test = fm[fm["month"].isin(test_months) & fm[target].notna()]
        if train.empty or test.empty:
            continue
        model = _make_model(seed, params)
        model.fit(train[feature_cols], train[target])
        y_pred = np.asarray(model.predict(test[feature_cols]), dtype=float)
        y_true = test[target].to_numpy(dtype=float)
        y_naive = test["lag_own_1m"].to_numpy(dtype=float)
        split_reports.append(
            {
                "train_start": train_months[0],
                "train_end": train_months[-1],
                "test_months": list(test_months),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **_metric_dict(y_true, y_pred),
                "naive": _metric_dict(y_true, y_naive),
            }
        )
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        pooled_naive.append(y_naive)

    if not split_reports:
        raise ValueError("no walk-forward split produced any test rows")
    y_true = np.concatenate(pooled_true)
    return {
        "splits": split_reports,
        "aggregate": _metric_dict(y_true, np.concatenate(pooled_pred)),
        "naive": _metric_dict(y_true, np.concatenate(pooled_naive)),
        "n_splits": len(split_reports),
    }


def evaluate_spatial_cv(
    fm: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = config.TARGET_COL,
    n_folds: int = 5,
    seed: int = 0,
    block_res: int = config.H3_BLOCK_RESOLUTION,
    params: dict | None = None,
) -> dict:
    """Blocked spatial CV: hold out whole H3 parent blocks, all months.

    Each fold's cells are fully unseen at train time (the model trains on
    the other folds across *all* months), so the score measures spatial
    transfer rather than temporal forecasting.  Folds that end up empty
    (possible when the cells span few parent blocks) are skipped.
    Same return structure as :func:`evaluate_walk_forward`, with per-fold
    entries under ``"splits"``.
    """
    cells = sorted(fm["h3"].unique().tolist())
    folds = spatial_block_folds(
        cells, n_folds=n_folds, block_res=block_res, seed=seed
    )
    fold_of = fm["h3"].map(folds)

    split_reports: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_naive: list[np.ndarray] = []
    for k in range(n_folds):
        test_mask = (fold_of == k) & fm[target].notna()
        train_mask = (fold_of != k) & fm[target].notna()
        if not bool(test_mask.any()) or not bool(train_mask.any()):
            continue
        train, test = fm[train_mask], fm[test_mask]
        model = _make_model(seed, params)
        model.fit(train[feature_cols], train[target])
        y_pred = np.asarray(model.predict(test[feature_cols]), dtype=float)
        y_true = test[target].to_numpy(dtype=float)
        y_naive = test["lag_own_1m"].to_numpy(dtype=float)
        split_reports.append(
            {
                "fold": k,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_test_cells": int(test["h3"].nunique()),
                **_metric_dict(y_true, y_pred),
                "naive": _metric_dict(y_true, y_naive),
            }
        )
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        pooled_naive.append(y_naive)

    if not split_reports:
        raise ValueError("no spatial fold produced any test rows")
    y_true = np.concatenate(pooled_true)
    return {
        "splits": split_reports,
        "aggregate": _metric_dict(y_true, np.concatenate(pooled_pred)),
        "naive": _metric_dict(y_true, np.concatenate(pooled_naive)),
        "n_splits": len(split_reports),
    }


def shap_summary(
    tb: TrainedBaseline, fm: pd.DataFrame, top_k: int = 15
) -> dict[str, float]:
    """Mean |SHAP| per feature, sorted descending, JSON-safe floats.

    Uses XGBoost's built-in exact TreeSHAP (``pred_contribs=True``); the
    last contribution column is the bias term and is excluded.  For very
    large matrices a deterministic random subsample of rows keeps this
    cheap without changing the ranking materially.
    """
    X = fm[tb.feature_cols]
    if len(X) > _SHAP_MAX_ROWS:
        idx = np.random.default_rng(0).choice(
            len(X), size=_SHAP_MAX_ROWS, replace=False
        )
        X = X.iloc[np.sort(idx)]
    contribs = tb.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)  # drop the bias column
    ranked = sorted(
        zip(tb.feature_cols, mean_abs), key=lambda kv: -float(kv[1])
    )
    return {name: float(value) for name, value in ranked[:top_k]}
