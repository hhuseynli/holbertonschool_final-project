"""Conformalized Quantile Regression (CQR) for calibrated price intervals.

Methodology (Romano, Patterson & Candes 2019, "Conformalized Quantile
Regression"):

1. Fit three XGBoost quantile regressors (``objective="reg:quantileerror"``)
   at levels 0.10 / 0.50 / 0.90 on the *training* months.
2. On a disjoint *calibration* window that comes strictly after training
   (respecting the project's temporal-leakage rule), compute conformity
   scores ``E_i = max(q10(x_i) - y_i,  y_i - q90(x_i))`` - how far each
   observed price falls outside the raw interval (negative when inside).
3. Take ``Q`` = the ``ceil((n+1)(1-alpha))/n``-th empirical quantile of the
   scores (alpha = 0.2 for a nominal 80% interval) and widen both bounds by
   ``Q``: ``[q10 - Q, q90 + Q]``.

Under exchangeability of calibration and future rows this guarantees
``P(y in [q10, q90]) >= 1 - alpha`` in finite samples, regardless of how
badly the quantile models themselves are calibrated.  On a drifting market
the guarantee is approximate - which is exactly why the pipeline reports
empirical coverage on held-out months next to the intervals.

The 0.50 model provides the point forecast; monotonicity
``q10 <= q50 <= q90`` is enforced after the conformal adjustment because
independently-fitted quantile models can cross.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from bakuml import config

# Lighter than the point-forecast baseline: three models are fitted, and
# quantile pinball loss is noisier per tree anyway.
DEFAULT_QUANTILE_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}


@dataclass
class ConformalModel:
    """Three fitted quantile models plus the CQR calibration offset."""

    models: dict[float, XGBRegressor]      # quantile level -> fitted model
    feature_cols: list[str]
    offset: float                          # CQR widening Q (applied to both bounds)
    alpha: float                           # nominal miscoverage (0.2 -> 80% interval)
    train_months: list[str] = field(default_factory=list)
    cal_months: list[str] = field(default_factory=list)


def fit_cqr(
    fm: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = config.TARGET_COL,
    train_months: list[str],
    cal_months: list[str],
    seed: int = 0,
    alpha: float = 0.2,
    params: dict | None = None,
) -> ConformalModel:
    """Fit quantile models on ``train_months``, calibrate on ``cal_months``.

    ``cal_months`` must come strictly after ``train_months`` (temporal
    leakage rule): calibration residuals may not inform months the models
    were trained on.
    """
    train_months = sorted(set(train_months))
    cal_months = sorted(set(cal_months))
    if not train_months or not cal_months:
        raise ValueError("train_months and cal_months must both be non-empty")
    if set(train_months) & set(cal_months):
        raise ValueError("train_months and cal_months overlap")
    if train_months[-1] >= cal_months[0]:
        raise ValueError(
            "cal_months must come strictly after train_months "
            f"(train ends {train_months[-1]}, calibration starts {cal_months[0]})"
        )

    train = fm[fm["month"].isin(train_months) & fm[target].notna()]
    cal = fm[fm["month"].isin(cal_months) & fm[target].notna()]
    if train.empty or cal.empty:
        raise ValueError("no usable rows in the train and/or calibration window")

    merged = {**DEFAULT_QUANTILE_PARAMS, **(params or {})}
    models: dict[float, XGBRegressor] = {}
    for q in config.CONFORMAL_QUANTILES:
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            random_state=seed, **merged,
        )
        model.fit(train[feature_cols], train[target])
        models[q] = model

    q_lo, _, q_hi = sorted(config.CONFORMAL_QUANTILES)
    y_cal = cal[target].to_numpy(dtype=float)
    lo = np.asarray(models[q_lo].predict(cal[feature_cols]), dtype=float)
    hi = np.asarray(models[q_hi].predict(cal[feature_cols]), dtype=float)
    scores = np.maximum(lo - y_cal, y_cal - hi)

    # Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest
    # score. With very small calibration sets the rank can exceed n; it is
    # clipped to the maximum score (the guarantee then degrades gracefully).
    n = len(scores)
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    offset = float(np.sort(scores)[rank - 1])

    return ConformalModel(
        models=models,
        feature_cols=list(feature_cols),
        offset=offset,
        alpha=float(alpha),
        train_months=train_months,
        cal_months=cal_months,
    )


def predict_intervals(cm: ConformalModel, fm: pd.DataFrame) -> pd.DataFrame:
    """Calibrated ``[q10, q50, q90]`` intervals, index-aligned with ``fm``.

    Both outer bounds are widened by the conformal offset; monotonicity
    ``q10 <= q50 <= q90`` is enforced afterwards (independently fitted
    quantile models can cross, especially after the adjustment).
    """
    X = fm[cm.feature_cols]
    q_lo, q_mid, q_hi = sorted(cm.models)
    lo = np.asarray(cm.models[q_lo].predict(X), dtype=float) - cm.offset
    mid = np.asarray(cm.models[q_mid].predict(X), dtype=float)
    hi = np.asarray(cm.models[q_hi].predict(X), dtype=float) + cm.offset
    return pd.DataFrame(
        {
            "q10": np.minimum(lo, mid),
            "q50": mid,
            "q90": np.maximum(hi, mid),
        },
        index=fm.index,
    )


def coverage(y, q10, q90) -> float:
    """Empirical share of observations falling inside ``[q10, q90]``."""
    y = np.asarray(y, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    ok = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)
    if not ok.any():
        return float("nan")
    inside = (q10[ok] <= y[ok]) & (y[ok] <= q90[ok])
    return float(inside.mean())
