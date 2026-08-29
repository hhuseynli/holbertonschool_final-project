"""Tests for bakuml.models.baseline and bakuml.models.conformal.

The models consume a feature matrix + feature_cols per the DESIGN.md
contract, so these tests stay decoupled from ``bakuml.features`` by
building a self-contained panel-like matrix over real H3 cells with a
*planted* data-generating process:

    log price(cell, t) = mu(cell) + trend * t + u(cell, t) + meas(cell, t)

* ``mu(cell)`` - a centre-distance gradient plus a small cell effect,
* ``trend`` - a citywide monthly drift,
* ``u`` - an AR(1) latent market state (phi=0.7), the genuinely
  predictable dynamics,
* ``meas`` - i.i.d. measurement noise, mimicking the sampling error of a
  cell-month *median* computed from a handful of listings.

The measurement noise is the crux of the "model beats persistence" test:
naive persistence copies last month's noisy median (so its error contains
two independent noise draws plus the drift it ignores), while the model can
filter the noise by combining ``lag_own_1m``, ``lag_own_3m`` and the static
gradient. Two pure-noise features are included so SHAP has something to
(correctly) rank at the bottom.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest

from bakuml import config
from bakuml.data.schema import month_range
from bakuml.geo import haversine_km
from bakuml.models.baseline import (
    TrainedBaseline,
    evaluate_spatial_cv,
    evaluate_walk_forward,
    predict,
    shap_summary,
    train_baseline,
)
from bakuml.models.conformal import coverage, fit_cqr, predict_intervals

FEATURE_COLS = [
    "lag_own_1m", "lag_own_3m", "dist_centre_km", "month_ix",
    "noise_a", "noise_b",
]
# Small trees keep the whole file fast (contract: < 90 s).
FAST_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.10,
    "max_depth": 4,
}
SEED = 123
# The compact test disk needs a finer block resolution than the citywide
# default to yield enough parent blocks for 5 folds (see test_validation).
BLOCK_RES = 7


@pytest.fixture(scope="module")
def fm() -> pd.DataFrame:
    """Feature matrix over 127 real H3 cells x 40 months (37 after lags)."""
    rng = np.random.default_rng(SEED)
    base = h3.latlng_to_cell(40.37, 49.84, 8)
    cells = sorted(h3.grid_disk(base, 6))
    months = month_range("2023-01", "2026-04")  # 40 months
    T, N = len(months), len(cells)

    lat = np.array([h3.cell_to_latlng(c)[0] for c in cells])
    lon = np.array([h3.cell_to_latlng(c)[1] for c in cells])
    dist = haversine_km(lat, lon, *config.CITY_CENTRE)

    mu = np.log(2400.0) - 0.09 * dist + rng.normal(0.0, 0.05, N)
    trend, phi, sigma, s_meas = 0.004, 0.7, 0.03, 0.05
    u = np.zeros((T, N))
    u[0] = rng.normal(0.0, sigma / np.sqrt(1 - phi**2), N)  # stationary start
    for t in range(1, T):
        u[t] = phi * u[t - 1] + rng.normal(0.0, sigma, N)
    latent = mu[None, :] + trend * np.arange(T)[:, None] + u
    price = np.exp(latent + rng.normal(0.0, s_meas, (T, N)))

    rows = []
    for t in range(3, T):  # drop warm-up months without a 3-month lag
        rows.append(pd.DataFrame({
            "h3": cells,
            "month": months[t],
            "price_azn_m2_median": price[t],
            "lag_own_1m": price[t - 1],
            "lag_own_3m": price[t - 3],
            "dist_centre_km": dist,
            "month_ix": t,
            "noise_a": rng.normal(size=N),
            "noise_b": rng.normal(size=N),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


def test_train_and_predict_shapes(fm):
    tb = train_baseline(fm, FEATURE_COLS, seed=0, params=FAST_PARAMS)
    assert isinstance(tb, TrainedBaseline)
    assert tb.feature_cols == FEATURE_COLS
    assert tb.train_months == sorted(fm["month"].unique().tolist())
    preds = predict(tb, fm)
    assert preds.shape == (len(fm),)
    assert np.isfinite(preds).all()
    # in-sample fit should at least be in the right ballpark
    mae = np.abs(preds - fm[config.TARGET_COL].to_numpy()).mean()
    assert mae < 0.15 * fm[config.TARGET_COL].mean()


# ---------------------------------------------------------------------------
# walk-forward evaluation
# ---------------------------------------------------------------------------


def test_walk_forward_beats_naive_persistence(fm):
    report = evaluate_walk_forward(fm, FEATURE_COLS, seed=0, params=FAST_PARAMS)
    assert set(report) >= {"splits", "aggregate", "naive"}
    n_months = fm["month"].nunique()
    assert report["n_splits"] == n_months - config.WALK_FORWARD_MIN_TRAIN_MONTHS

    for split in report["splits"]:
        assert split["train_end"] < min(split["test_months"])  # no leakage
        assert split["n_train"] > 0 and split["n_test"] > 0
        for key in ("mae", "mape", "r2"):
            assert np.isfinite(split[key])
            assert np.isfinite(split["naive"][key])

    agg, naive = report["aggregate"], report["naive"]
    for metrics in (agg, naive):
        assert np.isfinite(metrics["mae"])
        assert 0.0 < metrics["mape"] < 100.0
    # headline claim: the model must beat "next month = last month"
    assert agg["mae"] < naive["mae"]
    assert agg["r2"] > 0.5


# ---------------------------------------------------------------------------
# spatial CV evaluation
# ---------------------------------------------------------------------------


def test_spatial_cv_reports_finite_metrics(fm):
    report = evaluate_spatial_cv(
        fm, FEATURE_COLS, n_folds=5, seed=0, block_res=BLOCK_RES,
        params=FAST_PARAMS,
    )
    assert set(report) >= {"splits", "aggregate", "naive"}
    assert report["n_splits"] == 5
    # every row is held out exactly once across the folds
    assert sum(s["n_test"] for s in report["splits"]) == len(fm)
    for split in report["splits"]:
        for key in ("mae", "mape", "r2"):
            assert np.isfinite(split[key])
    assert np.isfinite(report["aggregate"]["mae"])
    assert 0.0 < report["aggregate"]["mape"] < 100.0
    assert np.isfinite(report["aggregate"]["r2"])


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


def test_shap_summary_ranks_planted_signal_first(fm):
    tb = train_baseline(fm, FEATURE_COLS, seed=0, params=FAST_PARAMS)
    summary = shap_summary(tb, fm, top_k=4)
    assert len(summary) == 4
    values = list(summary.values())
    assert all(isinstance(v, float) and v >= 0.0 for v in values)
    assert values == sorted(values, reverse=True)
    # the AR(1) process makes the own 1-month lag the dominant driver
    assert next(iter(summary)) == "lag_own_1m"
    assert set(summary) <= set(FEATURE_COLS)


# ---------------------------------------------------------------------------
# conformalized quantile regression
# ---------------------------------------------------------------------------


def test_conformal_coverage_and_monotone_quantiles(fm):
    months = sorted(fm["month"].unique().tolist())
    train_months, cal_months, test_months = (
        months[:26], months[26:34], months[34:]
    )
    cm = fit_cqr(
        fm, FEATURE_COLS,
        train_months=train_months, cal_months=cal_months,
        seed=0, params=FAST_PARAMS,
    )
    assert cm.feature_cols == FEATURE_COLS
    assert np.isfinite(cm.offset)

    test = fm[fm["month"].isin(test_months)]
    intervals = predict_intervals(cm, test)
    assert list(intervals.columns) == ["q10", "q50", "q90"]
    assert intervals.index.equals(test.index)  # index-aligned
    assert (intervals["q10"] <= intervals["q50"]).all()
    assert (intervals["q50"] <= intervals["q90"]).all()

    cov = coverage(
        test[config.TARGET_COL].to_numpy(),
        intervals["q10"].to_numpy(), intervals["q90"].to_numpy(),
    )
    # nominal 80%; the temporal shift makes the guarantee approximate
    assert 0.75 <= cov <= 0.95


def test_fit_cqr_rejects_leaky_windows(fm):
    months = sorted(fm["month"].unique().tolist())
    with pytest.raises(ValueError):  # overlapping windows
        fit_cqr(fm, FEATURE_COLS,
                train_months=months[:20], cal_months=months[15:25])
    with pytest.raises(ValueError):  # calibration before training
        fit_cqr(fm, FEATURE_COLS,
                train_months=months[10:20], cal_months=months[:10])


def test_coverage_helper():
    y = np.array([1.0, 2.0, 3.0, np.nan])
    q10 = np.array([0.0, 0.0, 4.0, 0.0])
    q90 = np.array([2.0, 3.0, 5.0, 1.0])
    assert coverage(y, q10, q90) == pytest.approx(2.0 / 3.0)
