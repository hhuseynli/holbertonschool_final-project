"""Tests for bakuml.models.stgcn (Tier-2 forecasters).

A small planted panel (37 H3 cells x 36 months) is generated with a known
data-generating process — per-cell trend + spatial diffusion toward the
neighbour mean + small noise, with random observation holes — so the tests
can check *recovery*, not just execution:

* ``panel_tensor`` densifies a long complete panel in the requested
  month/cell order and reports the observation mask faithfully;
* each forecaster's walk-forward 1-step-ahead MAE over the last 6 months
  beats the persistence forecast (last observed price);
* ``forecast()`` rolls out to the configured 12/24-month horizons with
  finite positive prices for every cell;
* STGCN training loss decreases;
* ``make_forecaster`` picks STGCN when torch is available and falls back
  to the ridge model when it is not (simulated by monkeypatching
  ``HAS_TORCH`` — torch stays installed).

Epochs/hidden size are kept small (40 / 16) so the file runs well under 90 s.
"""

from __future__ import annotations

from types import SimpleNamespace

import h3
import numpy as np
import pandas as pd
import pytest

from bakuml import config
from bakuml.data.schema import month_range
from bakuml.models import stgcn

if stgcn.HAS_TORCH:
    # The tensors here are tiny; a single thread is faster and keeps the
    # runtime stable when the CI machine is otherwise loaded.
    stgcn.torch.set_num_threads(1)

BASE_CELL = h3.latlng_to_cell(40.40, 49.85, 8)
STGCN_KW = dict(hidden=16, ks=3, seed=0)  # small net -> fast tests
EPOCHS = 40


def _normalized_adjacency(cells: list[str]) -> np.ndarray:
    """D^-1/2 (A+I) D^-1/2 over the H3 contiguity of `cells`.

    Built locally (h3 + numpy) so this test does not depend on the
    concurrently developed bakuml.spatial package.
    """
    pos = {c: i for i, c in enumerate(cells)}
    a = np.eye(len(cells))
    for c in cells:
        for nb in h3.grid_disk(c, 1):
            j = pos.get(nb)
            if j is not None and j != pos[c]:
                a[pos[c], j] = a[j, pos[c]] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(a.sum(axis=1))
    return a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


@pytest.fixture(scope="module")
def planted() -> SimpleNamespace:
    """Planted spatiotemporal process on a 37-cell disk, 36 months.

    log p_t = log p_{t-1} + trend_c + 0.3 * (nbr_mean_{t-1} - own_{t-1})
              + N(0, 0.004),   with ~8 % of entries masked out as holes.
    """
    rng = np.random.default_rng(42)
    cells = sorted(h3.grid_disk(BASE_CELL, 3))  # 37 cells
    months = month_range("2023-01", "2025-12")  # 36 months
    n, t = len(cells), len(months)

    adj = _normalized_adjacency(cells)
    w = adj.copy()
    np.fill_diagonal(w, 0.0)
    w /= w.sum(axis=1, keepdims=True)  # every cell in a disk has neighbours

    log_p = np.zeros((t, n))
    log_p[0] = np.log(2000.0) + rng.normal(0.0, 0.15, size=n)
    trend = rng.uniform(0.004, 0.012, size=n)
    for i in range(1, t):
        nbr = w @ log_p[i - 1]
        log_p[i] = (
            log_p[i - 1]
            + trend
            + 0.3 * (nbr - log_p[i - 1])
            + rng.normal(0.0, 0.004, size=n)
        )
    truth = np.exp(log_p)

    mask = rng.random((t, n)) > 0.08  # ~8 % holes
    mask[0] = True  # anchor the forward fill
    values = np.where(mask, truth, np.nan).astype(np.float32)
    return SimpleNamespace(
        cells=cells, months=months, adj=adj, truth=truth, values=values, mask=mask
    )


@pytest.fixture(scope="module")
def fitted_stgcn(planted: SimpleNamespace) -> stgcn.STGCNForecaster:
    """One full-panel STGCN fit shared by the rollout / loss tests."""
    fc = stgcn.STGCNForecaster(planted.adj, planted.cells, planted.months, **STGCN_KW)
    return fc.fit(planted.values, planted.mask, epochs=EPOCHS)


@pytest.fixture(scope="module")
def fitted_ridge(planted: SimpleNamespace) -> stgcn.SpatialLagRidgeForecaster:
    fc = stgcn.SpatialLagRidgeForecaster(
        planted.adj, planted.cells, planted.months, seed=0
    )
    return fc.fit(planted.values, planted.mask)


# ---------------------------------------------------------------------------
# panel_tensor
# ---------------------------------------------------------------------------


def _long_panel(planted: SimpleNamespace, shuffle_seed: int = 7) -> pd.DataFrame:
    """Complete long panel (every cell x month, NaN in holes), shuffled rows."""
    t, n = planted.values.shape
    frame = pd.DataFrame(
        {
            "h3": np.tile(planted.cells, t),
            "month": np.repeat(planted.months, n),
            config.TARGET_COL: planted.values.ravel().astype(np.float64),
        }
    )
    return frame.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)


def test_panel_tensor_shapes_mask_and_order(planted: SimpleNamespace) -> None:
    panel = _long_panel(planted)
    values, mask = stgcn.panel_tensor(panel, planted.months, planted.cells)

    t, n = len(planted.months), len(planted.cells)
    assert values.shape == (t, n) and mask.shape == (t, n)
    assert values.dtype == np.float32 and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, planted.mask)
    np.testing.assert_allclose(values[mask], planted.values[planted.mask], rtol=1e-6)
    assert np.isnan(values[~mask]).all()

    # ordering follows the *passed* lists, not the frame order
    rev_values, rev_mask = stgcn.panel_tensor(
        panel, planted.months, list(reversed(planted.cells))
    )
    np.testing.assert_array_equal(rev_values[:, ::-1], values)
    np.testing.assert_array_equal(rev_mask[:, ::-1], mask)

    # cells absent from the frame come out unobserved
    extra = ["8f2830828052d25"]  # not a real study cell
    ext_values, ext_mask = stgcn.panel_tensor(
        panel, planted.months, planted.cells + extra
    )
    assert not ext_mask[:, -1].any()
    assert np.isnan(ext_values[:, -1]).all()


# ---------------------------------------------------------------------------
# 1-step-ahead walk-forward vs persistence
# ---------------------------------------------------------------------------


def _last_observed(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell last observed price (the persistence forecast)."""
    out = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        idx = np.nonzero(mask[:, j])[0]
        if idx.size:
            out[j] = values[idx[-1], j]
    return out


def _walk_forward_mae(make_forecaster, planted: SimpleNamespace) -> tuple[float, float]:
    """(model MAE, persistence MAE) of 1-step-ahead forecasts, last 6 months."""
    t = len(planted.months)
    model_err, persist_err = [], []
    for i in range(t - 6, t):
        fc = make_forecaster(planted.months[:i])
        fc.fit(planted.values[:i], planted.mask[:i], epochs=EPOCHS)
        pred = (
            fc.forecast(horizons=(1,))
            .set_index("h3")["y_pred"]
            .reindex(planted.cells)
            .to_numpy()
        )
        obs = planted.mask[i]
        actual = planted.truth[i]
        persistence = _last_observed(planted.values[:i], planted.mask[:i])
        model_err.append(np.abs(pred[obs] - actual[obs]))
        persist_err.append(np.abs(persistence[obs] - actual[obs]))
    return (
        float(np.concatenate(model_err).mean()),
        float(np.concatenate(persist_err).mean()),
    )


def test_stgcn_beats_persistence(planted: SimpleNamespace) -> None:
    def make(months):
        return stgcn.STGCNForecaster(planted.adj, planted.cells, months, **STGCN_KW)

    model_mae, persistence_mae = _walk_forward_mae(make, planted)
    assert np.isfinite(model_mae)
    assert model_mae < persistence_mae, (
        f"STGCN MAE {model_mae:.2f} should beat persistence {persistence_mae:.2f}"
    )


def test_ridge_beats_persistence(planted: SimpleNamespace) -> None:
    def make(months):
        return stgcn.SpatialLagRidgeForecaster(
            planted.adj, planted.cells, months, seed=0
        )

    model_mae, persistence_mae = _walk_forward_mae(make, planted)
    assert np.isfinite(model_mae)
    assert model_mae < persistence_mae, (
        f"ridge MAE {model_mae:.2f} should beat persistence {persistence_mae:.2f}"
    )


# ---------------------------------------------------------------------------
# Rollout shape / finiteness at the configured horizons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["stgcn", "ridge"])
def test_forecast_rollout(
    which: str, planted: SimpleNamespace, fitted_stgcn, fitted_ridge
) -> None:
    fc = fitted_stgcn if which == "stgcn" else fitted_ridge
    out = fc.forecast()  # default config.FORECAST_HORIZONS == (12, 24)

    n = len(planted.cells)
    assert list(out.columns) == ["h3", "horizon_months", "y_pred"]
    assert len(out) == n * len(config.FORECAST_HORIZONS)
    assert out["horizon_months"].dtype == np.int64
    assert set(out["horizon_months"]) == set(config.FORECAST_HORIZONS)
    for h in config.FORECAST_HORIZONS:
        assert set(out.loc[out["horizon_months"] == h, "h3"]) == set(planted.cells)
    assert np.isfinite(out["y_pred"]).all()
    # back in AZN/m2 price space and economically sane for the planted market
    assert (out["y_pred"] > 500).all() and (out["y_pred"] < 20_000).all()


def test_forecast_requires_fit(planted: SimpleNamespace) -> None:
    fc = stgcn.SpatialLagRidgeForecaster(planted.adj, planted.cells, planted.months)
    with pytest.raises(RuntimeError):
        fc.forecast()


# ---------------------------------------------------------------------------
# STGCN training behaviour
# ---------------------------------------------------------------------------


def test_stgcn_training_loss_decreases(fitted_stgcn) -> None:
    hist = fitted_stgcn.history_
    assert len(hist) == EPOCHS
    assert all(np.isfinite(hist))
    assert np.mean(hist[-5:]) < np.mean(hist[:5]), "training loss should decrease"


def test_stgcn_deterministic(planted: SimpleNamespace, fitted_stgcn) -> None:
    """Same seed + data => identical forecasts."""
    again = stgcn.STGCNForecaster(
        planted.adj, planted.cells, planted.months, **STGCN_KW
    ).fit(planted.values, planted.mask, epochs=EPOCHS)
    pd.testing.assert_frame_equal(again.forecast(), fitted_stgcn.forecast())


# ---------------------------------------------------------------------------
# Factory + torch-less fallback (monkeypatched, torch stays installed)
# ---------------------------------------------------------------------------


def test_make_forecaster_prefers_stgcn(planted: SimpleNamespace) -> None:
    assert stgcn.HAS_TORCH, "torch is installed in the test environment"
    fc = stgcn.make_forecaster(planted.adj, planted.cells, planted.months)
    assert isinstance(fc, stgcn.STGCNForecaster)
    ridge = stgcn.make_forecaster(
        planted.adj, planted.cells, planted.months, prefer="ridge"
    )
    assert isinstance(ridge, stgcn.SpatialLagRidgeForecaster)
    with pytest.raises(ValueError):
        stgcn.make_forecaster(planted.adj, planted.cells, planted.months, prefer="lstm")


def test_fallback_without_torch(
    planted: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stgcn, "HAS_TORCH", False)
    fc = stgcn.make_forecaster(
        planted.adj, planted.cells, planted.months, prefer="stgcn"
    )
    assert isinstance(fc, stgcn.SpatialLagRidgeForecaster)
    with pytest.raises(ImportError):
        stgcn.STGCNForecaster(planted.adj, planted.cells, planted.months)
