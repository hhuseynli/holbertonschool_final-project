"""Tier-2 spatiotemporal forecasting: compact STGCN + spatial-lag ridge fallback.

Methodology
-----------
The Tier-1 XGBoost baseline treats every (cell, month) row independently;
this module instead models the *joint* evolution of the whole H3 price
surface, so that a shock in one cell (e.g. a metro opening) can propagate
to its neighbours over time.

Two forecasters share one interface (``fit(values, mask)`` +
``forecast(horizons)``):

1. ``STGCNForecaster`` — a compact Spatio-Temporal Graph Convolutional
   Network in the spirit of Yu, Yin & Zhu (IJCAI 2018). One ST block:

       gated temporal conv  →  graph conv (Â X W₁)  →  gated temporal conv
       →  1×1-over-time readout

   * Works entirely in **log price** space: real-estate prices are
     log-normal-ish and planted effects are multiplicative.
   * Inputs are sliding windows of ``WINDOW = 12`` months over the panel
     tensor; unobserved entries are forward-filled along time (leading
     gaps take the cell's median log price) so the network always sees a
     dense window, while the **loss is masked** to observed targets only —
     imputed values never act as supervision.
   * Series are z-score normalised (moments from observed entries only).
   * Temporal convolutions are *gated*: ``tanh(P) ⊙ σ(Q)`` where P, Q are
     two halves of one Conv2d over the time axis (kernel ``ks × 1``); the
     sigmoid gate learns which temporal patterns pass through.
   * The graph convolution multiplies by the symmetrically normalised
     adjacency ``Â = D^{-1/2}(A+I)D^{-1/2}`` (built by
     ``bakuml.spatial.graph.build_adjacency`` and passed in — this module
     never imports ``bakuml.spatial``), mixing each cell with its H3
     neighbours.
   * The network predicts the **increment** over the last observed window
     value (a residual/skip connection): at initialisation the model is
     therefore exactly the persistence forecast, and training only has to
     learn the deviation from it — a standard trick that stabilises
     optimisation on short panels.
   * Trained full-batch with Adam on masked MSE; everything is seeded
     (``torch.manual_seed`` + numpy) for reproducibility.

2. ``SpatialLagRidgeForecaster`` — a transparent econometric benchmark and
   the fallback when torch is unavailable: per (cell, t) it regresses
   log price on the cell's own three monthly lags, the *neighbour-mean*
   log price at the same three lags (weights = row-normalised off-diagonal
   of the passed adjacency, i.e. a spatial lag à la spatial econometrics),
   and a linear month trend, with ``Ridge(alpha=1.0)``.

Both forecasters roll forward **autoregressively**: each predicted month is
appended to the (log-space) series and becomes an input for the next step,
out to ``max(config.FORECAST_HORIZONS)`` months; predictions are returned
in AZN/m² price space.

Torch is OPTIONAL: the module imports cleanly without it (``HAS_TORCH``),
and ``make_forecaster`` silently falls back to the ridge model.

Expected input: the complete cells × months panel produced by
``bakuml.spatial.grid.complete_panel`` (one row per (h3, month), missing
cell-months present with NaN prices), turned into a dense tensor by
``panel_tensor``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bakuml import config

try:  # torch is an optional dependency for this module
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - exercised via monkeypatch in tests
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

#: Sliding-window length (months) fed to the temporal convolutions.
WINDOW = 12


# ---------------------------------------------------------------------------
# Panel tensor
# ---------------------------------------------------------------------------


def panel_tensor(
    panel_complete: pd.DataFrame,
    months: list[str],
    cells: list[str],
    value_col: str = config.TARGET_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """Densify a complete (h3, month) panel into a [T, N] tensor.

    Parameters
    ----------
    panel_complete:
        One row per (h3, month) with `value_col`; NaN where the cell-month
        was unobserved (as produced by ``bakuml.spatial.grid.complete_panel``).
    months, cells:
        Row / column order of the output. Cell-months absent from the frame
        come out as NaN (unobserved).

    Returns
    -------
    (values, mask):
        ``values`` — float32 array of shape (T, N) with NaN where
        unobserved; ``mask`` — bool array of the same shape, True where a
        finite value is present.
    """
    wide = panel_complete.pivot(index="month", columns="h3", values=value_col)
    wide = wide.reindex(index=list(months), columns=list(cells))
    values = wide.to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    return values, mask


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
# ---------------------------------------------------------------------------


def _log_price_matrices(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-transform and densify a [T, N] price matrix.

    Returns ``(log_obs, filled, obs)`` where ``log_obs`` is the log price
    with NaN at unobserved entries, ``filled`` is ``log_obs`` forward-filled
    along time (leading gaps take the cell's median log price; cells never
    observed take the global median), and ``obs`` is the effective
    observation mask (input mask restricted to finite positive prices).
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must be 2-D arrays of equal shape")
    obs = mask & np.isfinite(values) & (values > 0)
    if not obs.any():
        raise ValueError("panel contains no observed positive prices")
    log_obs = np.full(values.shape, np.nan)
    log_obs[obs] = np.log(values[obs])
    # copy=True: pandas 3 may hand back a read-only view, and we write holes below
    filled = pd.DataFrame(log_obs).ffill().to_numpy(copy=True)
    with warnings.catch_warnings():  # nanmedian warns on all-NaN columns
        warnings.simplefilter("ignore", RuntimeWarning)
        col_median = np.nanmedian(log_obs, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, np.nanmedian(log_obs))
    holes = ~np.isfinite(filled)
    filled[holes] = np.broadcast_to(col_median, filled.shape)[holes]
    return log_obs, filled, obs


def _neighbour_weights(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised neighbour weights from an adjacency matrix.

    The diagonal (self loop of the normalised adjacency) is removed so the
    spatial lag excludes the cell itself; rows are renormalised to sum to 1.
    Returns ``(weights, isolated)`` where ``isolated`` flags cells without
    any in-study neighbour (their spatial lag falls back to the own value).
    """
    w = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(w, 0.0)
    row_sum = w.sum(axis=1)
    isolated = row_sum <= 0
    safe = np.where(isolated, 1.0, row_sum)
    return w / safe[:, None], isolated


def _rollout_frame(
    pred_rows: list[np.ndarray],
    horizons: list[int],
    cells: list[str],
    mu: float,
    sigma: float,
) -> pd.DataFrame:
    """Assemble the long forecast frame [h3, horizon_months, y_pred].

    ``pred_rows[s]`` holds the (normalised) log-price prediction for rollout
    step ``s+1``; prices are mapped back to AZN/m² via ``exp(mu + sigma·z)``.
    """
    parts = []
    for h in horizons:
        z = np.asarray(pred_rows[h - 1], dtype=np.float64)
        parts.append(
            pd.DataFrame(
                {
                    "h3": list(cells),
                    "horizon_months": np.full(len(cells), h, dtype=np.int64),
                    "y_pred": np.exp(mu + sigma * z),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _check_horizons(horizons) -> list[int]:
    horizons = [int(h) for h in horizons]
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be a non-empty sequence of months >= 1")
    return horizons


# ---------------------------------------------------------------------------
# STGCN network (defined only when torch is importable)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _GatedTemporalConv(torch.nn.Module):
        """Gated 1-D convolution over the time axis: ``tanh(P) ⊙ σ(Q)``.

        Input/output layout is ``[batch, channels, time, nodes]``; the kernel
        spans ``ks`` time steps and a single node, shrinking the time axis by
        ``ks - 1`` (no padding — the model is strictly causal because every
        window ends before the target month).
        """

        def __init__(self, c_in: int, c_out: int, ks: int) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(c_in, 2 * c_out, kernel_size=(ks, 1))

        def forward(self, x):  # [B, C, T, N] -> [B, c_out, T-ks+1, N]
            p, q = self.conv(x).chunk(2, dim=1)
            return torch.tanh(p) * torch.sigmoid(q)

    class _STGCNNet(torch.nn.Module):
        """One ST block + readout, predicting next-month values per node.

        temporal conv → graph conv (Â X W₁, ReLU) → temporal conv →
        conv collapsing the remaining time axis to one step. The output is
        an *increment* added to the window's last value (skip connection),
        so the untrained network already equals the persistence forecast.
        """

        def __init__(self, a_hat, hidden: int, ks: int, window: int) -> None:
            super().__init__()
            t_out = window - 2 * (ks - 1)
            if t_out < 1:
                raise ValueError(
                    f"window={window} too short for two temporal convs of kernel {ks}"
                )
            self.register_buffer("a_hat", a_hat)  # [N, N], float32
            self.temporal_in = _GatedTemporalConv(1, hidden, ks)
            self.graph_theta = torch.nn.Linear(hidden, hidden)  # W1
            self.temporal_out = _GatedTemporalConv(hidden, hidden, ks)
            self.readout = torch.nn.Conv2d(hidden, 1, kernel_size=(t_out, 1))

        def forward(self, x):  # x: [B, 1, W, N] -> [B, N]
            h = self.temporal_in(x)  # [B, H, T1, N]
            h = torch.einsum("bctn,nm->bctm", h, self.a_hat)  # Â X (spatial mix)
            h = torch.relu(self.graph_theta(h.permute(0, 2, 3, 1)))  # · W1
            h = h.permute(0, 3, 1, 2)  # back to [B, H, T1, N]
            h = self.temporal_out(h)  # [B, H, T2, N]
            delta = self.readout(h)[:, 0, 0, :]  # [B, N]
            return x[:, 0, -1, :] + delta  # persistence + learned increment


# ---------------------------------------------------------------------------
# STGCN forecaster
# ---------------------------------------------------------------------------


class STGCNForecaster:
    """Spatio-temporal graph convolutional forecaster over the H3 panel.

    Parameters
    ----------
    adjacency:
        Symmetrically normalised adjacency ``D^{-1/2}(A+I)D^{-1/2}`` over
        `cells` (see ``bakuml.spatial.graph.build_adjacency``), shape (N, N).
    cells, months:
        Column / row labels of the panel tensor the model will be fit on.
    hidden, ks, seed:
        Channel width of the ST block, temporal kernel size, RNG seed.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        cells: list[str],
        months: list[str],
        *,
        hidden: int = 32,
        ks: int = 3,
        seed: int = 0,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "STGCNForecaster requires PyTorch, which is not installed. "
                "Install torch, or use make_forecaster(...) which falls back "
                "to SpatialLagRidgeForecaster."
            )
        adjacency = np.asarray(adjacency, dtype=np.float64)
        if adjacency.shape != (len(cells), len(cells)):
            raise ValueError(
                f"adjacency shape {adjacency.shape} does not match {len(cells)} cells"
            )
        self.adjacency = adjacency
        self.cells = list(cells)
        self.months = list(months)
        self.hidden = int(hidden)
        self.ks = int(ks)
        self.seed = int(seed)
        # Shrink the window on very short panels; two ks-kernels must fit.
        self.window = min(WINDOW, len(self.months) - 1)
        min_window = 2 * (self.ks - 1) + 1
        if self.window < min_window:
            raise ValueError(
                f"need at least {min_window + 1} months for ks={self.ks}; "
                f"got {len(self.months)}"
            )
        torch.manual_seed(self.seed)  # deterministic weight init
        np.random.seed(self.seed)
        self._net = _STGCNNet(
            torch.from_numpy(adjacency.astype(np.float32)),
            self.hidden,
            self.ks,
            self.window,
        )
        #: Per-epoch masked-MSE training loss, populated by ``fit``.
        self.history_: list[float] = []
        self._fitted = False

    # -- training ----------------------------------------------------------

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        epochs: int = 60,
        lr: float = 1e-2,
        verbose: bool = False,
    ) -> "STGCNForecaster":
        """Train on the [T, N] price tensor with Adam on masked MSE.

        `values` are raw AZN/m² prices (NaN where unobserved), `mask` the
        observation indicator; both ordered like `months` × `cells`.
        Only observed targets contribute to the loss.
        """
        values = np.asarray(values, dtype=np.float64)
        expected = (len(self.months), len(self.cells))
        if values.shape != expected:
            raise ValueError(f"values shape {values.shape}, expected {expected}")
        log_obs, filled, obs = _log_price_matrices(values, mask)
        mu = float(log_obs[obs].mean())
        sigma = float(log_obs[obs].std())
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        z_filled = (filled - mu) / sigma
        z_target = (log_obs - mu) / sigma

        t_total, w = values.shape[0], self.window
        # Train only where the window tail (month t-1) is genuinely observed:
        # in sparse panels a forward-filled tail makes the "one-month"
        # increment actually span a multi-month gap, and the skip connection
        # would learn gap-recovery growth as per-step drift. Fall back to any
        # observed target only if the strict pairing yields nothing.
        xs, ys, ms = [], [], []
        for require_fresh_tail in (True, False):
            for t in range(w, t_total):  # window [t-w, t) predicts month t
                target_mask = obs[t] & obs[t - 1] if require_fresh_tail else obs[t]
                if not target_mask.any():
                    continue
                xs.append(z_filled[t - w : t])
                ys.append(np.where(target_mask, z_target[t], 0.0))
                ms.append(target_mask)
            if xs:
                break
        if not xs:
            raise ValueError(
                f"no training windows: need > {w} months with observed targets"
            )
        x = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(1)  # [B,1,W,N]
        y = torch.tensor(np.stack(ys), dtype=torch.float32)  # [B, N]
        m = torch.tensor(np.stack(ms), dtype=torch.float32)  # [B, N]

        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        self.history_ = []
        self._net.train()
        for epoch in range(int(epochs)):
            opt.zero_grad()
            pred = self._net(x)
            loss = ((pred - y).pow(2) * m).sum() / m.sum()
            loss.backward()
            opt.step()
            self.history_.append(float(loss.detach()))
            if verbose and (epoch % 10 == 0 or epoch == int(epochs) - 1):
                print(f"[stgcn] epoch {epoch:3d} masked mse {self.history_[-1]:.5f}")
        self._net.eval()

        self._mu, self._sigma = mu, sigma
        self._z_filled = z_filled
        self._fitted = True
        return self

    # -- forecasting ---------------------------------------------------------

    def forecast(
        self, horizons: tuple[int, ...] = config.FORECAST_HORIZONS
    ) -> pd.DataFrame:
        """Autoregressive rollout from the last panel month.

        Each predicted month is appended to the normalised log series and
        feeds the next step's input window. Returns the long frame
        ``[h3, horizon_months, y_pred]`` with prices back in AZN/m².
        """
        if not self._fitted:
            raise RuntimeError("call fit() before forecast()")
        horizons = _check_horizons(horizons)
        seq = self._z_filled.copy()
        preds: list[np.ndarray] = []
        self._net.eval()
        with torch.no_grad():
            for _ in range(max(horizons)):
                window = torch.tensor(seq[-self.window :], dtype=torch.float32)
                nxt = self._net(window[None, None])[0].numpy().astype(np.float64)
                preds.append(nxt)
                seq = np.vstack([seq, nxt[None, :]])
        return _rollout_frame(preds, horizons, self.cells, self._mu, self._sigma)


# ---------------------------------------------------------------------------
# Spatial-lag ridge fallback / benchmark
# ---------------------------------------------------------------------------


class SpatialLagRidgeForecaster:
    """Ridge regression with own and spatial-lag features; STGCN's fallback.

    For every (cell, t) with an observed price, log price is regressed on:

    * the cell's own log price at t-1, t-2, t-3 (forward-filled series),
    * the adjacency-weighted neighbour-mean log price (self excluded) at
      t-1, t-2, t-3 — the classic spatial-lag term,
    * the scaled month index t / T (linear citywide trend, extrapolated
      past T during the rollout).

    ``hidden`` and ``ks`` are accepted for interface parity with
    ``STGCNForecaster`` and ignored; the closed-form ridge fit also ignores
    ``epochs``/``lr``. The model is deterministic, so ``seed`` is stored
    only for interface parity.
    """

    N_LAGS = 3

    def __init__(
        self,
        adjacency: np.ndarray,
        cells: list[str],
        months: list[str],
        *,
        hidden: int = 32,
        ks: int = 3,
        seed: int = 0,
    ) -> None:
        adjacency = np.asarray(adjacency, dtype=np.float64)
        if adjacency.shape != (len(cells), len(cells)):
            raise ValueError(
                f"adjacency shape {adjacency.shape} does not match {len(cells)} cells"
            )
        if len(months) <= self.N_LAGS:
            raise ValueError(f"need more than {self.N_LAGS} months to fit lags")
        self.adjacency = adjacency
        self.cells = list(cells)
        self.months = list(months)
        self.seed = int(seed)
        self._w_nb, self._isolated = _neighbour_weights(adjacency)
        self._fitted = False

    # -- feature construction ----------------------------------------------

    def _neighbour_mean(self, rows: np.ndarray) -> np.ndarray:
        """Adjacency-weighted neighbour mean of [k, N] rows (self excluded).

        Isolated cells (no in-study neighbour) fall back to their own value
        so the feature stays informative instead of collapsing to zero.
        """
        nb = rows @ self._w_nb.T
        if self._isolated.any():
            nb[:, self._isolated] = rows[:, self._isolated]
        return nb

    def _features_at(self, seq: np.ndarray, t_scaled: float) -> np.ndarray:
        """Feature matrix [N, 7] for the month following the tail of `seq`."""
        lags = seq[-self.N_LAGS :][::-1]  # rows: t-1, t-2, t-3
        nb_lags = self._neighbour_mean(lags)
        n = seq.shape[1]
        time_col = np.full((n, 1), t_scaled)
        return np.hstack([lags.T, nb_lags.T, time_col])

    # -- training ------------------------------------------------------------

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        epochs: int = 60,
        lr: float = 1e-2,
        verbose: bool = False,
    ) -> "SpatialLagRidgeForecaster":
        """Fit the persistence-anchored increment model.

        The one-month log-price increment is decomposed as::

            log p_t - log p_{t-1}  =  drift  +  Ridge(gap features)

        * ``drift`` is the *median* of genuinely consecutive one-month
          increments (cell observed at both t-1 and t) - a robust market
          trend that cannot be inflated by forward-filled gaps.
        * The Ridge (alpha=1.0, no intercept) learns spatial mean-reversion
          from level-free gap features (each own / neighbour lag minus the
          own 1-month lag); it captures appreciation diffusing between
          neighbours without carrying any drift of its own.

        This structure was chosen over a plain level regression after two
        sparse-panel failure modes surfaced: stale forward-filled lags teach
        a level model to absorb multi-month gap-recovery growth into its
        per-step drift, and an unconstrained intercept + time term
        extrapolates out of range - both of which an autoregressive rollout
        compounds into wildly inflated forecasts.

        `epochs`, `lr` and `verbose` are accepted for interface parity with
        the STGCN and ignored (the ridge solution is closed-form).
        """
        values = np.asarray(values, dtype=np.float64)
        expected = (len(self.months), len(self.cells))
        if values.shape != expected:
            raise ValueError(f"values shape {values.shape}, expected {expected}")
        log_obs, filled, obs = _log_price_matrices(values, mask)

        t_total = values.shape[0]
        feats, targets = [], []
        # Require a genuinely consecutive (t-1, t) observation pair; fall
        # back to any observed target only for degenerate panels with none.
        for require_fresh in (True, False):
            for t in range(self.N_LAGS, t_total):
                target_mask = obs[t] & obs[t - 1] if require_fresh else obs[t]
                if not target_mask.any():
                    continue
                f = self._gap_features(self._features_at(filled[:t], t / t_total))
                own_lag1 = filled[t - 1]
                feats.append(f[target_mask])
                targets.append(log_obs[t, target_mask] - own_lag1[target_mask])
            if feats:
                break
        if not feats:
            raise ValueError("no observed targets after the first N_LAGS months")
        increments = np.concatenate(targets)
        self._drift = float(np.median(increments))
        self._model = Ridge(alpha=1.0, fit_intercept=False).fit(
            np.vstack(feats), increments - self._drift
        )
        self._filled = filled
        self._fitted = True
        return self

    @staticmethod
    def _gap_features(f: np.ndarray) -> np.ndarray:
        """Level-free gaps: [own2-own1, own3-own1, nb1-own1, nb2-own1,
        nb3-own1]. The raw own-lag-1 column and the time column are dropped
        (the former is the persistence anchor, the latter would extrapolate
        out of the training range during rollout)."""
        return f[:, 1:6] - f[:, 0:1]

    # -- forecasting -----------------------------------------------------------

    def forecast(
        self, horizons: tuple[int, ...] = config.FORECAST_HORIZONS
    ) -> pd.DataFrame:
        """Autoregressive rollout; same output contract as the STGCN."""
        if not self._fitted:
            raise RuntimeError("call fit() before forecast()")
        horizons = _check_horizons(horizons)
        t_total = len(self.months)
        seq = self._filled.copy()
        preds: list[np.ndarray] = []
        for step in range(max(horizons)):
            f = self._features_at(seq, (t_total + step) / t_total)
            # persistence anchor + market drift + spatial mean-reversion
            row = f[:, 0] + self._drift + self._model.predict(self._gap_features(f))
            preds.append(row)
            seq = np.vstack([seq, row[None, :]])
        # ridge works in raw log space: identity de-normalisation (mu=0, sigma=1)
        return _rollout_frame(preds, horizons, self.cells, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_forecaster(
    adjacency: np.ndarray,
    cells: list[str],
    months: list[str],
    *,
    prefer: str = "stgcn",
    seed: int = 0,
):
    """Build the preferred forecaster, degrading gracefully without torch.

    ``prefer="stgcn"`` returns an ``STGCNForecaster`` when torch is
    importable and the ridge fallback otherwise; ``prefer="ridge"`` always
    returns ``SpatialLagRidgeForecaster``.
    """
    if prefer not in ("stgcn", "ridge"):
        raise ValueError(f"prefer must be 'stgcn' or 'ridge', got {prefer!r}")
    if prefer == "stgcn" and HAS_TORCH:
        return STGCNForecaster(adjacency, cells, months, seed=seed)
    return SpatialLagRidgeForecaster(adjacency, cells, months, seed=seed)
