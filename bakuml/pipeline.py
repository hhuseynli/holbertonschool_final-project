"""End-to-end orchestration: raw listings -> every app artifact.

Stages (each maps to a module documented in DESIGN.md):

1.  data       - synthetic listings (offline stand-in for the bina.az scrape)
2.  dedup      - broker re-post removal (pHash + TF-IDF)
3.  spatial    - H3 cell assignment, cell x month panel
4.  features   - Master Plan 2040 + infrastructure + leakage-safe lags
5.  validate   - temporal walk-forward + spatial blocked CV (vs naive)
6.  explain    - final XGBoost fit + SHAP price drivers
7.  intervals  - conformalized quantile regression (10/50/90)
8.  forecast   - STGCN (or spatial-lag ridge) 12/24-month rollout,
                 scenario engine on top
9.  causal     - Spatial DiD around the B-04 metro opening
10. artifacts  - parquet/json files consumed by the Streamlit app

The scenario engine composes *estimated* quantities, never planted truth:
the polycentric scenario scales the node-gravity appreciation slope
estimated from the panel's own recent history, and the transit scenario
applies the DiD-estimated ATT to cells around the next hypothetical
purple-line station (B-05, extrapolated one inter-station step beyond B-04).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from bakuml import config
from bakuml.causal.did import (
    did_artifact,
    event_study,
    prepare_did_frame,
    run_spatial_did,
)
from bakuml.data.dedup import dedupe
from bakuml.data.schema import ARTIFACT_FILES, PREDICTION_COLUMNS
from bakuml.data.synthetic import generate_listings
from bakuml.features.build import FEATURE_COLS, build_feature_matrix
from bakuml.features.infrastructure import metro_features
from bakuml.features.masterplan import masterplan_features
from bakuml.geo import haversine_km
from bakuml.models.baseline import (
    evaluate_spatial_cv,
    evaluate_walk_forward,
    shap_summary,
    train_baseline,
)
from bakuml.models.conformal import coverage, fit_cqr, predict_intervals
from bakuml.models.stgcn import make_forecaster, panel_tensor
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.graph import build_adjacency
from bakuml.spatial.grid import (
    assign_cells,
    build_cell_month_panel,
    cells_to_geojson,
    complete_panel,
    filter_land_cells,
)
from bakuml.spatial.tessellation import Tessellation
from bakuml.spatial.zones import build_zones
from bakuml.validation.spatial_cv import spatial_block_folds

# Width of the 1-month conformal interval is scaled by sqrt(horizon) for
# multi-month forecasts - a documented random-walk heuristic.
#: Months withheld from price-driven tessellation fitting (>= the conformal
#: calibration + holdout windows), so the geography stays leakage-free.
_TESS_HOLDOUT_MONTHS = 9

_Z_80 = norm.ppf(0.90)  # q10..q90 spans +/- 1.2816 sigma under normality


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[pipeline] {msg}", flush=True)


def _jsonable(obj):
    """Recursively convert numpy scalars so json.dumps succeeds."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _hypothetical_next_station() -> tuple[float, float]:
    """Extrapolate the purple line one inter-station step beyond B-04.

    Scenario assumption only (documented in the app): the 'transit build-out'
    scenario asks what happens if the line keeps extending at its current
    spacing, and applies the DiD-estimated premium around that point.
    """
    stations = {s.name: s for s in config.METRO_STATIONS}
    b04 = stations[config.DID_TREATMENT_STATION]
    prev = stations["8 Noyabr"]
    return (2 * b04.lat - prev.lat, 2 * b04.lon - prev.lon)


def _node_gravity_slope(panel: pd.DataFrame, gravity: pd.Series) -> float:
    """Estimate (from data, not planted truth) how much extra monthly log
    appreciation a unit of polycentric node gravity has been buying.

    Two-stage estimate over the last 22 months (~the horizon on which the
    Master Plan's polycentric phase-in should be visible): (1) per-cell OLS
    time trend of log median price using *all* months the cell was observed
    (>= 8 required), (2) cross-sectional regression of those trends on
    gravity, weighted by each cell's observation count. Using every month
    (rather than a two-endpoint difference of single-month medians) and
    weighting by coverage keeps the estimate from being noise-dominated.
    Floored at 0.
    """
    months = sorted(panel["month"].unique())
    if len(months) < 14:
        return 0.0
    recent = months[-min(22, len(months) - 1):]
    wide = panel.pivot_table(
        index="h3", columns="month", values="price_azn_m2_median", aggfunc="last"
    ).reindex(columns=recent)
    t = np.arange(len(recent), dtype=float)
    logp = np.log(wide.to_numpy(dtype=float))
    slopes: dict[str, float] = {}
    weights: dict[str, float] = {}
    for i, cell in enumerate(wide.index):
        y = logp[i]
        ok = np.isfinite(y)
        if ok.sum() < 8:
            continue
        slopes[cell] = float(np.polyfit(t[ok], y[ok], 1)[0])
        weights[cell] = float(ok.sum())
    if len(slopes) < 20:
        return 0.0
    s = pd.Series(slopes)
    g = gravity.reindex(s.index)
    ok = g.notna()
    if ok.sum() < 20:
        return 0.0
    w = pd.Series(weights).reindex(s.index)
    # np.polyfit multiplies residuals by w before squaring, so pass sqrt of
    # the observation counts to weight squared residuals by count.
    slope = np.polyfit(
        g[ok].to_numpy(dtype=float),
        s[ok].to_numpy(dtype=float),
        1,
        w=np.sqrt(w[ok].to_numpy(dtype=float)),
    )[0]
    return float(max(slope, 0.0))


def _scenario_adjustment_log(
    scenario: config.Scenario,
    cells: list[str],
    horizon: int,
    gravity: pd.Series,
    gravity_slope: float,
    att_log: float,
    centroids: pd.DataFrame,
) -> pd.Series:
    """Per-cell log-price adjustment applied on top of the base forecast."""
    adj = pd.Series(0.0, index=pd.Index(cells, name="h3"))
    if scenario.polycentric_pull != 1.0:
        adj += (scenario.polycentric_pull - 1.0) * gravity_slope * horizon * gravity
    if scenario.transit_buildout and att_log > 0:
        nlat, nlon = _hypothetical_next_station()
        d = haversine_km(
            centroids["lat"].to_numpy(), centroids["lon"].to_numpy(), nlat, nlon
        )
        near = pd.Series(d, index=centroids["h3"].to_numpy()) <= config.DID_TREATMENT_RADIUS_KM
        # Station assumed to open 6 months out, premium ramping over 3 months
        # (mirrors the DiD ramp actually estimated from history).
        months_active = max(0.0, min(1.0, (horizon - 6) / 3.0))
        adj += att_log * months_active * near.reindex(adj.index).fillna(False).astype(float)
    return adj


def run_pipeline(
    scenarios: tuple[str, ...] = ("baseline", "polycentric", "transit"),
    *,
    fast: bool = False,
    prefer_forecaster: str = "stgcn",
    outdir=None,
    seed: int = config.SYNTHETIC_TRUTH.seed,
    verbose: bool = True,
    tessellation: "Tessellation | str | None" = None,
) -> dict:
    """Run every stage and write the app artifacts. Returns a summary dict.

    ``tessellation`` picks the unit of analysis: a ``Tessellation`` instance,
    one of the names ``tessellation.build`` accepts (``h3``, ``kdtree``,
    ``market``), or ``None`` for ``config.TESSELLATION``. Fitted
    tessellations are trained here on the deduplicated listings, and
    price-driven ones are cut off before the evaluation window so the
    geography never sees prices it will later be scored on.
    """
    t_start = time.time()
    config.ensure_dirs()
    outdir = config.ARTIFACTS_DIR if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    truth = config.SYNTHETIC_TRUTH
    if fast:
        truth = replace(truth, n_listings_per_month=400)
    _log("generating listings (offline synthetic feed)...", verbose)
    listings, truth_info = generate_listings(truth=truth)

    # --------------------------------------------------------------- 2. dedup
    _log(f"dedup over {len(listings):,} listings...", verbose)
    clean, dup_map = dedupe(listings)
    _log(f"removed {len(listings) - len(clean):,} broker re-posts", verbose)

    # --------------------------------------------- 3. unit of analysis + panel
    spec = config.TESSELLATION if tessellation is None else tessellation
    tess = tess_mod.build(spec) if isinstance(spec, str) else spec
    # Price-driven tessellations must not see the evaluation window, so they
    # are cut off well before its start; coordinate-only ones ignore this.
    all_months = sorted(clean["listed_month"].unique())
    cutoff = (
        all_months[-_TESS_HOLDOUT_MONTHS]
        if len(all_months) > _TESS_HOLDOUT_MONTHS else None
    )
    tess.fit(clean, price_cutoff_month=cutoff)
    previous_tess = tess_mod.set_active(tess)
    _log(f"tessellation: {tess.describe()}", verbose)

    clean = assign_cells(clean)
    clean = filter_land_cells(clean)
    _log(f"filtered to {clean['h3'].nunique()} land cells "
         f"(dropped cells centred at sea)", verbose)
    panel = build_cell_month_panel(clean)
    months = sorted(panel["month"].unique())
    cells = sorted(panel["h3"].unique())
    _log(f"panel: {len(cells)} cells x {len(months)} months "
         f"({len(panel):,} observed cell-months)", verbose)

    # --------------------------------------------------------- 3b. micro-zones
    zone_df = build_zones(panel, max_zones=35)
    _log(f"built {zone_df['zone'].nunique()} market micro-zones", verbose)

    # ------------------------------------------------------------ 4. features
    fm = build_feature_matrix(panel)
    _log(f"feature matrix: {len(fm):,} rows x {len(FEATURE_COLS)} features", verbose)

    # ------------------------------------------------------------ 5. validate
    _log("temporal walk-forward evaluation...", verbose)
    wf_metrics = evaluate_walk_forward(fm, FEATURE_COLS, seed=seed)
    _log("spatial blocked CV evaluation...", verbose)
    # nbr_price_prev_month carries (lagged) prices of neighbouring cells;
    # at block boundaries that would hand training rows the held-out
    # region's target level, so the spatial holdout drops cross-cell
    # features entirely.
    scv_metrics = evaluate_spatial_cv(
        fm, FEATURE_COLS, seed=seed, exclude_features=["nbr_price_prev_month"]
    )

    # ------------------------------------------------------------- 6. explain
    _log("final baseline fit + SHAP...", verbose)
    tb = train_baseline(fm, FEATURE_COLS, seed=seed)
    shap_sum = shap_summary(tb, fm)

    # ----------------------------------------------------------- 7. intervals
    # Three-way temporal split: train | calibrate | holdout. Coverage is
    # reported on the holdout months only - measuring it on the calibration
    # months would be an arithmetic identity (CQR widens the band until
    # ~(1-alpha) of calibration points fit), not a validation.
    n_cal, n_holdout = 6, 3
    train_months = months[: -(n_cal + n_holdout)]
    cal_months = months[-(n_cal + n_holdout): -n_holdout]
    holdout_months = months[-n_holdout:]
    _log("conformalized quantile regression...", verbose)
    cm = fit_cqr(
        fm, FEATURE_COLS,
        train_months=train_months, cal_months=cal_months, seed=seed,
    )
    hold_mask = fm["month"].isin(holdout_months)
    hold_iv = predict_intervals(cm, fm.loc[hold_mask])
    holdout_cov = coverage(
        fm.loc[hold_mask, config.TARGET_COL].to_numpy(),
        hold_iv["q10"].to_numpy(), hold_iv["q90"].to_numpy(),
    )
    last_month = months[-1]
    last_mask = fm["month"] == last_month
    last_iv = predict_intervals(cm, fm.loc[last_mask])
    last_iv = last_iv.set_axis(fm.loc[last_mask, "h3"], axis=0)

    # ------------------------------------------------------------ 8. forecast
    _log(f"fitting {prefer_forecaster} forecaster...", verbose)
    panel_c = complete_panel(panel, months)
    adjacency, cell_order = build_adjacency(cells)
    values, mask = panel_tensor(panel_c, months, cell_order)
    forecaster = make_forecaster(
        adjacency, cell_order, months, prefer=prefer_forecaster, seed=seed
    )
    forecaster.fit(values, mask, epochs=25 if fast else 80)
    fc = forecaster.forecast(config.FORECAST_HORIZONS)  # h3, horizon_months, y_pred

    # last observed median per cell (for appreciation)
    last_obs = (
        panel.sort_values("month").groupby("h3")["price_azn_m2_median"].last()
    )

    # ------------------------------------------------------------- 9. causal
    _log("spatial DiD around the treatment station...", verbose)
    did_frame = prepare_did_frame(clean)
    did_res = run_spatial_did(did_frame)
    did_events = event_study(did_frame)
    did_json = did_artifact(did_res, did_events)
    att_log = float(max(did_res.att_log, 0.0))

    # -------------------------------------------------- scenario predictions
    mp = masterplan_features(cells).set_index("h3")
    gravity = mp["node_gravity"]
    gravity_slope = _node_gravity_slope(panel, gravity)
    from bakuml.spatial.grid import cell_centroids  # local import keeps top tidy

    centroids = cell_centroids(cells)

    # relative conformal widths at the last observed month (per-cell where
    # available, citywide median as fallback), scaled by sqrt(horizon).
    rel_lo = ((last_iv["q50"] - last_iv["q10"]) / last_iv["q50"]).clip(lower=0.01)
    rel_hi = ((last_iv["q90"] - last_iv["q50"]) / last_iv["q50"]).clip(lower=0.01)
    rel_lo_med, rel_hi_med = float(rel_lo.median()), float(rel_hi.median())

    pred_rows = []
    for scen_name in scenarios:
        scen = config.SCENARIOS[scen_name]
        for horizon in config.FORECAST_HORIZONS:
            f_h = fc[fc["horizon_months"] == horizon].set_index("h3")["y_pred"]
            f_h = f_h.reindex(cells)
            adj = _scenario_adjustment_log(
                scen, cells, horizon, gravity, gravity_slope, att_log, centroids
            )
            q50 = f_h * np.exp(adj.reindex(f_h.index).fillna(0.0))
            scale = np.sqrt(horizon)
            lo = rel_lo.reindex(q50.index).fillna(rel_lo_med) * scale
            hi = rel_hi.reindex(q50.index).fillna(rel_hi_med) * scale
            q10 = q50 * (1 - lo).clip(lower=0.05)
            q90 = q50 * (1 + hi)
            base = last_obs.reindex(q50.index)
            app = (q50 / base - 1.0) * 100.0
            # hotspot probability: P(appreciation above the citywide median),
            # normal approx with sigma from the (q10, q90) span.
            sigma_app = ((q90 - q10) / base * 100.0) / (2 * _Z_80)
            med_app = float(app.median())
            hotspot = 1.0 - norm.cdf((med_app - app) / sigma_app.clip(lower=1e-6))
            df = pd.DataFrame(
                {
                    "h3": q50.index,
                    "base_month": last_month,
                    "horizon_months": horizon,
                    "scenario": scen_name,
                    "q10": q10.to_numpy(),
                    "q50": q50.to_numpy(),
                    "q90": q90.to_numpy(),
                    "appreciation_pct": app.to_numpy(),
                    "hotspot_prob": np.asarray(hotspot),
                }
            )
            pred_rows.append(df.dropna(subset=["q50"]))
    predictions = pd.concat(pred_rows, ignore_index=True)[list(PREDICTION_COLUMNS)]

    # ---------------------------------------------------------- 10. artifacts
    _log("writing artifacts...", verbose)
    # Same cell list / seed / defaults as evaluate_spatial_cv uses
    # internally, so published fold labels match the evaluation; cells that
    # never reach the feature matrix get fold -1 ("not evaluated").
    folds = spatial_block_folds(sorted(fm["h3"].unique().tolist()), seed=seed)
    cell_feats = (
        mp.reset_index()
        .merge(metro_features(cells, last_month), on="h3")
        .merge(centroids, on="h3")
    )
    cell_feats["cv_fold"] = (
        cell_feats["h3"].map(folds).fillna(-1).astype(int)
    )

    metrics = {
        "tessellation": tess.describe(),
        "dataset": {
            "n_listings_raw": int(len(listings)),
            "n_listings_deduped": int(len(clean)),
            "n_duplicates_removed": int(len(listings) - len(clean)),
            "n_cells": int(len(cells)),
            "n_months": int(len(months)),
            "panel_rows": int(len(panel)),
            "feature_rows": int(len(fm)),
            "first_month": months[0],
            "last_month": last_month,
        },
        "walk_forward": wf_metrics,
        "spatial_cv": scv_metrics,
        "conformal": {
            "alpha": 0.2,
            "calibration_months": cal_months,
            "holdout_months": holdout_months,
            # empirical coverage on months never seen in training/calibration
            "coverage_q10_q90": float(holdout_cov),
        },
        "forecaster": type(forecaster).__name__,
        "scenario_engine": {
            "node_gravity_slope_log_per_month": gravity_slope,
            "did_att_log_used": att_log,
        },
        "runtime_seconds": round(time.time() - t_start, 1),
        "fast_mode": fast,
    }

    # Cell polygons travel with the run: the app must be able to draw a
    # KD-tree or market-region map without refitting the tessellation.
    (outdir / ARTIFACT_FILES["cell_geometry"]).write_text(
        json.dumps(cells_to_geojson(cells, tess=tess))
    )
    clean.to_parquet(outdir / ARTIFACT_FILES["listings"], index=False)
    panel.to_parquet(outdir / ARTIFACT_FILES["panel"], index=False)
    zone_df.to_parquet(outdir / ARTIFACT_FILES["zones"], index=False)
    cell_feats.to_parquet(outdir / ARTIFACT_FILES["cell_features"], index=False)
    predictions.to_parquet(outdir / ARTIFACT_FILES["predictions"], index=False)
    (outdir / ARTIFACT_FILES["metrics"]).write_text(
        json.dumps(_jsonable(metrics), indent=2)
    )
    (outdir / ARTIFACT_FILES["shap"]).write_text(
        json.dumps(_jsonable(shap_sum), indent=2)
    )
    (outdir / ARTIFACT_FILES["did"]).write_text(
        json.dumps(_jsonable(did_json), indent=2)
    )
    (outdir / ARTIFACT_FILES["truth"]).write_text(
        json.dumps(_jsonable(truth_info), indent=2)
    )

    _log(f"done in {metrics['runtime_seconds']}s -> {outdir}", verbose)
    tess_mod.set_active(previous_tess)
    return metrics
