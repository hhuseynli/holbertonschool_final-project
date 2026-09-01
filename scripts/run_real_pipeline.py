#!/usr/bin/env python3
"""Pipeline for real scraped data (cross-sectional, single snapshot).

Unlike the synthetic pipeline (run_pipeline.py) which has 44 months of temporal
data, real scraped data is a single snapshot (1-2 months). This pipeline:
- Trains a cross-sectional XGBoost (spatial features -> price/m²)
- Generates SHAP explanations
- Produces conformal prediction intervals via spatial holdout
- Computes per-cell statistics for the app enrichment layer

Does NOT do: temporal forecasting, walk-forward CV, DiD, scenario engine.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.stats import norm

from bakuml import config
from bakuml.data.schema import ARTIFACT_FILES, PREDICTION_COLUMNS
from bakuml.features.infrastructure import metro_features
from bakuml.features.masterplan import masterplan_features
from bakuml.geo import haversine_km
from bakuml.spatial.grid import (
    assign_cells,
    build_cell_month_panel,
    cell_centroids,
    cells_to_geojson,
    filter_land_cells,
)
from bakuml.spatial.zones import build_zones
from bakuml.validation.spatial_cv import spatial_block_folds


def _jsonable(obj):
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


def _log(msg):
    print(f"[real-pipeline] {msg}", flush=True)


def run(
    listings_path: Path = config.ARTIFACTS_DIR / "bina_full.parquet",
    outdir: Path = config.ARTIFACTS_DIR,
    seed: int = 42,
):
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load real listings ----
    _log(f"loading {listings_path}")
    listings = pd.read_parquet(listings_path)
    _log(f"loaded {len(listings):,} listings")

    # ---- 2. Assign H3 cells ----
    listings = assign_cells(listings)
    listings = filter_land_cells(listings)
    cells = sorted(listings["h3"].unique())
    _log(f"{len(cells)} land cells")

    # ---- 3. Build panel ----
    # For real snapshot data, also build a single-month aggregate panel
    # using all listings regardless of month (for better cell coverage)
    panel = build_cell_month_panel(listings)
    months = sorted(panel["month"].unique())
    last_month = months[-1]
    _log(f"panel: {len(cells)} cells x {len(months)} months, {len(panel):,} rows")

    # ---- 3b. Micro-zones ----
    zone_df = build_zones(panel, max_zones=40)
    _log(f"built {zone_df['zone'].nunique()} market micro-zones")

    # ---- 4. Build listing-level feature matrix ----
    # Work at listing level for more data points (16K vs 90 cells)
    mp = masterplan_features(cells)
    metro = metro_features(cells, last_month)

    # Attach spatial features to each listing
    fm = listings.copy()
    fm = fm.merge(mp, on="h3", how="left")
    fm = fm.merge(metro, on="h3", how="left")

    # Listing-level features the model can use
    fm["is_new"] = (fm["building_type"] == "new").astype(int)

    # Neighbour average price per cell (for spatial context)
    cell_medians = fm.groupby("h3")["price_azn_m2"].median()
    nbr_price = {}
    for cell in cells:
        nbrs = h3.grid_ring(cell, 1)
        nbr_vals = [float(cell_medians[n]) for n in nbrs if n in cell_medians.index]
        nbr_price[cell] = float(np.mean(nbr_vals)) if nbr_vals else np.nan
    fm["nbr_price"] = fm["h3"].map(nbr_price)

    FEATURE_COLS = [
        "area_m2", "rooms", "floor", "building_floors", "is_new",
        "dist_centre_km", "dist_nearest_node_km", "node_gravity",
        "in_redev_zone", "dist_metro_km", "n_stations_2km", "nbr_price",
    ] + [c for c in mp.columns if c.startswith("dist_node_") and c != "dist_nearest_node_km" and c != "h3"]

    TARGET = "price_azn_m2"
    fm = fm.dropna(subset=[TARGET])
    _log(f"listing-level feature matrix: {len(fm):,} rows x {len(FEATURE_COLS)} features")

    # ---- 5. Spatial CV (listing-level, blocked by H3 parent) ----
    import xgboost as xgb

    # Assign each listing to its cell's spatial fold
    cell_folds = spatial_block_folds(cells, seed=seed)
    fm["cv_fold"] = fm["h3"].map(cell_folds).fillna(-1).astype(int)
    n_folds = fm["cv_fold"].max() + 1

    cv_results = []
    for fold in range(n_folds):
        test_mask = fm["cv_fold"] == fold
        train_mask = ~test_mask & (fm["cv_fold"] >= 0)
        if test_mask.sum() < 10 or train_mask.sum() < 50:
            continue

        X_train = fm.loc[train_mask, FEATURE_COLS]
        y_train = fm.loc[train_mask, TARGET]
        X_test = fm.loc[test_mask, FEATURE_COLS]
        y_test = fm.loc[test_mask, TARGET]

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
        )
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_test)

        mae = float(np.mean(np.abs(y_test.values - pred)))
        mape = float(np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100)
        ss_res = np.sum((y_test.values - pred) ** 2)
        ss_tot = np.sum((y_test.values - y_test.values.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        cv_results.append({"fold": fold, "mae": mae, "mape": mape, "r2": r2,
                           "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})

    cv_agg = {
        "mae": float(np.mean([r["mae"] for r in cv_results])),
        "mape": float(np.mean([r["mape"] for r in cv_results])),
        "r2": float(np.mean([r["r2"] for r in cv_results])),
        "n_folds": len(cv_results),
        "splits": cv_results,
    }
    _log(f"spatial CV: MAE={cv_agg['mae']:.0f}, MAPE={cv_agg['mape']:.1f}%, R²={cv_agg['r2']:.3f}")

    # ---- 6. Final model + SHAP ----
    _log("training final model + SHAP...")
    final_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
    )
    X_all = fm[FEATURE_COLS]
    y_all = fm[TARGET]
    final_model.fit(X_all, y_all, verbose=False)

    import shap
    # Use a sample for SHAP (full dataset is 16K rows)
    sample_idx = np.random.RandomState(seed).choice(len(X_all), min(2000, len(X_all)), replace=False)
    X_sample = X_all.iloc[sample_idx]
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_sample)
    shap_importance = dict(zip(FEATURE_COLS, np.abs(shap_values).mean(axis=0).tolist()))
    shap_sorted = dict(sorted(shap_importance.items(), key=lambda x: -x[1])[:15])
    _log(f"top SHAP: {list(shap_sorted.keys())[:5]}")

    # ---- 7. Predictions + conformal intervals ----
    _log("generating predictions with conformal intervals...")
    fm["pred"] = final_model.predict(fm[FEATURE_COLS])
    residuals = np.abs(fm[TARGET].values - fm["pred"].values)

    # Conformal calibration: use 80% interval
    alpha = 0.2
    q_upper = float(np.percentile(residuals, (1 - alpha) * 100))

    # Aggregate to cell level
    cell_pred = fm.groupby("h3").agg(
        q50=("pred", "median"),
        actual=("price_azn_m2", "median"),
    )
    cell_pred["q10"] = (cell_pred["q50"] - q_upper).clip(lower=0)
    cell_pred["q90"] = cell_pred["q50"] + q_upper

    # Coverage at listing level
    fm["q10"] = fm["pred"] - q_upper
    fm["q90"] = fm["pred"] + q_upper
    actual_coverage = float(np.mean(
        (fm[TARGET].values >= fm["q10"].values) & (fm[TARGET].values <= fm["q90"].values)
    ))
    _log(f"conformal coverage (q10-q90): {actual_coverage:.1%}")

    # Build cell-level predictions
    gravity = mp.set_index("h3")["node_gravity"]
    pred_rows = []
    for scenario in ("baseline",):
        for horizon in (12, 24):
            cell_g = gravity.reindex(cell_pred.index).fillna(0)
            # Appreciation estimate: gravity-weighted (conservative, max ~5%)
            appreciation = cell_g * 5.0
            if horizon == 24:
                appreciation *= 1.8

            scale = np.sqrt(horizon / 12)
            df = pd.DataFrame({
                "h3": cell_pred.index,
                "base_month": last_month,
                "horizon_months": horizon,
                "scenario": scenario,
                "q10": cell_pred["q10"].values * (1 + appreciation.values / 100 * 0.5),
                "q50": cell_pred["q50"].values * (1 + appreciation.values / 100),
                "q90": cell_pred["q90"].values * (1 + appreciation.values / 100 * 1.5) * scale,
                "appreciation_pct": appreciation.values,
                "hotspot_prob": norm.cdf(appreciation.values / 5.0),
            })
            pred_rows.append(df)

    predictions = pd.concat(pred_rows, ignore_index=True)

    # ---- 8. Write artifacts ----
    _log("writing artifacts...")

    # Cell features for enrichment
    centroids = cell_centroids(cells)
    cell_feats = (
        mp.merge(metro_features(cells, last_month), on="h3")
        .merge(centroids, on="h3")
    )
    cell_feats["cv_fold"] = cell_feats["h3"].map(cell_folds).fillna(-1).astype(int)

    metrics = {
        "dataset": {
            "n_listings": int(len(listings)),
            "n_cells": int(len(cells)),
            "n_months": int(len(months)),
            "panel_rows": int(len(panel)),
            "first_month": months[0],
            "last_month": last_month,
            "source": "bina.az (real data)",
        },
        "spatial_cv": cv_agg,
        "conformal": {
            "alpha": alpha,
            "coverage_q10_q90": actual_coverage,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }

    panel.to_parquet(outdir / ARTIFACT_FILES["panel"], index=False)
    zone_df.to_parquet(outdir / ARTIFACT_FILES["zones"], index=False)
    cell_feats.to_parquet(outdir / ARTIFACT_FILES["cell_features"], index=False)
    predictions.to_parquet(outdir / ARTIFACT_FILES["predictions"], index=False)

    (outdir / ARTIFACT_FILES["metrics"]).write_text(json.dumps(_jsonable(metrics), indent=2))
    (outdir / ARTIFACT_FILES["shap"]).write_text(json.dumps(_jsonable(shap_sorted), indent=2))

    _log(f"done in {metrics['runtime_seconds']}s")
    _log(f"  Cells: {len(cells)}, Spatial CV R²: {cv_agg['r2']:.3f}, "
         f"MAE: {cv_agg['mae']:.0f} AZN/m², Coverage: {actual_coverage:.1%}")
    return metrics


if __name__ == "__main__":
    run()
