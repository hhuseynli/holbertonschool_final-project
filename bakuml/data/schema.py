"""Canonical dataframe schemas shared across the pipeline.

Three core tables flow through the project:

1. LISTINGS - one row per (possibly duplicated) advert, from bina.az/tap.az
   scrapes or the synthetic generator. Produced by `bakuml.data`.
2. PANEL - one row per (h3 cell, month) with aggregated prices and features.
   Produced by `bakuml.spatial.grid` + `bakuml.features.build`.
3. PREDICTIONS - one row per (h3 cell, horizon) with conformal intervals.
   Produced by `bakuml.models` and consumed by the Flask app.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# 1. Listings
# --------------------------------------------------------------------------

LISTING_COLUMNS: dict[str, str] = {
    "listing_id": "str",      # unique per advert (duplicates get their own id)
    "source": "str",          # "synthetic" | "bina.az" | "tap.az"
    "lat": "float64",
    "lon": "float64",
    "price_azn": "float64",
    "area_m2": "float64",
    "price_azn_m2": "float64",
    "rooms": "int64",
    "floor": "int64",
    "building_floors": "int64",
    "building_type": "str",   # "new" | "old"
    "listed_month": "str",    # "YYYY-MM"
    "title": "str",
    "description": "str",
    "image_phash": "str",     # 16-char hex perceptual hash of the cover photo
    "photo_urls": "str",      # JSON array of photo URLs (empty "[]" if unavailable)
    "district": "str",
}

REQUIRED_LISTING_COLUMNS = tuple(LISTING_COLUMNS)

# --------------------------------------------------------------------------
# 2. Cell-month panel
# --------------------------------------------------------------------------

PANEL_KEY_COLUMNS = ("h3", "month")

PANEL_BASE_COLUMNS: dict[str, str] = {
    "h3": "str",
    "month": "str",                    # "YYYY-MM"
    "price_azn_m2_median": "float64",
    "price_azn_m2_mean": "float64",
    "n_listings": "int64",
    "new_share": "float64",            # share of "new" building_type listings
}

# --------------------------------------------------------------------------
# 3. Predictions / app artifacts
# --------------------------------------------------------------------------

PREDICTION_COLUMNS: dict[str, str] = {
    "h3": "str",
    "base_month": "str",       # last observed month the forecast starts from
    "horizon_months": "int64",
    "scenario": "str",         # key into config.SCENARIOS
    "q10": "float64",          # forecast price_azn_m2 quantiles
    "q50": "float64",
    "q90": "float64",
    "appreciation_pct": "float64",   # q50 vs last observed median, in %
    "hotspot_prob": "float64",       # P(appreciation above city median)
}

# Artifact filenames consumed by the Flask app (under config.ARTIFACTS_DIR)
ARTIFACT_FILES = {
    "listings": "listings.parquet",
    "panel": "panel.parquet",
    "cell_features": "cell_features.parquet",
    # Cell polygons, so the app can draw whatever tessellation produced a
    # run without knowing (or refitting) its geometry.
    "cell_geometry": "cell_geometry.geojson",
    "predictions": "predictions.parquet",
    "metrics": "metrics.json",
    "shap": "shap_summary.json",
    "did": "did_results.json",
    "truth": "synthetic_truth.json",
    "zones": "zones.parquet",
}


class SchemaError(ValueError):
    """Raised when a dataframe does not match the expected schema."""


def validate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and lightly coerce) a listings dataframe.

    Returns the validated frame; raises SchemaError on structural problems.
    """
    missing = [c for c in REQUIRED_LISTING_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"listings missing columns: {missing}")
    if df["listing_id"].duplicated().any():
        raise SchemaError("listing_id values must be unique")
    bad_price = df["price_azn_m2"].le(0) | df["price_azn_m2"].isna()
    if bad_price.any():
        raise SchemaError(f"{int(bad_price.sum())} listings have non-positive price_azn_m2")
    if not df["listed_month"].str.fullmatch(r"\d{4}-\d{2}").all():
        raise SchemaError("listed_month must be formatted YYYY-MM")
    return df


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PANEL_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"panel missing columns: {missing}")
    if df.duplicated(subset=list(PANEL_KEY_COLUMNS)).any():
        raise SchemaError("panel has duplicate (h3, month) rows")
    return df


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def month_index(months: list[str]) -> dict[str, int]:
    return {m: i for i, m in enumerate(months)}
