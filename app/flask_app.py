"""Flask application — bina.az-style marketplace with ML analytics plugins."""

from __future__ import annotations

import math
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

from bakuml import config
from bakuml.data.schema import LISTING_COLUMNS
from bakuml.viz import METRIC_LABELS, hex_layer_geojson, load_artifacts

from app.enrichment import ListingEnricher

ARTIFACTS_DIR = config.ARTIFACTS_DIR
PER_PAGE = 24


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load artifacts and listings once at startup
    enricher = ListingEnricher(ARTIFACTS_DIR)
    arts = enricher.arts
    listings_df = _load_listings(ARTIFACTS_DIR)
    listings_raw = _df_to_listings(listings_df)

    # Enrich all listings with analytics
    listings = [enricher.enrich(l) for l in listings_raw]

    # Build indexes
    listings_by_id = {l["listing_id"]: l for l in listings}
    districts = sorted({l.get("district", "") for l in listings if l.get("district")})
    price_range = (
        float(listings_df["price_azn"].min()) if len(listings_df) else 0,
        float(listings_df["price_azn"].max()) if len(listings_df) else 1000000,
    )
    area_range = (
        float(listings_df["area_m2"].min()) if len(listings_df) else 0,
        float(listings_df["area_m2"].max()) if len(listings_df) else 300,
    )

    # --- Page routes ---

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            districts=districts,
            price_min=int(price_range[0]),
            price_max=int(price_range[1]),
            area_min=int(area_range[0]),
            area_max=int(area_range[1]),
            total=len(listings),
        )

    @app.route("/listing/<listing_id>")
    def detail(listing_id):
        listing = listings_by_id.get(listing_id)
        if not listing:
            abort(404)
        return render_template("detail.html", listing=listing)

    @app.route("/map")
    def map_view():
        return render_template(
            "map.html",
            metrics=METRIC_LABELS,
            scenarios=list(config.SCENARIOS.keys()),
        )

    # --- API routes ---

    @app.route("/api/listings")
    def api_listings():
        filtered = list(listings)

        # Filters
        district = request.args.get("district")
        if district:
            filtered = [l for l in filtered if l.get("district") == district]

        rooms = request.args.get("rooms")
        if rooms:
            rooms_int = int(rooms)
            filtered = [l for l in filtered if l.get("rooms") == rooms_int]

        building_type = request.args.get("building_type")
        if building_type:
            filtered = [l for l in filtered if l.get("building_type") == building_type]

        price_min = request.args.get("price_min", type=float)
        if price_min is not None:
            filtered = [l for l in filtered if (l.get("price_azn") or 0) >= price_min]

        price_max = request.args.get("price_max", type=float)
        if price_max is not None:
            filtered = [l for l in filtered if (l.get("price_azn") or 0) <= price_max]

        area_min = request.args.get("area_min", type=float)
        if area_min is not None:
            filtered = [l for l in filtered if (l.get("area_m2") or 0) >= area_min]

        area_max = request.args.get("area_max", type=float)
        if area_max is not None:
            filtered = [l for l in filtered if (l.get("area_m2") or 0) <= area_max]

        # Sort
        sort = request.args.get("sort", "score_desc")
        if sort == "price_asc":
            filtered.sort(key=lambda l: l.get("price_azn") or 0)
        elif sort == "price_desc":
            filtered.sort(key=lambda l: l.get("price_azn") or 0, reverse=True)
        elif sort == "price_m2_asc":
            filtered.sort(key=lambda l: l.get("price_azn_m2") or 0)
        elif sort == "price_m2_desc":
            filtered.sort(key=lambda l: l.get("price_azn_m2") or 0, reverse=True)
        elif sort == "area_desc":
            filtered.sort(key=lambda l: l.get("area_m2") or 0, reverse=True)
        elif sort == "score_desc":
            filtered.sort(
                key=lambda l: (l.get("analytics") or {}).get("investment_score", 0),
                reverse=True,
            )
        elif sort == "newest":
            filtered.sort(key=lambda l: l.get("listed_month") or "", reverse=True)

        # Paginate
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", PER_PAGE, type=int)
        per_page = min(per_page, 100)
        total = len(filtered)
        total_pages = max(1, math.ceil(total / per_page))
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        page_items = filtered[start : start + per_page]

        return jsonify({
            "listings": _serialize_listings(page_items),
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        })

    @app.route("/api/listing/<listing_id>")
    def api_listing(listing_id):
        listing = listings_by_id.get(listing_id)
        if not listing:
            return jsonify({"error": "not found"}), 404
        return jsonify(_serialize_listing(listing))

    @app.route("/api/hexlayer")
    def api_hexlayer():
        scenario = request.args.get("scenario", "baseline")
        horizon = request.args.get("horizon", 12, type=int)
        metric = request.args.get("metric", "appreciation_pct")

        predictions = arts.get("predictions")
        panel = arts.get("panel")
        if predictions is None or panel is None:
            return jsonify({"error": "artifacts not loaded"}), 500

        geojson, cmap = hex_layer_geojson(
            predictions, panel, scenario=scenario, horizon=horizon, metric=metric,
            geometry=arts.get("cell_geometry"),
        )
        return jsonify({
            "geojson": geojson,
            "vmin": cmap.vmin,
            "vmax": cmap.vmax,
            "colors": cmap.colors,
            "caption": cmap.caption,
        })

    @app.route("/api/cell/<h3_id>")
    def api_cell(h3_id):
        scenario = request.args.get("scenario", "baseline")
        return jsonify(enricher.cell_analytics(h3_id, scenario))

    @app.route("/api/metro_stations")
    def api_metro():
        stations = []
        for s in config.METRO_STATIONS:
            stations.append({
                "name": s.name,
                "lat": s.lat,
                "lon": s.lon,
                "line": s.line or "",
                "opened": s.opened,
                "simulated_open": s.simulated_open,
            })
        return jsonify(stations)

    @app.route("/api/config")
    def api_config():
        return jsonify({
            "centre": list(config.CITY_CENTRE),
            "bbox": list(config.BBOX),
            "scenarios": list(config.SCENARIOS.keys()),
            "metrics": METRIC_LABELS,
            "polycentric_nodes": [
                {"name": name, "lat": coords[0], "lon": coords[1]}
                for name, coords in config.POLYCENTRIC_NODES.items()
            ],
        })

    return app


def _load_listings(artifacts_dir: Path) -> pd.DataFrame:
    """Load real scraped listings from local parquet.

    Looks for bina_full.parquet first (full scrape), then bina_scraped.parquet
    (smaller scrape). Falls back to listings.parquet only if no scraped data
    exists.
    """
    import json as _json

    for name in ("bina_full.parquet", "bina_scraped.parquet", "listings.parquet"):
        path = artifacts_dir / name
        if path.is_file():
            df = pd.read_parquet(path)
            # Normalise photo column: scraped data has photo_url (single URL),
            # while the app expects photo_urls (JSON array of URLs).
            if "photo_urls" not in df.columns and "photo_url" in df.columns:
                df["photo_urls"] = df["photo_url"].apply(
                    lambda u: _json.dumps([u]) if pd.notna(u) and u else "[]"
                )
            if "photo_urls" not in df.columns:
                df["photo_urls"] = "[]"
            return df
    return pd.DataFrame(columns=list(LISTING_COLUMNS))


def _df_to_listings(df: pd.DataFrame) -> list[dict]:
    """Convert listings DataFrame to list of dicts with photo URLs."""
    records = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # Convert numpy types to Python types for JSON serialization
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                d[k] = bool(v)
        # Parse photo_urls JSON column into photos list
        import json as _json
        raw = d.pop("photo_urls", "[]") or "[]"
        try:
            d["photos"] = _json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (ValueError, TypeError):
            d["photos"] = []
        records.append(d)
    return records


def _serialize_listing(listing: dict) -> dict:
    """Prepare a single listing for JSON response."""
    safe = {}
    for k, v in listing.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            safe[k] = None
        elif isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, dict):
            safe[k] = _serialize_listing(v)
        else:
            safe[k] = v
    return safe


def _serialize_listings(listings: list[dict]) -> list[dict]:
    return [_serialize_listing(l) for l in listings]


# Allow `flask --app app.flask_app run`
app = create_app()
