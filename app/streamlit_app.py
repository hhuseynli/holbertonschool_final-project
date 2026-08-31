"""Beyond Price Prediction — the interactive Baku & Absheron demo.

Run with ``make app`` (= ``streamlit run app/streamlit_app.py``) after
building the artifacts with ``make demo`` (fast) or ``make full``.

The app is a thin UI shell: all data plumbing lives in :mod:`bakuml.viz`
(unit-tested, streamlit-free). Layout:

* **Sidebar** — scenario / forecast-horizon / map-metric controls.
* **Map** — H3 hexagon choropleth of the chosen metric on a CartoDB Positron
  base map, plus the study's fixed geography: metro stations coloured by
  line (hollow = not yet open), Master Plan 2040 polycentric nodes (stars),
  and redevelopment zones (translucent circles).
* **Cell detail** — clicking a hexagon reveals its observed price history,
  the forecast fan (q10–q90 band around the q50 path; the calibrated
  1-month conformal width scaled by √horizon), and its static features.
* **Tabs** — SHAP price drivers, the spatial DiD result with its event-study
  plot, leakage-proof validation metrics, and a methodology summary.

Every artifact is optional except ``predictions`` + ``panel``: missing
pieces degrade to an informative note instead of a crash.
"""

from __future__ import annotations

import math
from pathlib import Path

import folium
import h3
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from bakuml import config, viz
from bakuml.data.schema import ARTIFACT_FILES

st.set_page_config(layout="wide", page_title="Beyond Price Prediction - Baku")

_LINE_COLOURS = {"red": "#d32f2f", "green": "#2e7d32", "purple": "#7b1fa2"}


def _gradient_legend(cmap, label: str) -> str:
    """Self-contained HTML colour legend for the hex layer.

    branca's built-in legend (``colormap.add_to(fmap)``) pulls d3 from
    d3js.org at render time, which stalls the map in offline or proxied
    environments — so we sample the colormap into a pure-CSS gradient.
    """
    n = 24
    stops = ", ".join(
        f"{cmap(cmap.vmin + (cmap.vmax - cmap.vmin) * i / (n - 1))} {i / (n - 1):.0%}"
        for i in range(n)
    )
    return (
        f'<div style="font-size:0.8rem;">{label}'
        f'<div style="height:10px;border-radius:5px;'
        f'background:linear-gradient(to right, {stops});"></div>'
        f'<div style="display:flex;justify-content:space-between;">'
        f"<span>{cmap.vmin:,.1f}</span><span>{cmap.vmax:,.1f}</span></div></div>"
    )
_ACCENT = "#1565c0"


# ---------------------------------------------------------------------------
# Artifact loading (cached once per session / artifacts change)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading pipeline artifacts...")
def _load_artifacts(artifacts_dir: str, signature: tuple) -> dict:
    # `signature` (file mtimes) is part of the cache key so a re-run of
    # `make demo`/`make full` is picked up on the next page interaction
    # instead of serving stale (or permanently empty) artifacts.
    return viz.load_artifacts(Path(artifacts_dir))


def _artifact_signature(artifacts_dir: Path) -> tuple:
    sig = []
    for name in sorted(set(ARTIFACT_FILES.values())):
        p = artifacts_dir / name
        if p.is_file():
            sig.append((name, p.stat().st_mtime_ns))
    return tuple(sig)


artifacts = _load_artifacts(
    str(config.ARTIFACTS_DIR), _artifact_signature(config.ARTIFACTS_DIR)
)

if "predictions" not in artifacts or "panel" not in artifacts:
    st.error(
        "**No pipeline artifacts found.** The app needs at least "
        "`predictions.parquet` and `panel.parquet` under "
        f"`{config.ARTIFACTS_DIR}`.\n\n"
        "Build them first:\n\n"
        "```bash\nmake demo   # fast end-to-end run (~2-4 min)\n"
        "make full   # full pipeline with the STGCN forecaster\n```\n"
        "then reload this page."
    )
    st.stop()

predictions: pd.DataFrame = artifacts["predictions"]
panel: pd.DataFrame = artifacts["panel"]
zone_df: pd.DataFrame | None = artifacts.get("zones")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Beyond Price Prediction")
st.sidebar.caption("Spatiotemporal ML for Baku & Absheron — Holberton Group 3")

scenario = st.sidebar.selectbox("Scenario", sorted(predictions["scenario"].unique()))
if scenario in config.SCENARIOS:
    st.sidebar.caption(config.SCENARIOS[scenario].description)

_horizons = sorted(int(h) for h in predictions["horizon_months"].unique())
horizon = st.sidebar.radio("Forecast horizon (months)", _horizons, horizontal=True)

metric = st.sidebar.selectbox(
    "Map metric", list(viz.METRIC_LABELS), format_func=viz.METRIC_LABELS.get
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Map legend**  \n"
    "● metro station (hollow = not yet open)  \n"
    "★ Master Plan 2040 polycentric node  \n"
    "◯ redevelopment zone"
)

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

st.title("Where is Baku appreciating — and why?")

# Use zone layer (merged polygons) when available, hex layer as fallback
_use_zones = zone_df is not None and len(zone_df) > 0
if _use_zones:
    geojson, colormap = viz.zone_layer_geojson(
        predictions, panel, zone_df,
        scenario=scenario, horizon=int(horizon), metric=metric,
        # Cell polygons come from the run's artifact: the app cannot
        # reconstruct a fitted tessellation's geometry.
        geometry=artifacts.get("cell_geometry"),
    )
    _tooltip_fields = ["label", "zone_name"]
    _tooltip_aliases = [viz.METRIC_LABELS[metric], "zone"]
    _layer_name = "zones"
else:
    geojson, colormap = viz.hex_layer_geojson(
        predictions, panel, scenario=scenario, horizon=int(horizon), metric=metric,
        geometry=artifacts.get("cell_geometry"),
    )
    _tooltip_fields = ["label", "h3"]
    _tooltip_aliases = [viz.METRIC_LABELS[metric], "cell"]
    _layer_name = "cells"
layer_cells = {feat["id"] for feat in geojson["features"]}

fmap = folium.Map(
    location=list(config.CITY_CENTRE), zoom_start=11, tiles="OpenStreetMap"
)

folium.GeoJson(
    geojson,
    name=_layer_name,
    style_function=lambda feat: {
        "fillColor": colormap(feat["properties"]["value"]),
        "color": "#546e7a",
        "weight": 0.8 if _use_zones else 0.4,
        "fillOpacity": 0.65,
    },
    highlight_function=lambda feat: {
        "weight": 2.5,
        "color": "#212121",
        "fillOpacity": 0.85,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=_tooltip_fields, aliases=_tooltip_aliases,
    ),
).add_to(fmap)

for station in config.METRO_STATIONS:
    is_open = station.opened is not None
    colour = _LINE_COLOURS.get(station.line, "#455a64")
    status = f"opened {station.opened}" if is_open else "under construction"
    folium.CircleMarker(
        location=(station.lat, station.lon),
        radius=4,
        color=colour,
        weight=2,
        fill=True,
        fill_color=colour,
        fill_opacity=0.9 if is_open else 0.0,
        tooltip=f"{station.name} — {station.line} line, {status}",
    ).add_to(fmap)

for node_name, (nlat, nlon) in config.POLYCENTRIC_NODES.items():
    folium.Marker(
        location=(nlat, nlon),
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:18px;color:#1565c0;'
                'text-shadow:0 0 3px #ffffff;">★</div>'
            ),
            icon_size=(20, 20),
            icon_anchor=(10, 10),
        ),
        tooltip=f"Master Plan 2040 node: {node_name}",
    ).add_to(fmap)

for zone_name, (zlat, zlon, radius_km) in config.REDEVELOPMENT_ZONES.items():
    folium.Circle(
        location=(zlat, zlon),
        radius=radius_km * 1000.0,
        color="#8d6e63",
        weight=1,
        fill=True,
        fill_color="#8d6e63",
        fill_opacity=0.12,
        tooltip=f"Redevelopment zone: {zone_name}",
    ).add_to(fmap)

map_state = st_folium(
    fmap,
    height=560,
    use_container_width=True,
    returned_objects=["last_object_clicked", "last_active_drawing"],
)
st.markdown(
    _gradient_legend(colormap, viz.METRIC_LABELS[metric]), unsafe_allow_html=True
)


def _selected_cell(state: dict | None, panel_cells: set[str]) -> str | None:
    """Resolve a map click to an H3 cell id.

    Works for both zone and hex layers: converts the raw click lat/lng
    into an H3 cell and checks it exists in the panel.
    """
    if not state:
        return None
    clicked = state.get("last_object_clicked") or {}
    lat, lng = clicked.get("lat"), clicked.get("lng")
    if lat is not None and lng is not None:
        cell = h3.latlng_to_cell(float(lat), float(lng), config.H3_RESOLUTION)
        if cell in panel_cells:
            return cell
    return None


_panel_cells = set(panel["h3"].unique())
selected = _selected_cell(map_state, _panel_cells)

# ---------------------------------------------------------------------------
# Selected-cell detail
# ---------------------------------------------------------------------------


def _history_figure(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hist["month"],
            y=hist["n_listings"],
            name="listings",
            yaxis="y2",
            marker_color="rgba(84,110,122,0.35)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["month"],
            y=hist["price_azn_m2_median"],
            mode="lines+markers",
            name="median AZN/m²",
            line=dict(color=_ACCENT, width=2),
        )
    )
    fig.update_layout(
        title="Observed history",
        yaxis=dict(title="AZN/m²"),
        yaxis2=dict(title="listings", overlaying="y", side="right", showgrid=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _fan_figure(hist: pd.DataFrame, forecast: pd.DataFrame, base_month: str) -> go.Figure:
    """Forecast fan: q10–q90 band + q50 line anchored at the last observation."""
    base_price = (
        float(hist["price_azn_m2_median"].iloc[-1]) if len(hist) else float("nan")
    )
    x = [base_month] + [
        str(pd.Period(base_month, freq="M") + int(h))
        for h in forecast["horizon_months"]
    ]
    q10 = [base_price] + forecast["q10"].astype(float).tolist()
    q50 = [base_price] + forecast["q50"].astype(float).tolist()
    q90 = [base_price] + forecast["q90"].astype(float).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q10,
            fill="tonexty",
            fillcolor="rgba(21,101,192,0.18)",
            line=dict(width=0),
            name="q10–q90 (√horizon-scaled band)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q50,
            mode="lines+markers",
            name="q50 forecast",
            line=dict(color=_ACCENT, width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Forecast fan (heuristic uncertainty band)",
        yaxis=dict(title="AZN/m²"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


if selected is None:
    st.info("Click a hexagon on the map to inspect its history, forecast, and features.")
else:
    st.subheader(f"Cell {selected}")
    hist = viz.cell_history(panel, selected)
    forecast = viz.cell_forecast(predictions, selected, scenario)
    sel_pred = predictions[
        (predictions["h3"] == selected) & (predictions["scenario"] == scenario)
    ]
    base_month = (
        str(sel_pred["base_month"].iloc[0])
        if len(sel_pred)
        else (str(hist["month"].iloc[-1]) if len(hist) else config.PANEL_END)
    )

    col_hist, col_fan = st.columns(2)
    with col_hist:
        if len(hist):
            st.plotly_chart(_history_figure(hist))
        else:
            st.info("No observed history for this cell (thin cell-months are dropped).")
    with col_fan:
        if len(forecast):
            st.plotly_chart(_fan_figure(hist, forecast, base_month))
        else:
            st.info("No forecast rows for this cell under the selected scenario.")

    if "cell_features" in artifacts:
        feats = artifacts["cell_features"]
        row = feats[feats["h3"] == selected]
        if len(row):
            preferred = [
                "dist_centre_km",
                "dist_metro_km",
                "dist_nearest_node_km",
                "node_gravity",
                "in_redev_zone",
                "n_stations_2km",
                "dist_treatment_km",
                "cv_fold",
            ]
            cols = [c for c in preferred if c in row.columns]
            if cols:
                st.caption("Static cell features (Master Plan 2040 + infrastructure)")
                show = row[cols].T
                show.columns = ["value"]
                st.dataframe(show)

# ---------------------------------------------------------------------------
# Analysis tabs
# ---------------------------------------------------------------------------

tab_shap, tab_did, tab_val, tab_about = st.tabs(
    ["Price drivers (SHAP)", "Metro causality (DiD)", "Validation", "About"]
)

with tab_shap:
    if "shap" not in artifacts or not artifacts["shap"]:
        st.info("SHAP artifact missing — run `make demo` to build it.")
    else:
        shap_items = sorted(
            artifacts["shap"].items(), key=lambda kv: kv[1], reverse=True
        )[:15]
        names = [k for k, _ in shap_items][::-1]  # biggest on top
        vals = [v for _, v in shap_items][::-1]
        fig = go.Figure(
            go.Bar(x=vals, y=names, orientation="h", marker_color=_ACCENT)
        )
        fig.update_layout(
            title="Mean |SHAP| per feature (final XGBoost fit)",
            xaxis_title="mean |SHAP| (AZN/m²)",
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig)
        st.caption(
            "SHAP decomposes each prediction into per-feature contributions; "
            "the bar length is the average absolute contribution across the "
            "panel — a global ranking of what moves prices."
        )

with tab_did:
    if "did" not in artifacts:
        st.info("DiD artifact missing — run `make demo` to build it.")
    else:
        did = artifacts["did"]
        att_pct = float(did.get("att_pct", float("nan")))
        ci_lo_pct = 100.0 * (math.exp(float(did.get("ci_low", float("nan")))) - 1.0)
        ci_hi_pct = 100.0 * (math.exp(float(did.get("ci_high", float("nan")))) - 1.0)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"ATT of the {config.DID_TREATMENT_STATION} opening",
            f"{att_pct:+.1f} %",
            help="Average treatment effect on log price within the 1 km "
            "walking catchment, converted to percent.",
        )
        c2.metric("95% CI", f"[{ci_lo_pct:+.1f} %, {ci_hi_pct:+.1f} %]")
        c3.metric("p-value", f"{float(did.get('p_value', float('nan'))):.4f}")
        st.caption(
            f"Opening month {did.get('open_month', '?')} · "
            f"{did.get('n_treated_listings', '?')} treated / "
            f"{did.get('n_control_listings', '?')} control listings · "
            f"spec: {did.get('spec', 'n/a')}"
        )

        events = did.get("event_study") or []
        if events:
            ev = pd.DataFrame(events).sort_values("rel_month")
            fig = go.Figure(
                go.Scatter(
                    x=ev["rel_month"],
                    y=ev["coef"],
                    mode="markers",
                    marker=dict(color=_ACCENT, size=7),
                    error_y=dict(
                        type="data",
                        array=(ev["ci_high"] - ev["coef"]).to_numpy(),
                        arrayminus=(ev["coef"] - ev["ci_low"]).to_numpy(),
                        color="rgba(21,101,192,0.5)",
                    ),
                    name="event-time coefficient",
                )
            )
            fig.add_hline(y=0.0, line_color="#9e9e9e", line_width=1)
            fig.add_vline(x=-0.5, line_color="#9e9e9e", line_dash="dot")
            fig.update_layout(
                title="Event study: log-price effect by months since opening "
                "(base period: −1)",
                xaxis_title="months relative to opening",
                yaxis_title="coefficient (log points)",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig)
            st.caption(
                "Flat pre-period coefficients support parallel trends; the "
                "post-period ramp traces how the transit premium capitalises "
                "into prices over ~3 months."
            )
        else:
            st.info("No event-study rows in the DiD artifact.")

with tab_val:
    if "metrics" not in artifacts:
        st.info("Metrics artifact missing — run `make demo` to build it.")
    else:
        metrics = artifacts["metrics"]

        def _vs_naive_table(block: dict | None) -> pd.DataFrame | None:
            """Two-row model-vs-naive table from a walk-forward/CV block."""
            if not isinstance(block, dict):
                return None
            rows = {}
            if isinstance(block.get("aggregate"), dict):
                rows["XGBoost"] = block["aggregate"]
            if isinstance(block.get("naive"), dict):
                rows["Naive persistence"] = block["naive"]
            if not rows:
                return None
            table = pd.DataFrame(rows).T
            return table.rename(
                columns={"mae": "MAE (AZN/m²)", "mape": "MAPE (%)", "r2": "R²"}
            )

        wf_table = _vs_naive_table(metrics.get("walk_forward"))
        st.markdown("**Temporal walk-forward** (expanding window, train strictly before test)")
        if wf_table is not None:
            st.dataframe(wf_table.style.format("{:.3f}"))
        else:
            st.info("No walk-forward metrics in the artifact.")

        scv_table = _vs_naive_table(metrics.get("spatial_cv"))
        st.markdown(
            f"**Spatial blocked CV** (whole H3 res-{config.H3_BLOCK_RESOLUTION} "
            "blocks held out)"
        )
        if scv_table is not None:
            st.dataframe(scv_table.style.format("{:.3f}"))
        else:
            st.info("No spatial-CV metrics in the artifact.")

        conformal = metrics.get("conformal")
        if isinstance(conformal, dict) and "coverage_q10_q90" in conformal:
            st.metric(
                "Held-out coverage of the 80% conformal interval",
                f"{100.0 * float(conformal['coverage_q10_q90']):.1f} %",
                help="Share of cell-months in the final holdout window "
                "(never used for training or calibration) whose true median "
                "falls inside [q10, q90]; the CQR target is 80%.",
            )

        dataset = metrics.get("dataset")
        if isinstance(dataset, dict):
            st.caption(
                f"Dataset: {dataset.get('n_listings_raw', '?')} raw listings → "
                f"{dataset.get('n_listings_deduped', '?')} after dedup · "
                f"{dataset.get('n_cells', '?')} cells × "
                f"{dataset.get('n_months', '?')} months."
            )

with tab_about:
    st.markdown(
        """
### Beyond Price Prediction — methodology in one page

**Question.** Not *what does a flat cost*, but *where is Baku appreciating,
by how much, with what confidence — and what does the metro cause?*

**Spatial backbone.** Listings are aggregated onto Uber **H3 hexagons at
resolution 8** (~0.73 km² each): uniform-area cells with exactly six
neighbours give comparable medians, clean spatial lags, and a hierarchy for
blocked cross-validation. Cell-months with fewer than
{min_listings} listings are dropped as broker noise.

**Data hygiene.** Broker re-posts are removed by perceptual-hash (pHash)
photo matching plus character-n-gram TF-IDF text similarity, blocked by
location/rooms/area/price so it scales.

**Leakage-proof features.** Every feature at month *t* uses only information
from *t−1* or earlier (own lags, neighbour prices, listing counts) or purely
static geometry — distances to the centre, metro stations, **Master Plan
2040 polycentric nodes**, and redevelopment zones.

**Validation.** Temporal **walk-forward** (expanding window, train strictly
before test) against a naive persistence benchmark, plus **spatial blocked
CV** (whole H3 res-{block_res} blocks held out) to measure transfer to
unseen neighbourhoods.

**Uncertainty.** **Conformalized quantile regression** (CQR, α = 0.2):
XGBoost quantile models calibrated on later months, with empirical q10–q90
coverage reported on a final never-used holdout window. The calibrated
1-month band inherits CQR's finite-sample coverage logic; the 12/24-month
forecast fan scales that width by √horizon — a stated random-walk
heuristic, not a guarantee.

**Causality.** The staggered metro build-out is a natural experiment:
a **spatial difference-in-differences** around station
{treatment} compares log-price changes inside the 1 km walking
catchment against 2–6 km controls, with month and cell fixed effects and
cluster-robust errors — the event study checks parallel pre-trends.

**Forecasts & scenarios.** A spatio-temporal graph model (STGCN, with a
spatial-lag ridge fallback) rolls prices forward {horizons} months;
scenario adjustments (polycentric shift, transit build-out) rescale
*estimated* — never planted — effects.

*Holberton School graduation project, Group 3. Data shown here comes from a
synthetic generator with planted ground truth (see `synthetic_truth.json`);
the scraping stack for live bina.az data ships in `bakuml/data/scraping/`.*
        """.format(
            min_listings=config.MIN_LISTINGS_PER_CELL_MONTH,
            block_res=config.H3_BLOCK_RESOLUTION,
            treatment=config.DID_TREATMENT_STATION,
            horizons="/".join(str(h) for h in config.FORECAST_HORIZONS),
        )
    )
