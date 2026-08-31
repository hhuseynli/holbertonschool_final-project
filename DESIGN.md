# Design & Module Contracts

**Beyond Price Prediction** — a spatiotemporal ML study of Baku & Absheron.
Holberton School ML graduation project, Group 3.

This document is the binding contract between modules. Every function listed
here must exist with exactly this signature; the pipeline (`bakuml/pipeline.py`)
and the Streamlit app wire modules together purely through these interfaces.

## Ground rules (all modules)

- Python 3.11, **pandas 3.0.x** (modern API only — no deprecated calls),
  **h3 v4 API** (`latlng_to_cell`, `cell_to_latlng`, `cell_to_boundary`,
  `grid_disk`, `grid_ring`, `cell_to_parent`), xgboost 3.x, sklearn 1.9.
- **The unit of analysis is a parameter, not an assumption.** Cells come
  from a `bakuml.spatial.tessellation.Tessellation`; pipeline code must not
  call `h3.*` to derive geometry, contiguity or CV blocks. The panel key
  column stays named `h3` for schema stability but holds an opaque id (an H3
  index, a KD path `kd:0110`, or a region id `mr:0042`).
- All constants come from `bakuml.config`; schemas from `bakuml.data.schema`;
  distances via `bakuml.geo.haversine_km` / `min_distance_km`.
- Determinism: every stochastic step takes/uses a seed
  (default `config.SYNTHETIC_TRUTH.seed`).
- Months are `"YYYY-MM"` strings everywhere; convert with `pd.Period` internally.
- **Leakage rule**: any feature attached to `(cell, month=t)` may only use
  information from months `<= t-1`, except purely static geometry
  (distances to fixed points, zone flags). This is a headline claim of the
  project — violating it is a bug.
- No network access in tests. Tests live in `tests/test_<module>.py`, must
  pass with plain `pytest`, and each test file must run in < 90 s.
- Package already installed editable (`pip install -e .`).

## Data tables (see `bakuml/data/schema.py`)

- **LISTINGS**: one row per advert — columns in `LISTING_COLUMNS`.
- **PANEL**: one row per (h3, month) — base columns in `PANEL_BASE_COLUMNS`,
  feature modules append columns.
- **PREDICTIONS**: `PREDICTION_COLUMNS`.
- App artifacts written under `config.ARTIFACTS_DIR` with names from
  `schema.ARTIFACT_FILES`.

Ground truth planted in the synthetic data (`config.SYNTHETIC_TRUTH`,
`bakuml/data/synthetic.py`) — tests must *recover* these, not just run:
duplicate map, DiD effect (`did_effect_log=0.08` reached after a 3-month
ramp from `simulated_open=2025-06`), centre-decay gradient, metro premium.

---

## 1. `bakuml/data/scraping/` — bina.az Scrapy project

Files: `settings.py`, `items.py`, `middlewares.py`, `pipelines.py`,
`spiders/bina.py`, `runner.py`.

- `items.py: ListingItem(scrapy.Item)` — fields: `url, listing_id, title,
  description, price_raw, area_raw, rooms_raw, floor_raw, building_type_raw,
  lat, lon, district, listed_date_raw, image_urls`.
- `middlewares.py`:
  - `CloudflareRetryMiddleware` — detects CF challenge responses
    (403/503 + `cf-` markers / "Just a moment"), exponential backoff,
    rotates `User-Agent` from a bundled list, gives up after N retries.
  - `ProxyRotationMiddleware` — round-robins `request.meta["proxy"]` over
    `settings.PROXY_LIST` (default: empty ⇒ no proxy).
- `pipelines.py`:
  - `NormalizePipeline` — uses pure functions (also exported for tests):
    `parse_price("150 000 AZN") -> 150000.0`, `parse_area("85.5 m²") -> 85.5`,
    `parse_rooms`, `parse_floor("4/9") -> (4, 9)`,
    `parse_listed_month("28 Avqust 2026"/"bugün"/"dünən", today) -> "YYYY-MM"`
    (Azerbaijani month names). Drops rows without price/area/coords.
  - `PhashPipeline` — if image bytes provided, computes
    `imagehash.phash` hex; must degrade gracefully offline (phash="").
- `spiders/bina.py: BinaSpider` — start URL
  `https://bina.az/alqi-satqi/menziller`, parses listing cards + pagination +
  detail pages with defensively-guarded CSS selectors (site markup changes;
  selectors documented as of 2026). `ROBOTSTXT_OBEY=True`, autothrottle on.
- `runner.py: scrape_to_parquet(out_path: Path, max_pages: int = 5) -> Path`
  via `CrawlerProcess`. Never invoked by tests.
- Tests parse a **local HTML fixture** (`tests/fixtures/bina_sample.html`,
  written by you, structurally plausible) and unit-test all parse functions
  and both middlewares with fake Request/Response objects.

## 2. `bakuml/data/dedup.py` — broker re-post detection

Brokers re-post the same flat under different names: near-identical photos
(pHash within a few bits) and lightly reworded text.

- `phash_hamming(a: str, b: str) -> int` — hamming distance of hex hashes
  (64-bit); `64` if either is empty/invalid.
- `find_duplicate_pairs(listings, *, phash_max_hamming=6, text_sim_min=0.80,
  max_dist_m=300.0, price_rel_tol=0.10) -> pd.DataFrame`
  — columns `[listing_id_a, listing_id_b, phash_hamming, text_sim]`.
  Blocking: compare only candidates within `max_dist_m` (use a rounded
  lat/lon grid — do NOT do O(n²) over all listings), same `rooms`, area
  within 5 %, price within `price_rel_tol`. A candidate pair is a duplicate
  if `phash_hamming <= phash_max_hamming` **or** TF-IDF cosine of
  `description` (char ngrams (3,5), sklearn) `>= text_sim_min`.
- `dedupe(listings) -> tuple[pd.DataFrame, dict[str, str]]` — union-find over
  pairs; canonical = earliest `listed_month`, ties by `listing_id`; returns
  (clean_frame, map dup_id -> canonical_id).
- Tests: on a synthetic sample (subset of months for speed) recover ≥ 90 % of
  `truth_info["duplicate_map"]` pairs with ≤ 2 % false positives.

## 2b. `bakuml/spatial/tessellation.py` — the unit of analysis

Hexagons beat administrative rayons (MAUP), but they only make the
arbitrariness *uniform*. On this panel res-8 discards ~37 % of listings to
the thinness filter, leaves the median cell ~5 months of history, and its
spatial-transfer skill swings by 0.48 across resolutions 7–9 (changing
sign). So the geometry is pluggable and the default is chosen by
`scripts/maup_study.py`.

- `Tessellation` (Protocol) — `assign(lat, lon) -> ids`,
  `centroid(cell) -> (lat, lon)`, `neighbours(cell, k) -> tuple[str, ...]`
  (self excluded), `boundary(cell) -> [(lat, lon), ...]` (unclosed ring),
  `block(cell) -> str` (coarse contiguous id for spatially blocked CV).
  Fitted implementations also expose `fit(listings, price_cutoff_month=None)`,
  `cells()` and `describe() -> dict`.
- `H3Tessellation(resolution, block_resolution)` — the documented baseline;
  `fit` is a no-op.
- `AdaptiveKDTessellation(target_per_cell, block_depth, min_span_m)` —
  recursive median splits on **coordinates only** (no prices ⇒ no target
  leakage) giving equal-count cells. Leaf sizes are quantised to
  `n / 2**depth`, so `target_per_cell` selects the nearest achievable
  partition and `describe()` reports what was achieved. `block()` truncates
  the KD path; every prefix is a rectangle, so blocks are contiguous by
  construction.
- `MarketRegionTessellation(n_regions, base_resolution, n_blocks)` —
  contiguity-constrained Ward clustering of base H3 cells on (mean log
  price, new-build share). **The only price-driven tessellation**, so `fit`
  must honour `price_cutoff_month`. Clusters per connected component of the
  contiguity graph — the observed grid is disconnected (Sumgait, Alat, the
  Absheron villages) and a disconnected connectivity matrix makes sklearn
  silently bridge islands.
- Module state: `get_active()` / `set_active(tess)` / `resolve(tess)` /
  `reset_active()`, and `build(kind, **kwargs)` for `h3` / `kdtree` /
  `market`.
- Tests assert the shared contract for all three (total partition,
  symmetric contiguity, monotone k-rings, closed `[lon, lat]` rings, blocks
  coarser than cells, panel builds) plus each one's reason to exist: equal
  counts for the KD-tree, homogeneity-at-equal-granularity and a
  post-cutoff-price leakage canary for market regions.

## 3. `bakuml/spatial/` — spatial backbone (tessellation-agnostic)

- `grid.py`:
  - `assign_cells(listings, res=config.H3_RESOLUTION) -> pd.DataFrame`
    (copy with added `h3` column).
  - `build_cell_month_panel(listings, min_listings=config.MIN_LISTINGS_PER_CELL_MONTH)
    -> pd.DataFrame` — aggregates to `PANEL_BASE_COLUMNS`
    (median/mean of `price_azn_m2`, count, share of `building_type=="new"`),
    keeps only rows with `n_listings >= min_listings`, sorted by (h3, month).
  - `complete_panel(panel, months: list[str]) -> pd.DataFrame` — reindex to
    the full cells × months rectangle; missing rows get `n_listings=0` and
    NaN prices (needed for tensors).
  - `cell_centroids(cells: list[str]) -> pd.DataFrame[h3, lat, lon]`.
  - `cells_to_geojson(cells: list[str], properties: dict[str, dict] | None = None)
    -> dict` — GeoJSON FeatureCollection of hexagon polygons, `id`=h3,
    lon/lat ring order (GeoJSON convention!), optional per-cell properties.
- `lags.py`:
  - `spatial_lag(panel, value_col: str, k: int = 1) -> pd.Series` — same-month
    mean of `value_col` over `grid_disk(h3, k)` **excluding self**; aligned to
    `panel.index`; NaN when no neighbour has data.
  - `neighbour_target_encoding(panel, target_col: str, months_lag: int = 1)
    -> pd.Series` — neighbour mean of `target_col` taken from month `t - months_lag`
    (leakage-safe by construction).
- `graph.py`:
  - `build_adjacency(cells: list[str]) -> tuple[np.ndarray, list[str]]` —
    dense symmetric normalized adjacency `D^-1/2 (A+I) D^-1/2` over h3
    neighbour relations restricted to `cells`; returns (matrix, cell order).
  - `edge_index(cells: list[str]) -> np.ndarray` shape (2, E), both directions.

## 4. `bakuml/features/` — Master Plan 2040 & infrastructure (imports spatial)

- `masterplan.py: masterplan_features(cells: list[str]) -> pd.DataFrame` —
  one row per cell: `h3`, `dist_centre_km`, `dist_node_<name>_km` for every
  `config.POLYCENTRIC_NODES` entry, `dist_nearest_node_km`,
  `node_gravity` (= max over nodes of 1/(1+dist_km)), `in_redev_zone` (0/1
  any of `config.REDEVELOPMENT_ZONES`).
- `infrastructure.py: metro_features(cells: list[str], as_of_month: str, *,
  include_planned: bool = False) -> pd.DataFrame` — per cell: `h3`,
  `dist_metro_km` (nearest station with `opened <= as_of_month`; when
  `include_planned`, a station also counts from its `simulated_open`),
  `dist_treatment_km` (distance to `config.DID_TREATMENT_STATION`),
  `n_stations_2km`.
- `build.py`:
  - `FEATURE_COLS: list[str]` — the canonical model feature list.
  - `build_feature_matrix(panel: pd.DataFrame, *, include_planned_metro=False)
    -> pd.DataFrame` — panel plus: all masterplan + metro features
    (metro recomputed per month so openings switch on over time),
    `month_ix` (int position in panel months), `lag_own_1m`, `lag_own_3m`
    (cell's own median price at t-1 / t-3), `mom_3m` (t-1 vs t-4 % change),
    `nbr_price_prev_month` (from `lags.neighbour_target_encoding`),
    `new_share_prev_month`, `n_listings_prev_month`. **All lags use months
    <= t-1.** Rows lacking `lag_own_1m` may be dropped.
- Tests must include a leakage canary: perturb all prices at months > t₀ and
  assert features at months <= t₀ are bit-identical.

## 5. `bakuml/causal/did.py` — Spatial Difference-in-Differences

Natural experiment: staggered metro expansion; treatment station
`config.DID_TREATMENT_STATION` (B-04) with (simulated) opening
`simulated_open`. Uses **deduplicated listings**, not the panel.

- `@dataclass DiDResult`: `att_log, att_pct, se, p_value, ci_low, ci_high,
  n_treated_listings, n_control_listings, open_month, spec: str`.
- `prepare_did_frame(listings, *, treatment=config.DID_TREATMENT_STATION,
  radius_km=config.DID_TREATMENT_RADIUS_KM,
  buffer_km=config.DID_BUFFER_RADIUS_KM,
  control_max_km=config.DID_CONTROL_MAX_RADIUS_KM) -> pd.DataFrame`
  — adds `dist_treatment_km`, `treated` (<= radius), drops the fuzzy ring
  (radius, buffer], keeps controls in (buffer, control_max]; adds `post`,
  `rel_month` (months since opening; use `simulated_open` when `opened` is
  None), `log_price = log(price_azn_m2)`, `h3` cell id.
- `run_spatial_did(frame) -> DiDResult` — OLS:
  `log_price ~ treated:post + C(month FE) + C(cell FE) + area_m2 + rooms +
  floor + is_new`, cluster-robust SEs by cell (statsmodels,
  `cov_type="cluster"`). Implement FE via dummies or within-demeaning —
  document the choice.
- `event_study(frame, window=(-12, 14)) -> pd.DataFrame[rel_month, coef, se,
  ci_low, ci_high, n]` — `treated × C(rel_month)` interactions, base period
  `rel_month = -1`, clipped to window.
- `did_artifact(result: DiDResult, events: pd.DataFrame) -> dict` — JSON-safe
  dict for `ARTIFACT_FILES["did"]` with keys `att_log, att_pct, se, ci_low,
  ci_high, p_value, n_treated_listings, n_control_listings, open_month,
  spec, event_study: list[{rel_month, coef, se, ci_low, ci_high, n}]`.
- Tests: event-study coefficients at `rel_month >= 3` (post-ramp) recover
  `0.08 ± 0.025`; pre-period coefficients `|coef| < 0.02`; ATT positive and
  significant. Generate listings once per module (fixture, module scope).

## 6. `bakuml/validation/` — leakage-proof evaluation

- `temporal.py: walk_forward_splits(months: list[str], *,
  min_train=config.WALK_FORWARD_MIN_TRAIN_MONTHS,
  test_size=config.WALK_FORWARD_TEST_MONTHS, step=1)
  -> list[tuple[list[str], list[str]]]` — expanding window; train strictly
  before test; last split's test ends at the final month.
- `spatial_cv.py: spatial_block_folds(cells: list[str], *,
  n_folds=config.SPATIAL_CV_FOLDS, block_res=config.H3_BLOCK_RESOLUTION,
  seed=0) -> dict[str, int]` — cells grouped by `cell_to_parent(block_res)`;
  whole blocks assigned to folds, greedily balancing cell counts. Every
  cell mapped to exactly one fold 0..n_folds-1.

## 7. `bakuml/models/` — Tier 1 baseline, conformal intervals, Tier 2 STGCN

Models consume the feature matrix as a plain DataFrame + `feature_cols`
list; they never call `bakuml.features` themselves.

- `baseline.py`:
  - `@dataclass TrainedBaseline`: `model` (XGBRegressor), `feature_cols`,
    `train_months: list[str]`.
  - `train_baseline(fm: pd.DataFrame, feature_cols: list[str], *,
    target=config.TARGET_COL, seed=0, params: dict | None = None)
    -> TrainedBaseline` (sane defaults: ~600 trees, lr 0.05, depth 6,
    subsample/colsample 0.8, early stopping off).
  - `predict(tb: TrainedBaseline, fm) -> np.ndarray`.
  - `evaluate_walk_forward(fm, feature_cols, *, target=..., seed=0) -> dict`
    — per-split and aggregate `mae, mape, r2`, plus the same metrics for a
    **naive persistence baseline** (`lag_own_1m` as the prediction);
    keys: `splits: list[...]`, `aggregate: {...}`, `naive: {...}`.
  - `evaluate_spatial_cv(fm, feature_cols, *, target=..., n_folds=5, seed=0)
    -> dict` — GroupKFold-style over `spatial_block_folds`, same metric keys.
  - `shap_summary(tb: TrainedBaseline, fm, top_k=15) -> dict[str, float]` —
    mean |SHAP| per feature, sorted desc, JSON-safe floats.
- `conformal.py`:
  - `@dataclass ConformalModel`: quantile models for (0.10, 0.50, 0.90),
    `feature_cols`, calibration offsets.
  - `fit_cqr(fm, feature_cols, *, target=..., train_months, cal_months,
    seed=0) -> ConformalModel` — XGBoost `reg:quantileerror` per quantile,
    then CQR calibration on `cal_months` (temporally after train_months):
    conformity score `max(q10 - y, y - q90)`, adjust both bounds by the
    (1-α)(1+1/n)-quantile, α=0.2.
  - `predict_intervals(cm, fm) -> pd.DataFrame[q10, q50, q90]` (index-aligned,
    monotonicity enforced: q10 <= q50 <= q90).
  - `coverage(y, q10, q90) -> float`.
- `stgcn.py` (torch is OPTIONAL — module must import without torch):
  - `HAS_TORCH: bool`.
  - `panel_tensor(panel_complete, months, cells, value_col=config.TARGET_COL)
    -> tuple[np.ndarray, np.ndarray]` — (values [T, N] float32 with NaN,
    observed mask [T, N] bool), ordered by `months` / `cells`.
  - `class STGCNForecaster` — `__init__(adjacency: np.ndarray, cells: list[str],
    months: list[str], *, hidden=32, ks=3, seed=0)`,
    `fit(values, mask, *, epochs=60, lr=1e-2, verbose=False) -> self`
    (works on log prices internally; NaNs ffilled for inputs, masked in the
    loss; temporal conv → graph conv (Â X W) → temporal conv, predicts
    next-month log price for all cells),
    `forecast(horizons=config.FORECAST_HORIZONS) -> pd.DataFrame[h3,
    horizon_months, y_pred]` — autoregressive rollout from the last month.
  - `class SpatialLagRidgeForecaster` — identical interface, sklearn Ridge on
    [own lags ×3, neighbour-mean lags ×3, linear month trend]; the fallback
    when torch is missing (and a comparison baseline when it isn't).
  - `make_forecaster(adjacency, cells, months, *, prefer="stgcn", seed=0)`.
  - Tests: on a small planted panel (trend + diffusion), each available
    forecaster beats persistence MAE; rollout shapes/finiteness; keep torch
    epochs small so the file runs in < 90 s.

## 8. `bakuml/viz.py` + `app/streamlit_app.py` — the demo

- `bakuml/viz.py` (pure, unit-testable, no streamlit import):
  - `load_artifacts(artifacts_dir=config.ARTIFACTS_DIR) -> dict` — reads every
    `ARTIFACT_FILES` entry that exists (parquet → DataFrame, json → dict);
    missing files simply absent from the dict.
  - `hex_layer_geojson(predictions, panel, *, scenario, horizon, metric)
    -> tuple[dict, "branca colormap"]` — joins the chosen metric onto cell
    polygons (`spatial.grid.cells_to_geojson`), returns geojson + colormap;
    metrics: `"appreciation_pct" | "q50" | "hotspot_prob" |
    "uncertainty" (q90-q10) | "current_price"` (last observed median).
  - `cell_history(panel, h3_id) -> pd.DataFrame[month, price_azn_m2_median,
    n_listings]`; `cell_forecast(predictions, h3_id, scenario)
    -> pd.DataFrame[horizon_months, q10, q50, q90]`.
- `app/streamlit_app.py` — Folium map of the hex layer (+ metro stations,
  polycentric nodes, redevelopment circles), sidebar controls (scenario,
  horizon, metric), `st_folium` click → selected hexagon detail: history +
  forecast fan chart (plotly), top SHAP drivers bar, tabs for DiD results
  (event-study plot) and validation metrics. Graceful empty state pointing
  at `make demo` when artifacts are missing.

## 9. `bakuml/pipeline.py` + `scripts/` (integration — written last)

`run_pipeline(scenarios=("baseline",), fast=False)`:
generate/load listings → dedupe → assign cells → panel → features →
walk-forward + spatial CV metrics → SHAP → conformal intervals → STGCN
forecasts per scenario → DiD → write every `ARTIFACT_FILES` artifact.
`predictions.parquet` combines conformal interval geometry with forecaster
q50 path per `PREDICTION_COLUMNS`; `hotspot_prob` = P(cell appreciation >
citywide median appreciation) derived from the interval width (normal
approx around q50).
