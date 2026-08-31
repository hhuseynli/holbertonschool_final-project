# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BakuML ("Beyond Price Prediction") is a spatiotemporal ML study of Baku & Absheron real estate. It predicts price gradients, urban expansion, and infrastructure hotspots over *spatial cells*, not individual apartment listings. The cell geometry is a parameter (`config.TESSELLATION`), chosen by measurement rather than assumed - see "Unit of analysis" below. The pipeline runs entirely offline on synthetic data that mirrors the real market, with planted ground truth that tests must *recover*.

## Commands

```bash
make install       # pip install -r requirements.txt && pip install -e .
make test          # python -m pytest tests/ -q
make demo          # fast end-to-end pipeline (~2-4 min), writes artifacts/
make full          # full pipeline with STGCN forecaster
make app           # streamlit run app/streamlit_app.py
make maup          # compare units of analysis under leakage-proof protocols

# Run a single test file
python -m pytest tests/test_did.py -q

# Run a specific test
python -m pytest tests/test_did.py::test_att_positive -q

# Pipeline scripts directly
python scripts/run_pipeline.py --fast
python scripts/run_pipeline.py --tessellation h3    # hexagonal baseline
python scripts/maup_study.py
python scripts/make_dataset.py
```

## Architecture

The pipeline flows through 10 stages orchestrated by `bakuml/pipeline.py`:

**Data layer** (`bakuml/data/`): Synthetic listings generator (`synthetic.py`) produces the offline dataset. Scrapy project (`scraping/`) handles bina.az with Cloudflare-retry and proxy-rotation middlewares. Dedup (`dedup.py`) removes broker re-posts via pHash hamming distance + TF-IDF cosine similarity with spatial blocking.

**Spatial backbone** (`bakuml/spatial/`): `tessellation.py` defines the unit of analysis behind a five-method protocol (assign / centroid / neighbours / boundary / block) with three implementations - H3 hexagons, an adaptive KD-tree of equal-count cells (the default), and contiguity-constrained market regions. `grid.py` assigns listings to cells and builds the `(h3, month)` panel; `lags.py` computes spatial lag features (leakage-safe: only months <= t-1); `graph.py` builds the normalized adjacency matrix for the GNN; `zones.py` groups cells into organic market micro-zones for the map. All of them delegate geometry and contiguity to the active tessellation, so none is H3-specific.

**Features** (`bakuml/features/`): Master Plan 2040 distances and node-gravity scores (`masterplan.py`), metro station distances recomputed per-month as stations open (`infrastructure.py`), and temporal lags assembled in `build.py`. `FEATURE_COLS` in `build.py` is the canonical feature list consumed by all models.

**Models** (`bakuml/models/`): Tier 1 is XGBoost with SHAP (`baseline.py`). Conformal quantile regression produces calibrated q10/q50/q90 intervals (`conformal.py`). Tier 2 is an STGCN graph neural net (`stgcn.py`) with a `SpatialLagRidgeForecaster` fallback when torch is unavailable.

**Causal** (`bakuml/causal/did.py`): Spatial Difference-in-Differences using the B-04 metro station opening as a natural experiment. OLS with month + cell fixed effects, cluster-robust SEs.

**Validation** (`bakuml/validation/`): Temporal walk-forward (expanding window, train strictly before test) and spatial blocked CV (hold out contiguous blocks, defined by `Tessellation.block`). Random k-fold is explicitly forbidden.

**Scenario engine** (in `pipeline.py`): Three scenarios (baseline, polycentric, transit) adjust forecasts using *estimated* quantities from the panel's own history, never planted truth.

**App** (`app/streamlit_app.py`): Folium map (zones when the `zones` artifact is present, else raw cells) with scenario/horizon/metric controls. Reads all artifacts from `artifacts/`, and takes cell polygons from `cell_geometry.geojson` because a fitted tessellation cannot be reconstructed outside the run that built it.

## Unit of analysis

The spec argues H3 hexagons over administrative rayons (the Modifiable Areal Unit Problem), and that argument holds. But a uniform grid over a non-uniform city is costly: measured on this panel, res-8 discards ~37 % of deduplicated listings to the thinness filter and leaves the median cell ~5 months of price history, and its spatial-transfer skill swings by 0.48 (changing sign) across resolutions 7-9.

So the geometry is pluggable and the default is empirical. `scripts/maup_study.py` (`make maup`) scores candidates under the project's own leakage-proof protocols; the adaptive KD-tree wins and is the default. Run `--tessellation h3` to reproduce the hexagonal baseline.

- Never derive geometry, contiguity or CV blocks by calling `h3.*` directly in pipeline code - go through the active tessellation, or the alternatives break.
- The panel key column is still named `h3` (schema stability), but it holds whatever id the active tessellation produces: an H3 index, a KD path (`kd:0110`), or a region id (`mr:0042`). Treat it as opaque.
- Only `MarketRegionTessellation` uses prices, so it must be fitted with `price_cutoff_month` (the pipeline passes `_TESS_HOLDOUT_MONTHS` before the end). Coordinate-only tessellations carry no target information.
- The observed grid is disconnected (Sumgait, Alat, the Absheron villages). Any clustering with a connectivity constraint must run per connected component; handing sklearn a disconnected matrix makes it silently bridge islands.

## Critical invariants

- **Leakage rule**: Any feature at `(cell, month=t)` may only use information from months `<= t-1`, except purely static geometry. Violating this is a bug. Tests include a leakage canary that perturbs future prices and asserts earlier features are bit-identical.
- **H3 v4 API only**: where h3 is used directly (`tessellation.H3Tessellation`, H3-specific shortcuts), use `latlng_to_cell`, `cell_to_latlng`, `cell_to_boundary`, `grid_disk`, `grid_ring`, `cell_to_parent`. Do not use deprecated v3 names.
- **Months are `"YYYY-MM"` strings** everywhere; convert with `pd.Period` internally.
- **All constants** come from `bakuml.config`; schemas from `bakuml.data.schema`.
- **Determinism**: Every stochastic step uses a seed (default `config.SYNTHETIC_TRUTH.seed`).
- **No network access in tests**. Each test file must complete in < 90 seconds.
- **torch is optional**: `bakuml/models/stgcn.py` must import without torch installed; check `HAS_TORCH` flag.
- **Tests assert recovery**, not just execution: the synthetic data has planted ground truth (`config.SYNTHETIC_TRUTH`), and tests verify the pipeline recovers the planted DiD effect, duplicate map, gradient, etc.

## Key configuration

All in `bakuml/config.py`: `TESSELLATION="kdtree"`, `H3_RESOLUTION=8`, `CITY_CENTRE`, `BBOX`, `METRO_STATIONS` list (with `opened`/`simulated_open` dates), `POLYCENTRIC_NODES`, `REDEVELOPMENT_ZONES`, `SCENARIOS`, `SYNTHETIC_TRUTH` dataclass, `DID_TREATMENT_STATION="B-04"`.

## Module contracts

`DESIGN.md` is the binding contract between modules — every function signature listed there must be maintained. The pipeline and Streamlit app wire modules together purely through these interfaces.
