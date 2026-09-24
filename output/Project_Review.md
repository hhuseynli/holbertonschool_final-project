# BakuML: current project review

Reviewed 17 September 2026 at commit `015d9da`. The presentation uses the current source code and clearly labels README benchmark results as synthetic.

## What the project currently contains

The project has two analytical paths and a Flask web application. The research path generates controlled monthly listings, removes duplicates, constructs a spatial panel, builds historical features, evaluates XGBoost, fits conformal quantiles, forecasts with STGCN or a Ridge fallback, and estimates a simulated metro effect. The real-data path ingests a bina.az snapshot and fits a listing-level cross-sectional price model. The application provides property search, listing details, and map-based analytics from saved artifacts.

The README and DESIGN documents lag behind the implementation: their Streamlit and Scrapy descriptions no longer match the Flask application and GraphQL client.

## Strongest contributions to present

1. **Geography is an experimental choice.** The common tessellation contract supports H3, adaptive KD-tree cells and market regions. The documented study compares retention, available history and blocked transfer skill. KD-tree ~140/cell balances those criteria rather than simply winning one score.
2. **The research evaluation distinguishes time and space.** Walk-forward evaluation compares the model with persistence. Spatial folds hold out contiguous blocks and exclude neighbour target encoding. The canary test verifies that changing future prices does not change earlier features.
3. **Synthetic ground truth supports meaningful tests.** Dedup and DiD tests check recovery of known planted quantities. This is stronger than checking only that a pipeline returns a dataframe.
4. **The application makes the outputs accessible.** Flask pages combine listing search with cell-level context, while JSON/Parquet artifacts separate model computation from web requests.

## Evidence boundaries

The README reports temporal MAE of 213 versus persistence 264 AZN/m², and spatial MAE of 209 versus 253. These imply approximately 19.3% and 17.4% lower error. These are XGBoost results on a synthetic dataset, not validated real-market or long-horizon STGCN performance.

The README reports 71.4% empirical CQR coverage against an 80% target. The research pipeline evaluates coverage on a final three-month holdout, but scales interval widths by square root of horizon for longer forecasts. Long-horizon coverage still needs independent backtesting.

The DiD figures recover a **simulated** metro treatment. They do not demonstrate the effect of an observed real-world opening. The average post-treatment estimate and post-ramp event-study estimate target different quantities.

The configured plan features use approximate node coordinates, simplified zones and haversine distances. The current code does not implement a full OSM network-distance pipeline, satellite embeddings, or a separately validated urban-expansion detector.

## Findings to address before a real-data defense or populated demo

| Priority | Finding | Evidence | Practical consequence |
|---|---|---|---|
| High | Neighbour target features precede spatial folds | `scripts/run_real_pipeline.py:103–126` | Held-out prices can enter neighbour features. Recompute features within each training fold or exclude them for the evaluation. |
| High | Real-data interval calibration uses training residuals | `scripts/run_real_pipeline.py:177–210` | In-sample coverage is not evidence of future coverage. Separate training, calibration and final evaluation. |
| High | Real-data appreciation is a fixed gravity formula | `scripts/run_real_pipeline.py:214–232` | The 12/24-month outputs and normal-CDF score are heuristic. Label them accordingly until longitudinal validation exists. |
| High | Listing enrichment assumes H3 | `app/enrichment.py:103` and `bakuml/config.py:38` | The research default produces KD-tree IDs, so listing-level lookups cannot match those artifacts. Persist and reuse the actual spatial assignment. |
| Medium | The UI offers scenarios absent from real outputs | `app/flask_app.py:73` and `scripts/run_real_pipeline.py:214` | Restrict selectors to available artifact combinations or supply compatible outputs. |
| Medium | Real and research runs share default artifact filenames | Both pipeline entry points and `bakuml/data/schema.py` | Runs can mix or overwrite artifact families. Store run provenance and isolate output directories. |
| Medium | Synthetic-only listings can pass the application loader | `app/flask_app.py:214–239` | The filter runs only when at least one non-synthetic source exists. The docstring's “never shown” statement is not enforced for an all-synthetic file. |
| Medium | Score and price-position labels need clearer meaning | `app/enrichment.py:106–198` | The score combines hand-set weights. Price position compares current asking price with a 12-month cell estimate, rather than a property-specific current valuation. |

The real pipeline also does not call the research dedup stage. The investment score weights are 30% appreciation, 25% price position, 20% metro proximity, 15% plan alignment and 10% momentum. They are design choices, not empirically validated investment weights.

## Verification performed

All 52 Python files parsed successfully. The repository contains 11 test modules with 155 test function definitions, including class methods and parametrized tests. This is a static count, not the number of collected or passing pytest cases.

The available Python runtime lacks pytest and the project's ML/geospatial dependencies, so the suite and model pipelines were not rerun. The local `artifacts/` and `data/` directories have no saved dataset/model outputs. Accordingly, the presentation uses documented benchmark values and does not invent real-data measurements or populated application screenshots.

## Presentation use

Slides 1–16 form the main approximately 10–12 minute presentation. Slides 17–19 are technical backup. PowerPoint speaker notes include pacing, explanations and source references. Present the product-demo sequence only after compatible listing and model artifacts are available.

The cover is an AI-generated conceptual Baku illustration created with the built-in imagegen tool. The embedded asset is retained at `.presentation-build/baku-concept.png`. Prompt: a portrait editorial illustration of Baku's Caspian waterfront, Flame Towers, midnight blue sea and sky, amber windows and a subtle cyan spatial-analysis grid, without text or labels. It is illustrative, not a factual map or documentary photograph.
