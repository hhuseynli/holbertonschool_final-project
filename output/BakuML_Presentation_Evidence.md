# Revised presentation — evidence notes

Prepared 18 September 2026. This note supersedes the earlier review’s statements that no local synthetic data or runnable artifacts were available. Both full synthetic research runs were completed for this revision.

## Reproduction

- Pipeline: bakuml.pipeline.run_pipeline, fast=False, seed 20243.
- KD-tree: target approximately 140 listings per cell.
- H3: resolution 8, used for the honeycomb maps and local application capture.
- Forecast: SpatialLagRidgeForecaster, explicitly selected; these are not STGCN results.
- Run script: .presentation-build/research_run.py.
- Recorded inputs/outputs: .presentation-build/research-kdtree and research-h3.
- Captures: .presentation-build/prepare_evidence.py and demo_server.py.

## Fresh KD-tree measurements

| Measurement | Value |
|---|---:|
| Raw synthetic listings | 37312 |
| Deduplicated listings | 35195 |
| Reposts removed | 2117 |
| Months | 44 |
| Cells | 256 |
| Observed cell-months | 6799 |
| Temporal MAE, model | 211.935 AZN/m² |
| Temporal MAE, persistence | 264.274 AZN/m² |
| Spatial MAE, model | 207.556 AZN/m² |
| Spatial MAE, persistence | 253.263 AZN/m² |
| Holdout interval coverage | 74.643% |
| Target coverage | 80% |
| Simulated metro ATT | 0.067492 log units |

## Scope and interpretation

All numerical performance and forecast examples in this deck are synthetic research results. The maps use real geographic coordinates and an OpenStreetMap basemap, but their price data are synthetic. The current-price map uses each cell’s latest available observation, which may differ between cells. Colour endpoints clamp at the 5th and 95th percentiles.

The monthly XGBoost validation does not establish 12/24-month forecast accuracy. Spatial evaluation retains own-cell historical prices, so it is model transfer with local history. TreeSHAP ranks fitted contributions, not causal effects. The metro result is a recovery experiment against planted truth. The cover image is an AI-generated conceptual illustration.

The selected forecast example is kd:10010011, centroid (40.37937125, 49.849948999999995), selected as the nearest eligible cell to the configured centre with at least 30 observed months. It was not selected for positive appreciation. All six endpoints are below its latest observed median.

The local application capture uses genuine Flask pages and H3 artifacts. A separate capture wrapper adds the synthetic-demo banner and converts RGBA colour arrays to CSS hex colours. No production app source was edited.

The real-data snapshot pipeline remains distinct: its future appreciation is heuristic, and source review identified neighbour-feature leakage and in-sample interval calibration. Credible real-market claims require longitudinal collection and horizon-specific backtesting. The project explores market potential; it does not separately validate physical urban expansion.

## Slide sources

1. README.md, bakuml/config.py, bakuml/pipeline.py; conceptual cover illustration.
2. research-h3/panel.parquet, predictions.parquet, cell_geometry.geojson; bakuml/viz.py; OpenStreetMap contributors.
3–4. research-kdtree/metrics.json; bakuml/models/baseline.py.
5. research-kdtree/shap_summary.json.
6. research-kdtree/did_results.json, synthetic_truth.json; bakuml/causal/did.py.
7. research-h3/predictions.parquet, metrics.json; bakuml/pipeline.py; OpenStreetMap contributors.
8. research-kdtree/predictions.parquet, panel.parquet, cell_features.parquet.
9. app/flask_app.py, app/static/js/map.js; demo_server.py.
10. research-kdtree/metrics.json; bakuml/models/conformal.py.
11–12. The preceding research outputs; scripts/run_real_pipeline.py; README.md.

Detailed explanations and limitations are also embedded in each slide’s speaker notes.
