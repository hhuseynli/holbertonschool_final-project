# Beyond Price Prediction

**A spatiotemporal ML study of Baku & Absheron — urban expansion, infrastructure
causality, and where the market moves next.**

Holberton School · ML Graduation Project · Group 3
Hüseyn Hüseynli · Nihad Süleymanov · Eldəniz Arifzadə · Zeynəb Mirzəzadə · Vüqar Dünyamalıyev

---

## Why this is not another price regressor

| | A typical price model | This project |
|---|---|---|
| **Unit of analysis** | A listing (rows in a table) | H3 hexagonal grid cells over the whole peninsula |
| **Question asked** | "What does this flat cost today?" | "Where will the city grow — and appreciate — next?" |
| **Time dimension** | None: one static snapshot | 12–24 month forward forecasts, walk-forward in time |
| **Method** | Fit a regression, report R² | Causal inference (Spatial DiD) + spatiotemporal graph nets |
| **Output** | One number per apartment | Probability heatmaps + calibrated price intervals |

Three things we honestly predict: **asking-price gradients** (price/m² as a
continuous spatial surface), **urban expansion**, and **infrastructure
hotspots** (which grid cells appreciate above average as metro milestones come
online). Deliberately out of scope: executed transaction prices and
days-on-market — the State Registry is closed.

## Quick start

```bash
make install       # pip install -r requirements.txt && pip install -e .
make demo          # end-to-end pipeline on the offline dataset (~2-4 min)
make app           # interactive Streamlit + Folium map
make test          # full test suite
make full          # full-size dataset + STGCN forecaster
```

`make demo` runs every stage — data → dedup → H3 panel → features →
leakage-proof validation → SHAP → conformal intervals → STGCN forecast →
Spatial DiD → artifacts — and drops the results into `artifacts/`, which the
Streamlit app reads. Click any hexagon on the map to see its price history and
its 12/24-month forecast fan.

## Data ecosystem

Five streams, one spatial backbone (see `DESIGN.md` for module contracts):

1. **bina.az / tap.az listings** — the core feature matrix. The Scrapy project
   (`bakuml/data/scraping/`) ships Cloudflare-retry + proxy-rotation
   middlewares and Azerbaijani-format parsers; brokers re-post the same flat
   under different names, so `bakuml/data/dedup.py` removes re-posts via
   perceptual image hashing (pHash) + TF-IDF text similarity.
2. **Geofabrik / OpenStreetMap** — building footprints, roads, POIs.
3. **Master Plan 2040 (ARXKOM)** — deterministic state policy, vectorized into
   features: network distance to each designated polycentric node (Alat,
   Mardakan, Sumgait + 5 local centres), redevelopment-zone flags.
4. **MBA Group analytics** — district-level price dynamics (priors/sanity).
5. **State Statistical Committee** — official indices (aggregated, integrity
   checks).

Because bina.az sits behind Cloudflare (and CI must run offline), the
repository ships a **synthetic stand-in feed** (`bakuml/data/synthetic.py`)
that mirrors the real market's structure — centre-to-periphery gradient
(~3,200 → ~1,500 AZN/m²), metro premiums, a staggered station opening, broker
duplicates — with every planted parameter recorded in
`artifacts/synthetic_truth.json`. The test suite asserts the pipeline
*recovers* what was planted (the DiD estimate, the duplicate map), not merely
that it runs.

## The novel part: the city's official future, digitized as features

Most models only learn a city's past. Baku publishes its future: the Master
Plan 2040 mandates a shift from a monocentric to a polycentric city. We
vectorize that mandate (`bakuml/features/masterplan.py`) into distances to
each polycentric node, node-gravity scores, and demolition/redevelopment-zone
flags, and feed them to the models alongside infrastructure timelines.

## Causal inference: what a metro station *causes*

Anyone can show prices are higher near metro. `bakuml/causal/did.py` uses the
staggered metro expansion as a natural experiment: a **Spatial
Difference-in-Differences** design (treatment = listings within 1 km of the
new purple-line station; controls = structurally similar listings 2–6 km out;
month + cell fixed effects; cluster-robust SEs) isolates the premium created
purely by new transit access, with an event-study to verify pre-trends.

## Spatial architecture

- **Hexagons, not districts**: administrative rayons blend elite blocks with
  industrial outskirts (the Modifiable Areal Unit Problem). We cast the
  peninsula into Uber's H3 grid — resolution 8, ~0.73 km² cells, every
  neighbour equidistant (`bakuml/spatial/`).
- **Tier 1 — spatial gradient boosting**: XGBoost with spatial-lag features
  and leakage-safe neighbour target encoding (`bakuml/models/baseline.py`),
  explained with SHAP.
- **Tier 2 — spatiotemporal GNN**: a compact STGCN over the H3 adjacency
  graph (`bakuml/models/stgcn.py`), with a spatial-lag ridge fallback when
  torch is unavailable.
- **Every prediction is an interval**: conformalized quantile regression
  outputs calibrated 10th/50th/90th percentiles
  (`bakuml/models/conformal.py`) — honest in a market spanning
  700–7,000+ AZN/m².

## Validation: random k-fold here would be academic fraud

Prices are correlated in space and time; random splits let the model copy
answers from the flat next door listed the same month. `bakuml/validation/`
enforces two isolation protocols:

- **Temporal walk-forward** — train strictly on the past, evaluate strictly on
  the future (the only split that simulates actually forecasting), always
  benchmarked against naive persistence.
- **Spatial blocked CV** — hold out entire contiguous geographic blocks
  (H3 parent cells), forcing the model to deduce economic rules rather than
  memorize neighbourhoods.

A leakage canary test perturbs all future prices and asserts that features at
earlier months are bit-identical.

## Scenario engine

Forecasts are produced under three scenarios (`bakuml/pipeline.py`):
**baseline** (trends continue), **polycentric** (Master Plan succeeds — the
node-gravity appreciation slope, *estimated from the panel's own history*, is
doubled), and **transit** (full build-out — the *DiD-estimated* premium is
applied around the next hypothetical purple-line station). Estimated
quantities only; planted truth is never reused.

## Results on the offline dataset (full run, `make full`)

| Check | Result |
|---|---|
| Walk-forward (25 splits) | MAE **210** vs naive persistence 263 AZN/m² · R² **0.71** vs 0.54 |
| Spatial blocked CV (5 folds) | MAE **209** vs naive 247 AZN/m² — the model beats naive on *every* held-out region |
| Conformal coverage (target 80 %) | **80.2 %** empirical q10–q90 coverage on the calibration months |
| Spatial DiD vs planted truth | ATT **+0.069 log** (planted ramp-averaged ≈ 0.075); event-study post-ramp mean **0.088** vs planted 0.08; pre-trends ≈ 0 |
| Dedup vs planted duplicates | **100 %** recall, 0.24 % false-positive pairs |

Numbers regenerate with `make full` (≈ 40 s on 4 CPUs); the planted ground
truth lives in `artifacts/synthetic_truth.json`.

## Repository map

```
bakuml/
├── config.py            # geography, metro timeline, study design, scenarios
├── geo.py               # vectorised haversine helpers
├── data/                # schema, synthetic feed, dedup, Scrapy project
├── spatial/             # H3 grid, spatial lags, adjacency graph
├── features/            # Master Plan 2040 + infrastructure + lag features
├── causal/              # Spatial DiD + event study
├── models/              # XGBoost+SHAP, conformal intervals, STGCN
├── validation/          # walk-forward + spatial blocked CV
├── viz.py               # pure helpers for the app
└── pipeline.py          # end-to-end orchestration -> artifacts/
app/streamlit_app.py     # interactive hexagon map + drilldowns
scripts/                 # run_pipeline / make_dataset / scrape_bina
tests/                   # offline, deterministic; recovery-based assertions
DESIGN.md                # binding module contracts
```

## Scope management

- **MVP (guaranteed)**: offline dataset → tuned XGBoost on AZN/m² + SHAP +
  spatial k-fold validation + interactive map. ✅ shipped
- **Advanced (this repo also ships)**: STGCN 12–24-month grid forecasts with
  conformal intervals, Spatial DiD, scenario engine, dedup pipeline,
  Cloudflare-aware scraper skeleton.
- **Future work**: 6 months of weekly live scrapes → longitudinal panel;
  Sentinel-2 → ResNet neighbourhood embeddings; network (not haversine)
  distances from OSM.
