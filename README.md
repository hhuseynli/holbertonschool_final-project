# Beyond Price Prediction

**A spatiotemporal ML study of Baku & Absheron — urban expansion, infrastructure
causality, and where the market moves next.**

Holberton School · ML Graduation Project · Group 3
Hüseyn Hüseynli · Nihad Süleymanov · Eldəniz Arifzadə · Zeynəb Mirzəzadə · Vüqar Dünyamalıyev

---

## Why this is not another price regressor

| | A typical price model | This project |
|---|---|---|
| **Unit of analysis** | A listing (rows in a table) | Spatial cells over the whole peninsula — geometry chosen by measurement, not assumed |
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
make maup          # compare units of analysis (H3 vs KD-tree vs market regions)
```

`make demo` runs every stage — data → dedup → tessellation → cell panel → features →
leakage-proof validation → SHAP → conformal intervals → STGCN forecast →
Spatial DiD → artifacts — and drops the results into `artifacts/`, which the
Streamlit app reads. Click any cell on the map to see its price history and
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

## Spatial architecture: the unit of analysis is measured, not assumed

Administrative rayons blend elite blocks with industrial outskirts — the
Modifiable Areal Unit Problem (MAUP). Uber's H3 hexagons fix that much:
uniform area, equidistant neighbours, a parent hierarchy for CV blocks. But
hexagons only make the arbitrariness *uniform*, and a uniform grid over a
radically non-uniform city is costly. On this panel a res-8 grid discards
**37 %** of deduplicated listings to the thinness filter and leaves the
median cell just **5 months** of price history — which is why forecasting
from those cells was unstable.

So the tessellation is a **parameter, not an assumption**
(`bakuml/spatial/tessellation.py`), scored under the same leakage-proof
protocols as everything else (`make maup`):

| unit of analysis | listings kept | median months of history | spatial-transfer skill vs naive |
|---|---|---|---|
| H3 res 7 | 93.7 % | 33 | **−0.174** |
| H3 res 8 *(documented baseline)* | 62.6 % | 5 | +0.139 |
| H3 res 9 | 11.3 % | 2 | **−0.339** |
| Adaptive KD-tree, ~70/cell | 46.1 % | 9 | **+0.204** |
| **Adaptive KD-tree, ~140/cell** *(default)* | **81.6 %** | **26** | **+0.181** |
| Adaptive KD-tree, ~280/cell | 98.5 % | 42 | +0.176 |
| Market regions, k=120 | 94.5 % | 14 | −0.251 |
| Market regions, k=240 | 88.4 % | 8 | +0.011 |

*Skill = 1 − model MAE / naive-persistence MAE on spatially blocked CV;
positive means the model beat "next month = last month" on regions it had
never seen.*

The decisive result is not one winning row — it is **stability across each
family's own tuning knob**, which the study reports explicitly:

| family | spatial-transfer skill range | verdict |
|---|---|---|
| H3 | −0.339 … +0.139 (spread 0.478) | **sign flips** — beats naive at one resolution of three |
| Market regions | −0.251 … +0.011 (spread 0.262) | **sign flips** — never convincingly beats naive |
| **Adaptive KD-tree** | **+0.176 … +0.204 (spread 0.028)** | positive at every setting |

"Why resolution 8?" has no answer but the winning number: at res 7 and res 9
the hexagonal model is *worse than assuming next month equals last month* on
regions it has not seen. The adaptive grid never is.

Market regions are the honest negative result of the study. They cut the map
where prices actually change — at matched granularity their within-cell
variance is marginally lower than hexagons' — but that homogeneity does not
convert into transfer skill, and the sign flip across `k` means the method
cannot be defended at any particular setting. They ship anyway, because a
comparison that only contained the winner would not be a comparison. One
caveat the study prints and this table cannot: the market geography is
leakage-free in *time* (prices are cut off before the evaluation window) but
its boundaries are still drawn using pre-cutoff prices of cells that spatial
CV later holds out, so its skill is not strictly comparable with the
coordinate-only candidates.

Three tessellations ship behind one contract (assign / centroid /
neighbours / boundary / block), so every downstream module is agnostic:

- **`h3`** — the documented hexagonal baseline, at any resolution.
- **`kdtree`** (default) — recursive median splits of the listing cloud, so
  every cell holds ~the same number of listings: homogeneous estimation
  variance, far less data discarded, and real price histories in the
  periphery. Splits use **coordinates only**, never prices, so no target
  information enters the geography. CV blocks are KD path prefixes, which
  are rectangles — contiguous by construction.
- **`market`** — contiguity-constrained Ward clustering on price level and
  building mix, so boundaries follow actual market discontinuities. The one
  price-driven option, therefore fitted strictly on a pre-cutoff window.

Switch with `--tessellation {h3,kdtree,market}` or `config.TESSELLATION`.
Cell polygons ship in the `cell_geometry.geojson` artifact, so the app draws
whatever geometry a run used without refitting it.

- **Cells, not districts** (`bakuml/spatial/`): whichever tessellation is
  active, listings aggregate to one row per (cell, month) with leakage-safe
  neighbour features over its contiguity graph. Cells whose centroid falls
  in the Caspian are dropped (`filter_land_cells`), and `zones.py` groups
  cells into organic market micro-zones for the map — both now work over any
  tessellation, and cluster per connected component so a "zone" is always
  one place.
- **Tier 1 — spatial gradient boosting**: XGBoost with spatial-lag features
  and leakage-safe neighbour target encoding (`bakuml/models/baseline.py`),
  explained with SHAP.
- **Tier 2 — spatiotemporal GNN**: a compact STGCN over the cell adjacency
  graph (`bakuml/models/stgcn.py`), with a spatial-lag ridge fallback when
  torch is unavailable.
- **Every prediction is an interval**: conformalized quantile regression
  outputs calibrated 10th/50th/90th percentiles
  (`bakuml/models/conformal.py`).

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
| Unit of analysis | adaptive KD-tree, 256 cells of ~137 listings each (chosen by `make maup`) |
| Panel size | **6,799** usable cell-months / 3,953 training rows (H3 res-8: 4,319 / 2,600) |
| Walk-forward (25 splits) | MAE **213** vs naive persistence 264 AZN/m² · R² **0.81** vs 0.70 |
| Spatial blocked CV (5 folds) | MAE **207** vs naive 253 AZN/m² · R² **0.81** vs 0.72 — cross-cell features excluded so held-out regions stay airtight |
| Conformal coverage (target 80 %) | **70.7 %** empirical q10–q90 coverage on a final 3-month holdout never used for training *or* calibration — under target and reported as measured, not tuned: CQR assumes exchangeability, which a trending market violates |
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
├── spatial/             # tessellations, cell panel, lags, graph, zones
├── features/            # Master Plan 2040 + infrastructure + lag features
├── causal/              # Spatial DiD + event study
├── models/              # XGBoost+SHAP, conformal intervals, STGCN
├── validation/          # walk-forward + spatial blocked CV
├── viz.py               # pure helpers for the app
└── pipeline.py          # end-to-end orchestration -> artifacts/
app/streamlit_app.py     # interactive cell/zone map + drilldowns
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
