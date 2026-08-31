#!/usr/bin/env python3
"""Compare units of analysis under the project's leakage-proof validation.

The Modifiable Areal Unit Problem says results depend on how space is
carved up. The spec argues H3 hexagons beat administrative rayons, which is
true - but hexagons are still an arbitrary geometry imposed on a
non-uniform city. This script settles the choice with evidence instead of
doctrine: every candidate tessellation is run through the *same* pipeline
stages and scored on the same two isolation protocols (temporal
walk-forward and spatially blocked CV), always against naive persistence.

Reported per candidate:

* ``retained_pct``      - share of deduplicated listings that survive the
  ``MIN_LISTINGS_PER_CELL_MONTH`` thinness filter (H3's big loss);
* ``median_months``     - median months of price history per cell, which is
  what autoregressive forecasting actually needs;
* ``within_var_share``  - mean within-cell log-price variance / total, the
  direct MAUP diagnostic (lower = cells are internally homogeneous, so a
  cell median means something);
* ``wf_mae`` / ``scv_mae`` and their naive counterparts, plus
  ``skill`` = 1 - model MAE / naive MAE on each protocol.

Usage
-----
    python scripts/maup_study.py                 # full sweep
    python scripts/maup_study.py --fast          # smaller dataset
    python scripts/maup_study.py --only h3_res8 kdtree_n120
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from bakuml import config
from bakuml.data.dedup import dedupe
from bakuml.data.synthetic import generate_listings
from bakuml.features.build import FEATURE_COLS, build_feature_matrix
from bakuml.models.baseline import evaluate_spatial_cv, evaluate_walk_forward
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.grid import (
    assign_cells,
    build_cell_month_panel,
    filter_land_cells,
)

# Candidates: the spec baseline at three resolutions (a resolution
# sensitivity analysis is itself a MAUP diagnostic), plus the two
# data-driven alternatives at comparable cell counts.
def candidates() -> dict:
    return {
        "h3_res7": tess_mod.H3Tessellation(7, 5),
        "h3_res8": tess_mod.H3Tessellation(8, 6),
        "h3_res9": tess_mod.H3Tessellation(9, 7),
        # Targets chosen to land in distinct partitions: KD leaf sizes are
        # quantised to n/2**depth, so nearby targets collapse to one tree.
        "kdtree_n70": tess_mod.AdaptiveKDTessellation(70),
        "kdtree_n140": tess_mod.AdaptiveKDTessellation(140),
        "kdtree_n280": tess_mod.AdaptiveKDTessellation(280),
        "market_k120": tess_mod.MarketRegionTessellation(120),
        "market_k240": tess_mod.MarketRegionTessellation(240),
    }


def within_cell_variance_share(listings: pd.DataFrame) -> float:
    """Within-cell share of total log-price variance (proper decomposition).

    The core MAUP question: do the cells carve the market at its joints?
    Cells that mix an elite block with an industrial strip have high
    internal variance, so their median describes nothing real.

    This must be a *variance decomposition*, not a mean of per-cell
    variances: candidates differ systematically in cell size, so an
    unweighted mean would measure the cell-size mix as much as the
    geography, and would silently drop every single-listing cell (whose
    sample variance is undefined) while letting a 2-listing cell count as
    much as a 200-listing one. Both numerator and denominator use the same
    estimator, so the ratio is comparable across candidates.

        within_share = sum_i (n_i - 1) * var_i / ((N - 1) * total_var)
    """
    logp = np.log(listings["price_azn_m2"].to_numpy(dtype=float))
    d = pd.DataFrame({"h3": listings["h3"].to_numpy(), "logp": logp})
    n_total = len(d)
    total_var = float(d["logp"].var(ddof=1))
    if n_total < 2 or not np.isfinite(total_var) or total_var <= 0:
        return float("nan")
    grp = d.groupby("h3")["logp"]
    counts = grp.size().to_numpy(dtype=float)
    variances = grp.var(ddof=1).to_numpy(dtype=float)
    ok = counts >= 2  # single-listing cells contribute 0 dof, not NaN
    ss_within = float(((counts[ok] - 1.0) * variances[ok]).sum())
    return ss_within / ((n_total - 1.0) * total_var)


def evaluate(name: str, tess, clean: pd.DataFrame, *, seed: int, cutoff: str) -> dict:
    """Score one tessellation end to end."""
    tess.fit(clean, price_cutoff_month=cutoff)
    assigned = assign_cells(clean, tess=tess)
    # Mirror run_pipeline exactly: it drops cells centred at sea before
    # building the panel, so a study that skipped this would be choosing a
    # default by scoring a pipeline nobody runs.
    assigned = filter_land_cells(assigned, tess=tess)
    panel = build_cell_month_panel(assigned)

    # Retention is measured against the pre-filter listing count, so the
    # sea filter's cost is visible rather than hidden in the denominator.
    retained = panel["n_listings"].sum() / len(clean) * 100.0
    months_per_cell = panel.groupby("h3")["month"].nunique()

    row = {
        "tessellation": name,
        **{k: v for k, v in tess.describe().items() if k not in {"name"}},
        "n_cells": int(panel["h3"].nunique()),
        "panel_rows": int(len(panel)),
        "retained_pct": round(float(retained), 1),
        "median_months": int(months_per_cell.median()),
        "cells_ge_12_months_pct": round(
            float((months_per_cell >= 12).mean() * 100.0), 1
        ),
        "within_var_share": round(within_cell_variance_share(assigned), 4),
    }

    # The feature matrix and both validation protocols need the tessellation
    # active (neighbour encodings, adjacency, CV blocks all consult it).
    previous = tess_mod.set_active(tess)
    try:
        fm = build_feature_matrix(panel)
        row["feature_rows"] = int(len(fm))
        wf = evaluate_walk_forward(fm, FEATURE_COLS, seed=seed)
        scv = evaluate_spatial_cv(
            fm, FEATURE_COLS, seed=seed,
            exclude_features=["nbr_price_prev_month"],
        )
        for tag, res in (("wf", wf), ("scv", scv)):
            row[f"{tag}_mae"] = round(res["aggregate"]["mae"], 1)
            row[f"{tag}_naive_mae"] = round(res["naive"]["mae"], 1)
            row[f"{tag}_r2"] = round(res["aggregate"]["r2"], 3)
            naive = res["naive"]["mae"]
            row[f"{tag}_skill"] = (
                round(1.0 - res["aggregate"]["mae"] / naive, 3)
                if naive and np.isfinite(naive) else float("nan")
            )
    finally:
        tess_mod.set_active(previous)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="smaller dataset")
    ap.add_argument("--only", nargs="+", help="subset of candidate names")
    ap.add_argument("--seed", type=int, default=config.SYNTHETIC_TRUTH.seed)
    ap.add_argument(
        "--out",
        default=str(config.ARTIFACTS_DIR / "maup_study.json"),
        help="where to write the comparison table",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    truth = config.SYNTHETIC_TRUTH
    if args.fast:
        truth = replace(truth, n_listings_per_month=400)
    print("generating + deduplicating listings...", flush=True)
    listings, _ = generate_listings(truth=truth)
    clean, _ = dedupe(listings)

    # Price-driven tessellations see only pre-cutoff months.
    all_months = sorted(clean["listed_month"].unique())
    hold = config.TESS_PRICE_HOLDOUT_MONTHS
    cutoff = all_months[-hold] if len(all_months) > hold else None

    cands = candidates()
    if args.only:
        missing = [n for n in args.only if n not in cands]
        if missing:
            raise SystemExit(f"unknown candidates {missing}; have {sorted(cands)}")
        cands = {n: cands[n] for n in args.only}

    rows = []
    for name, tess in cands.items():
        print(f"--- {name} ---", flush=True)
        try:
            row = evaluate(name, tess, clean, seed=args.seed, cutoff=cutoff)
        except Exception as exc:  # a candidate may be infeasible on this data
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"tessellation": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(
            f"    cells={row['n_cells']:4d}  retained={row['retained_pct']:5.1f}%  "
            f"median_months={row['median_months']:2d}  "
            f"within_var={row['within_var_share']:.3f}  "
            f"wf_mae={row['wf_mae']:7.1f} (skill {row['wf_skill']:+.3f})  "
            f"scv_mae={row['scv_mae']:7.1f} (skill {row['scv_skill']:+.3f})",
            flush=True,
        )

    table = pd.DataFrame(rows)
    cols = [
        "tessellation", "n_cells", "retained_pct", "median_months",
        "cells_ge_12_months_pct", "within_var_share",
        "wf_mae", "wf_naive_mae", "wf_skill",
        "scv_mae", "scv_naive_mae", "scv_skill",
    ]
    present = [c for c in cols if c in table.columns]
    print("\n=== MAUP comparison (leakage-proof protocols) ===")
    print(table[present].to_string(index=False))

    ok = table[table["wf_skill"].notna()] if "wf_skill" in table else table
    if len(ok):
        # Spatial-transfer skill is the headline metric - it measures
        # whether the model learned economics or memorised neighbourhoods.
        # But skill alone can crown a tessellation whose cells are too
        # data-starved to forecast from, so candidates must first clear a
        # viability floor: keep most of the data, and give the median cell
        # enough history for a 12-month rollout.
        viable = ok[(ok["retained_pct"] >= 60.0) & (ok["median_months"] >= 12)]
        print("\n--- recommendation ---")
        if viable.empty:
            print("No candidate clears the viability floor "
                  "(retained >= 60%, median history >= 12 months).")
        else:
            best = viable.sort_values("scv_skill", ascending=False).iloc[0]
            print(
                f"Recommended: {best['tessellation']}  "
                f"(scv_skill {best['scv_skill']:+.3f}, wf_skill "
                f"{best['wf_skill']:+.3f}, retains {best['retained_pct']}% of "
                f"listings, median {best['median_months']} months of history)"
            )
        starved = ok[~ok.index.isin(viable.index)]["tessellation"].tolist()
        if starved:
            print(f"Excluded as unviable: {', '.join(starved)}")

        if any(t.startswith("market") for t in ok["tessellation"]):
            print(
                "\nCaveat on market_*: the price cutoff makes the geography "
                "leakage-free in TIME, but regions are still drawn using the "
                "pre-cutoff prices of every cell, including cells that "
                "spatial CV later holds out. Its scv_skill is therefore not "
                "directly comparable with the coordinate-only candidates."
            )

        # Stability across a tessellation's own tuning knob matters as much
        # as any single score: a family whose skill flips sign with an
        # arbitrary parameter cannot be defended, whichever value wins.
        print("\n--- stability across each family's tuning parameter ---")
        for family in ("h3", "kdtree", "market"):
            fam = ok[ok["tessellation"].str.startswith(family)]
            if len(fam) < 2:
                continue
            lo, hi = fam["scv_skill"].min(), fam["scv_skill"].max()
            flips = (fam["scv_skill"] > 0).nunique() > 1
            print(
                f"{family:8s} scv_skill range [{lo:+.3f}, {hi:+.3f}] "
                f"spread {hi - lo:.3f}"
                + ("  <-- SIGN FLIPS: beats naive at some settings, "
                   "loses at others" if flips else "")
            )

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
