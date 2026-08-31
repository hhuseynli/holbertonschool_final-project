#!/usr/bin/env python3
"""Run the full pipeline and write app artifacts.

Examples
--------
python scripts/run_pipeline.py --fast            # quick demo (~2-4 min)
python scripts/run_pipeline.py                   # full run
python scripts/run_pipeline.py --forecaster ridge
python scripts/run_pipeline.py --tessellation h3      # hexagonal baseline
"""

import argparse
import json

from bakuml import config
from bakuml.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="smaller dataset + fewer training epochs")
    ap.add_argument("--forecaster", choices=["stgcn", "ridge"], default="stgcn",
                    help="Tier-2 forecaster (ridge = no-torch fallback)")
    ap.add_argument("--scenarios", nargs="+", default=list(config.SCENARIOS),
                    choices=list(config.SCENARIOS))
    ap.add_argument(
        "--tessellation", choices=["h3", "kdtree", "market"],
        default=config.TESSELLATION,
        help="unit of analysis (default from config.TESSELLATION); "
             "compare options with scripts/maup_study.py",
    )
    args = ap.parse_args()

    metrics = run_pipeline(
        scenarios=tuple(args.scenarios),
        fast=args.fast,
        prefer_forecaster=args.forecaster,
        tessellation=args.tessellation,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
