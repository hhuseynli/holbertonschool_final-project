"""End-to-end smoke test: the full pipeline writes coherent artifacts.

Marked slow-ish (a few minutes in fast mode); run explicitly with
`pytest tests/test_pipeline.py` or as part of the full suite.
"""

import json

import pandas as pd
import pytest

from bakuml import config
from bakuml.data.schema import ARTIFACT_FILES, PREDICTION_COLUMNS


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    from bakuml.pipeline import run_pipeline

    outdir = tmp_path_factory.mktemp("artifacts")
    metrics = run_pipeline(
        scenarios=("baseline", "transit"),
        fast=True,
        prefer_forecaster="ridge",  # keep CI light; STGCN covered in test_stgcn
        outdir=outdir,
        verbose=False,
    )
    return outdir, metrics


def test_all_artifacts_written(artifacts):
    outdir, _ = artifacts
    for name in ARTIFACT_FILES.values():
        assert (outdir / name).exists(), f"missing artifact {name}"


def test_predictions_schema_and_sanity(artifacts):
    outdir, _ = artifacts
    preds = pd.read_parquet(outdir / ARTIFACT_FILES["predictions"])
    assert list(preds.columns) == list(PREDICTION_COLUMNS)
    assert set(preds["scenario"].unique()) == {"baseline", "transit"}
    assert set(preds["horizon_months"].unique()) == set(config.FORECAST_HORIZONS)
    # calibrated interval geometry
    assert (preds["q10"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q90"]).all()
    # prices stay inside a sane market envelope
    assert preds["q50"].between(300, 20_000).all()
    assert preds["hotspot_prob"].between(0, 1).all()


def test_metrics_meaningful(artifacts):
    _, metrics = artifacts
    agg = metrics["walk_forward"]["aggregate"]
    naive = metrics["walk_forward"]["naive"]
    assert agg["mae"] > 0
    # the model must not be wildly worse than persistence
    assert agg["mae"] < naive["mae"] * 1.10
    cov = metrics["conformal"]["coverage_q10_q90"]
    assert 0.6 <= cov <= 1.0


def test_did_artifact_recovers_planted_effect(artifacts):
    outdir, _ = artifacts
    did = json.loads((outdir / ARTIFACT_FILES["did"]).read_text())
    planted = config.SYNTHETIC_TRUTH.did_effect_log
    # fast mode uses a smaller sample; allow a generous but bounded window
    assert did["att_log"] == pytest.approx(planted, abs=0.04)
    assert did["p_value"] < 0.05
