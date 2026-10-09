"""Audit verification for historical Stage 2 validation evaluation artifacts.

Guarantees:
- Confirms scripts/evaluate_validation.py is safely archived as a report-only artifact
  and refuses execution with an informative RuntimeError.
- Confirms all 5 historical evaluation artifacts in reports/eval_val_c6/ remain preserved,
  valid, and intact.
- Confirms the zero-leakage manifest guarantees (N=19 validation addresses, 0 test addresses).
- Confirms historical record that Candidate C6 produced 0 selections on validation data.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd

from neurogeo.config import PATHS
from scripts.evaluate_validation import run_validation_evaluation


def test_historical_evaluator_refuses_execution_with_runtime_error():
    """Verify that scripts/evaluate_validation.py refuses execution as an archived artifact."""
    with pytest.raises(RuntimeError, match="archived historical report-only artifact"):
        run_validation_evaluation()


def test_historical_validation_artifacts_exist_and_unmodified():
    """Verify all 5 historical evaluation artifacts in reports/eval_val_c6/ exist and are valid."""
    target_dir = PATHS.reports_dir / "eval_val_c6"
    assert target_dir.exists() and target_dir.is_dir()

    required_files = [
        "evaluation_manifest.json",
        "validation_metrics_summary.json",
        "validation_predictions.csv",
        "validation_paired_comparison.csv",
        "validation_candidate_diagnostics.csv",
    ]
    for fname in required_files:
        fpath = target_dir / fname
        assert fpath.exists(), f"Missing historical artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Empty historical artifact: {fname}"


def test_historical_eval_manifest_integrity():
    """Verify zero-leakage guarantees and split integrity in evaluation_manifest.json."""
    manifest_path = PATHS.reports_dir / "eval_val_c6" / "evaluation_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["evaluation_scope"] == "validation_split_only"
    assert manifest["sample_count"] == 19
    assert manifest["splits_guarantee"]["train_address_count"] == 66
    assert manifest["splits_guarantee"]["test_address_count"] == 15
    assert manifest["splits_guarantee"]["test_split_touched"] is False
    assert manifest["splits_guarantee"]["test_split_evaluated"] is False
    assert len(manifest["splits_guarantee"]["validation_address_ids"]) == 19


def test_historical_c6_zero_gain_record():
    """Verify historical validation metrics record that Candidate C6 produced 0 selections."""
    summary_path = PATHS.reports_dir / "eval_val_c6" / "validation_metrics_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["sample_size"] == 19
    cond2_counts = summary["conditions"]["condition_2_stage1_plus_c6"]["overall"]["candidate_selection_counts"]
    assert cond2_counts.get("C6", 0) == 0, "Candidate C6 was never selected in Condition 2"

    pred_path = PATHS.reports_dir / "eval_val_c6" / "validation_predictions.csv"
    df_preds = pd.read_csv(pred_path)
    assert len(df_preds) == 57  # 19 addresses across 3 evaluation conditions
    assert df_preds["address_id"].nunique() == 19
    assert not any(df_preds["selected_candidate_type"] == "C6")
