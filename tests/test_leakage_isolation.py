"""Tests for Zero-Leakage Isolation, Closure(A*), and Provenance Verification.

Authority: PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

import pandas as pd
import pytest

from neurogeo.leakage import (
    compute_isolation_closure,
    validate_fit_scope,
    assert_fit_scope_valid,
    check_artifact_for_leakage,
    assert_no_leakage,
)
from neurogeo.evaluation import evaluate_commercial_baseline


def test_target_address_closure_exclusion():
    """Verify that Closure(A*) strictly purges all visits and GPS points for target address."""
    # Synthetic test data
    df_addresses = pd.DataFrame([
        {"address_id": "AD001", "account_id": "AC001"},
        {"address_id": "AD002", "account_id": "AC002"},
        {"address_id": "AD003", "account_id": "AC003"},
    ])
    df_visits = pd.DataFrame([
        {"visit_id": "VS001", "address_id": "AD001"},
        {"visit_id": "VS002", "address_id": "AD001"},
        {"visit_id": "VS003", "address_id": "AD002"},
    ])
    df_gps = pd.DataFrame([
        {"visit_id": "VS001", "seq": 0, "x": 10.0, "y": 20.0},
        {"visit_id": "VS001", "seq": 1, "x": 11.0, "y": 21.0},
        {"visit_id": "VS002", "seq": 0, "x": 15.0, "y": 25.0},
        {"visit_id": "VS003", "seq": 0, "x": 50.0, "y": 60.0},
    ])

    closure = compute_isolation_closure("AD001", df_visits, df_addresses)

    assert closure.target_address_id == "AD001"
    assert closure.target_account_id == "AC001"
    assert closure.purged_visit_ids == {"VS001", "VS002"}

    # Filtered tables must not contain AD001, VS001, or VS002
    filtered_addr = closure.filter_addresses(df_addresses)
    assert "AD001" not in set(filtered_addr["address_id"])
    assert len(filtered_addr) == 2

    filtered_visits = closure.filter_visits(df_visits)
    assert "AD001" not in set(filtered_visits["address_id"])
    assert set(filtered_visits["visit_id"]) == {"VS003"}

    filtered_gps = closure.filter_gps_points(df_gps)
    assert set(filtered_gps["visit_id"]) == {"VS003"}
    assert len(filtered_gps) == 1


def test_neighbor_and_centroid_leakage_assertion():
    """Verify that assert_no_leakage detects when target address or its visits contaminate an artifact."""
    target_addr = "AD000527"
    target_visits = {"VS000101", "VS000102"}

    # 1. Clean neighbor aggregate: Should pass
    clean_neighbors = pd.DataFrame([
        {"neighbor_address_id": "AD000100", "weight": 1.5},
        {"neighbor_address_id": "AD000101", "weight": 2.0},
    ])
    assert_no_leakage(target_addr, clean_neighbors, target_visits)

    # 2. Contaminated neighbor aggregate: Should fail
    contaminated_neighbors = pd.DataFrame([
        {"neighbor_address_id": "AD000100", "weight": 1.5},
        {"neighbor_address_id": target_addr, "weight": 2.0},
    ])
    with pytest.raises(AssertionError, match="LEAKAGE CONTRACT VIOLATION"):
        assert_no_leakage(target_addr, contaminated_neighbors, target_visits)

    # 3. Contaminated street centroid using target visit: Should fail
    contaminated_visits = pd.DataFrame([
        {"visit_id": "VS000101", "x": 100.0, "y": 200.0},
    ])
    with pytest.raises(AssertionError, match="LEAKAGE CONTRACT VIOLATION"):
        assert_no_leakage(target_addr, contaminated_visits, target_visits)

    # 4. Contaminated dictionary artifact: Should fail
    contaminated_dict = {target_addr: (120.0, 340.0)}
    with pytest.raises(AssertionError, match="LEAKAGE CONTRACT VIOLATION"):
        assert_no_leakage(target_addr, contaminated_dict)


def test_fit_scope_provenance_validation():
    """Verify acceptance of valid fit_scope tags and rejection of invalid tags."""
    # Valid fit scopes
    assert validate_fit_scope("visit_features", "unsupervised_all") is True
    assert validate_fit_scope("street_centroids", "train_fold_0") is True
    assert validate_fit_scope("neighbor_graph", "train_fold_1") is True
    assert validate_fit_scope("calibration_quantiles", "calibration_oof") is True
    assert validate_fit_scope("calibration_quantiles", "calibration_oof_0") is True

    assert_fit_scope_valid("visit_features", "unsupervised_all")
    assert_fit_scope_valid("street_centroids", "train_fold_2")

    # Invalid fit scopes
    assert validate_fit_scope("street_centroids", "test_split") is False
    assert validate_fit_scope("calibration_quantiles", "val_set") is False
    assert validate_fit_scope("unknown_artifact", "anything") is False

    with pytest.raises(ValueError, match="Invalid fit_scope"):
        assert_fit_scope_valid("street_centroids", "leaked_all_data")


def test_baseline_evaluation_determinism():
    """Verify that repeated evaluation produces 100% deterministic identical outputs."""
    report1 = evaluate_commercial_baseline(write_report=False)
    report2 = evaluate_commercial_baseline(write_report=False)

    assert report1["overall"] == report2["overall"]
    assert report1["by_split"] == report2["by_split"]
    assert report1["by_precision"] == report2["by_precision"]
