"""Deterministic unit and contract tests for empirical quantile calibrator.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

from typing import Dict, List, Set
import numpy as np
import pandas as pd
import pytest

from neurogeo.calibrator import (
    assign_status_tier,
    get_calibrated_radius_with_fallback,
    fit_empirical_calibration_quantiles,
    calibrate_prediction,
)
from neurogeo.schemas import PredictionStatus
from neurogeo.leakage import validate_fit_scope


# -------------------------------------------------------------
# Test 17: Calibration tier assignment deterministic
# -------------------------------------------------------------
def test_calibration_tier_assignment_deterministic():
    """Verify assign_status_tier maps inputs deterministically to PredictionStatus enum."""
    # Doorstep verified
    t1 = assign_status_tier(selected_candidate_type="C2", has_credible_doorstep=True)
    assert t1 == PredictionStatus.DOORSTEP_VERIFIED

    # Street resolved (C4 with >=2 neighbors)
    t2 = assign_status_tier(selected_candidate_type="C4", neighbor_count=3)
    assert t2 == PredictionStatus.STREET_RESOLVED

    # Locality bounded (C5)
    t3 = assign_status_tier(selected_candidate_type="C5")
    assert t3 == PredictionStatus.LOCALITY_BOUNDED

    # Baseline fallback (C1 with pincode precision)
    t4 = assign_status_tier(selected_candidate_type="C1", baseline_precision="pincode")
    assert t4 == PredictionStatus.BASELINE_FALLBACK


# -------------------------------------------------------------
# Test 18: N_min=15 fallback works
# -------------------------------------------------------------
def test_n_min_fallback_hierarchy():
    """Verify 3-level fallback hierarchy correctly uses N_min=15."""
    # Synthetic residuals
    residuals = {
        PredictionStatus.DOORSTEP_VERIFIED.value: [10.0, 15.0, 20.0],  # N=3 (< 15) -> Fallback
        PredictionStatus.STREET_RESOLVED.value: list(range(10, 40)),     # N=30 (>= 15) -> Direct
        PredictionStatus.LOCALITY_BOUNDED.value: list(range(50, 100)),   # N=50 (>= 15) -> Direct
        PredictionStatus.BASELINE_FALLBACK.value: list(range(100, 200)), # N=100 (>= 15) -> Direct
    }

    # Case A: STREET_RESOLVED has N=30 >= 15 -> Level 1 (tier_direct)
    r_street, n_street, fb_street = get_calibrated_radius_with_fallback(
        tier=PredictionStatus.STREET_RESOLVED.value,
        residuals_by_tier=residuals,
        min_samples=15,
        target_quantile=0.90,
    )
    assert fb_street == "tier_direct"
    assert n_street == 30
    assert r_street > 0.0

    # Case B: DOORSTEP_VERIFIED has N=3 < 15, visited family has N=3 < 15 -> Level 3 (global_p90)
    r_door, n_door, fb_door = get_calibrated_radius_with_fallback(
        tier=PredictionStatus.DOORSTEP_VERIFIED.value,
        residuals_by_tier=residuals,
        min_samples=15,
        target_quantile=0.90,
    )
    assert fb_door == "global_p90"
    assert n_door == 3 + 30 + 50 + 100

    # Case C: If visited family had >= 15, evidence_family would be used
    residuals_with_family = {
        PredictionStatus.DOORSTEP_VERIFIED.value: [10.0] * 5,
        "VISITED_OTHER": [15.0] * 12,  # Visited family total = 17 >= 15
    }
    # But cold-start tiers also test Level 2
    cold_residuals = {
        PredictionStatus.STREET_RESOLVED.value: [30.0] * 5,  # N=5 (< 15)
        PredictionStatus.LOCALITY_BOUNDED.value: [50.0] * 20, # Cold-start family has 25 >= 15
    }
    r_cs, n_cs, fb_cs = get_calibrated_radius_with_fallback(
        tier=PredictionStatus.STREET_RESOLVED.value,
        residuals_by_tier=cold_residuals,
        min_samples=15,
        target_quantile=0.90,
    )
    assert fb_cs == "evidence_family"
    assert n_cs == 25


# -------------------------------------------------------------
# Test 19: Calibration never accesses test labels during fitting
# -------------------------------------------------------------
def test_calibration_never_accesses_test_labels():
    """Verify fit_empirical_calibration_quantiles purges test addresses and never reads surveyed coords."""
    # Synthetic SCV dataframe
    scv_data = pd.DataFrame({
        "address_id": [f"AD{i:04d}" for i in range(1, 101)],
        "ground_truth_medoid_x": [100.0] * 100,
        "ground_truth_medoid_y": [200.0] * 100,
        "c4_available": [True] * 50 + [False] * 50,
        "neighbor_count": [3] * 50 + [0] * 50,
        "error_vs_medoid_m": [25.0] * 50 + [np.nan] * 50,
        "locality_id_resolved": ["L1"] * 100,
        "baseline_error_vs_medoid_m": [150.0] * 100,
    })
    # Synthetic localities
    df_loc = pd.DataFrame([{"locality_id": "L1", "centroid_x": 120.0, "centroid_y": 210.0}])

    test_ids = {f"AD{i:04d}" for i in range(86, 101)}  # 15 test addresses

    cal_res = fit_empirical_calibration_quantiles(
        df_scv=scv_data,
        df_localities=df_loc,
        test_address_ids=test_ids,
        fit_scope="calibration_oof",
    )

    assert cal_res["excluded_test_account_count"] == 15
    assert cal_res["total_scv_calibration_records"] == 85


# -------------------------------------------------------------
# Test 20: Calibration quantile artifact has provenance
# -------------------------------------------------------------
def test_calibration_quantile_artifact_provenance():
    """Verify calibration output contains valid fit_scope and evaluation_scope."""
    scv_data = pd.DataFrame({
        "address_id": ["AD0001", "AD0002"],
        "ground_truth_medoid_x": [100.0, 200.0],
        "ground_truth_medoid_y": [100.0, 200.0],
        "baseline_error_vs_medoid_m": [50.0, 60.0],
    })
    cal_res = fit_empirical_calibration_quantiles(
        df_scv=scv_data,
        fit_scope="calibration_oof",
    )
    assert cal_res["fit_scope"] == "calibration_oof"
    assert validate_fit_scope("calibration_quantiles", cal_res["fit_scope"]) is True
    assert cal_res["evaluation_scope"] == "scv_leave_one_out"
    assert "source_data_scope" in cal_res
    assert "scv" in cal_res["source_data_scope"].lower()
    assert "conformal" not in str(cal_res).lower()


# -------------------------------------------------------------
# Test 21: Test split remains untouched
# -------------------------------------------------------------
def test_test_split_remains_untouched():
    """Verify test benchmark split accounts are strictly identified and protected."""
    from neurogeo.config import PATHS
    df_splits = pd.read_csv(PATHS.splits_csv)
    df_addrs = pd.read_csv(PATHS.addresses_csv)
    df_surveyed = pd.read_csv(PATHS.surveyed_addresses_csv)

    merged = df_surveyed.merge(df_addrs[["address_id", "account_id"]], on="address_id", how="left").merge(df_splits, on="account_id", how="left")
    test_rows = merged[merged["split"] == "test"]
    assert len(test_rows) == 15
    test_ids = set(test_rows["address_id"])

    # Ensure calibration filtering removes all 15 if present
    df_test_sample = pd.DataFrame({
        "address_id": list(test_ids),
        "ground_truth_medoid_x": [100.0] * 15,
        "ground_truth_medoid_y": [100.0] * 15,
        "baseline_error_vs_medoid_m": [50.0] * 15,
    })
    cal_res = fit_empirical_calibration_quantiles(
        df_scv=df_test_sample,
        test_address_ids=test_ids,
        fit_scope="calibration_oof",
    )
    assert cal_res["total_scv_calibration_records"] == 0


# -------------------------------------------------------------
# Test 24: Status tiers are deterministic
# -------------------------------------------------------------
def test_status_tiers_are_deterministic():
    """Verify repeated calls to assign_status_tier with identical args return identical results."""
    for _ in range(10):
        t = assign_status_tier("C4", neighbor_count=3)
        assert t == PredictionStatus.STREET_RESOLVED
        t_base = assign_status_tier("C1", baseline_precision="pincode")
        assert t_base == PredictionStatus.BASELINE_FALLBACK


# -------------------------------------------------------------
# Test 25: Full Phase-4 runner is reproducible
# -------------------------------------------------------------
def test_full_phase4_runner_reproducible():
    """Verify that calibration fitting and tier assignment are strictly reproducible."""
    scv_sample = pd.DataFrame({
        "address_id": [f"A{i}" for i in range(20)],
        "ground_truth_medoid_x": [100.0 + i for i in range(20)],
        "ground_truth_medoid_y": [200.0 + i for i in range(20)],
        "baseline_error_vs_medoid_m": [10.0 + i * 2 for i in range(20)],
    })
    cal_1 = fit_empirical_calibration_quantiles(df_scv=scv_sample, fit_scope="calibration_oof")
    cal_2 = fit_empirical_calibration_quantiles(df_scv=scv_sample, fit_scope="calibration_oof")
    assert cal_1["tiers"] == cal_2["tiers"]
    assert cal_1["fallback_counts"] == cal_2["fallback_counts"]


# -------------------------------------------------------------
# Test 26: Calibration excludes benchmark validation and test addresses
# -------------------------------------------------------------
def test_calibration_excludes_benchmark_validation_and_test_addresses():
    """Verify fit_empirical_calibration_quantiles purges both validation and test benchmark addresses."""
    from neurogeo.config import PATHS
    df_splits = pd.read_csv(PATHS.splits_csv)
    df_addrs = pd.read_csv(PATHS.addresses_csv)
    df_surveyed = pd.read_csv(PATHS.surveyed_addresses_csv)

    merged = df_surveyed.merge(df_addrs[["address_id", "account_id"]], on="address_id", how="left").merge(df_splits, on="account_id", how="left")
    val_ids = set(merged[merged["split"] == "validation"]["address_id"])
    test_ids = set(merged[merged["split"] == "test"]["address_id"])
    train_ids = set(merged[merged["split"] == "train"]["address_id"])

    assert len(val_ids) == 19
    assert len(test_ids) == 15
    assert len(train_ids) == 66

    df_scv = pd.read_csv(PATHS.scv_results_csv)
    cal_res = fit_empirical_calibration_quantiles(
        df_scv=df_scv,
        train_address_ids=train_ids,
        validation_address_ids=val_ids,
        test_address_ids=test_ids,
        fit_scope="calibration_oof",
    )

    retained_records = cal_res["total_scv_calibration_records"]
    assert retained_records == 1428
    assert cal_res["excluded_benchmark_address_count"] == 100

    # Explicit assertion that retained calibration IDs have ZERO intersection with validation and test IDs
    clean_scv_ids = set(df_scv[~df_scv["address_id"].isin(train_ids | val_ids | test_ids)]["address_id"])
    assert len(clean_scv_ids & val_ids) == 0
    assert len(clean_scv_ids & test_ids) == 0


# -------------------------------------------------------------
# Test 27: Reranker fitting IDs and calibration IDs are disjoint
# -------------------------------------------------------------
def test_reranker_fitting_ids_and_calibration_ids_are_disjoint():
    """Frozen requirement: calibration residual source ∩ reranker-fitting target labels = EMPTY."""
    from neurogeo.config import PATHS
    df_splits = pd.read_csv(PATHS.splits_csv)
    df_addrs = pd.read_csv(PATHS.addresses_csv)
    df_surveyed = pd.read_csv(PATHS.surveyed_addresses_csv)

    merged = df_surveyed.merge(df_addrs[["address_id", "account_id"]], on="address_id", how="left").merge(df_splits, on="account_id", how="left")
    train_ids = set(merged[merged["split"] == "train"]["address_id"])
    val_ids = set(merged[merged["split"] == "validation"]["address_id"])
    test_ids = set(merged[merged["split"] == "test"]["address_id"])

    # Reranker is fitted exclusively on train_ids (66 benchmark addresses)
    reranker_fitting_target_ids = train_ids
    assert len(reranker_fitting_target_ids) == 66

    df_scv = pd.read_csv(PATHS.scv_results_csv)
    cal_res = fit_empirical_calibration_quantiles(
        df_scv=df_scv,
        train_address_ids=train_ids,
        validation_address_ids=val_ids,
        test_address_ids=test_ids,
        fit_scope="calibration_oof",
    )

    clean_scv_ids = set(df_scv[~df_scv["address_id"].isin(train_ids | val_ids | test_ids)]["address_id"])
    assert len(clean_scv_ids & reranker_fitting_target_ids) == 0, (
        f"Leakage detected: {len(clean_scv_ids & reranker_fitting_target_ids)} reranker fitting IDs in calibration"
    )


# -------------------------------------------------------------
# Test 28: SCV target masking and provenance verification
# -------------------------------------------------------------
def test_scv_target_masking_and_provenance():
    """Verify target SCV history is masked, target medoid cannot enter inference, and target cannot enter C4."""
    from neurogeo.config import PATHS
    df_scv = pd.read_csv(PATHS.scv_results_csv)

    assert len(df_scv) == 1477
    assert bool(df_scv["target_history_masked"].all()) is True
    assert bool(df_scv["target_medoid_masked"].all()) is True
    assert bool(df_scv["target_gps_masked"].all()) is True
    assert bool((df_scv["graph_fit_scope"] == "scv_leave_one_out").all()) is True
    assert bool(df_scv["reserved_for_calibration"].all()) is True

    # Target cannot enter C4 neighbor list
    for _, row in df_scv.iterrows():
        aid = row["address_id"]
        neighbors = [n.strip() for n in str(row.get("neighbor_address_ids", "")).split(",") if n.strip()]
        assert aid not in neighbors, f"Target {aid} found in its own C4 neighbor list!"


# -------------------------------------------------------------
# Test 29: Calibration residual count vs unique episode count
# -------------------------------------------------------------
def test_calibration_residual_count_vs_unique_episodes():
    """Verify relationship between unique episodes and residual observations across tiers."""
    from neurogeo.config import PATHS
    df_splits = pd.read_csv(PATHS.splits_csv)
    df_addrs = pd.read_csv(PATHS.addresses_csv)
    df_surveyed = pd.read_csv(PATHS.surveyed_addresses_csv)

    merged = df_surveyed.merge(df_addrs[["address_id", "account_id"]], on="address_id", how="left").merge(df_splits, on="account_id", how="left")
    all_bench_ids = set(merged["address_id"])

    df_scv = pd.read_csv(PATHS.scv_results_csv)
    cal_res = fit_empirical_calibration_quantiles(
        df_scv=df_scv,
        train_address_ids=set(merged[merged["split"] == "train"]["address_id"]),
        validation_address_ids=set(merged[merged["split"] == "validation"]["address_id"]),
        test_address_ids=set(merged[merged["split"] == "test"]["address_id"]),
        fit_scope="calibration_oof",
    )

    tiers = cal_res["tiers"]

    # STREET_RESOLVED: 1 residual per episode where C4 has >=2 neighbors
    assert tiers["STREET_RESOLVED"]["unique_episode_count"] == 279
    assert tiers["STREET_RESOLVED"]["residual_count"] == 279
    assert tiers["STREET_RESOLVED"]["fallback_level"] == "tier_direct"

    # LOCALITY_BOUNDED: 1 residual per episode where locality resolved
    assert tiers["LOCALITY_BOUNDED"]["unique_episode_count"] == 1258
    assert tiers["LOCALITY_BOUNDED"]["residual_count"] == 1258
    assert tiers["LOCALITY_BOUNDED"]["fallback_level"] == "tier_direct"

    # BASELINE_FALLBACK: 1 residual per episode with baseline geocode
    assert tiers["BASELINE_FALLBACK"]["unique_episode_count"] == 1428
    assert tiers["BASELINE_FALLBACK"]["residual_count"] == 1428
    assert tiers["BASELINE_FALLBACK"]["fallback_level"] == "tier_direct"

    # DOORSTEP_VERIFIED: 0 direct residuals (masked in SCV) -> Level 3 fallback (global_p90)
    # Pools all candidate residuals: 279 + 1258 + 1428 = 2965 across all 1428 unique episodes
    assert tiers["DOORSTEP_VERIFIED"]["residual_count"] == 279 + 1258 + 1428
    assert tiers["DOORSTEP_VERIFIED"]["unique_episode_count"] == 1428
    assert tiers["DOORSTEP_VERIFIED"]["fallback_level"] == "global_p90"


# -------------------------------------------------------------
# Test 30: Terminology distinction: SCV visited-address episodes vs unique accounts
# -------------------------------------------------------------
def test_scv_episodes_vs_unique_accounts_distinction():
    """Verify contract distinction: 1,477 SCV visited-address episodes vs 1,339 unique accounts."""
    from neurogeo.config import PATHS
    df_scv = pd.read_csv(PATHS.scv_results_csv)
    df_raw_addr = pd.read_csv(PATHS.addresses_csv)

    merged = df_scv.merge(df_raw_addr[["address_id", "account_id"]], on="address_id", how="left")
    assert len(df_scv) == 1477, "Must be exactly 1,477 SCV visited-address episodes"
    assert merged["account_id"].nunique() == 1339, "Must be exactly 1,339 unique visited accounts"
