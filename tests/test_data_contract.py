"""Tests for Data Contract Validation & Source Table Integrity.

Authority: PS3_DATA_CONTRACT.md and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

from pathlib import Path
import pandas as pd
import pytest

from neurogeo.config import PATHS
from neurogeo.evaluation import load_benchmark_ground_truth, evaluate_commercial_baseline


def test_required_tables_exist():
    """Verify all 8 core PS3 tables exist at expected repository-relative paths."""
    required_files = [
        PATHS.addresses_csv,
        PATHS.baseline_geocodes_csv,
        PATHS.field_visits_csv,
        PATHS.visit_gps_points_csv,
        PATHS.landmarks_poi_csv,
        PATHS.localities_csv,
        PATHS.towns_csv,
        PATHS.surveyed_addresses_csv,
        PATHS.splits_csv,
    ]
    for file_path in required_files:
        assert file_path.exists(), f"Required table missing: {file_path}"
        assert file_path.is_file(), f"Path is not a regular file: {file_path}"
        assert file_path.stat().st_size > 0, f"Table file is empty: {file_path}"


def test_source_tables_row_counts_and_keys():
    """Validate primary keys, row counts, and uniqueness constraints from PS3_DATA_CONTRACT.md."""
    # 1. Addresses
    df_addr = pd.read_csv(PATHS.addresses_csv)
    assert len(df_addr) == 3117
    assert df_addr["address_id"].is_unique
    assert df_addr["address_id"].notna().all()
    assert set(df_addr["town_id"].unique()) == {"T1", "T2", "T3", "OUT"}

    # 2. Baseline geocodes (in-territory only: 3117 - 237 'OUT' = 2880)
    df_base = pd.read_csv(PATHS.baseline_geocodes_csv)
    assert len(df_base) == 2880
    assert df_base["address_id"].is_unique
    assert set(df_base["precision"].unique()).issubset({"rooftop", "street", "locality", "pincode"})

    # 3. Surveyed benchmark
    df_surv = pd.read_csv(PATHS.surveyed_addresses_csv)
    assert len(df_surv) == 100
    assert df_surv["address_id"].is_unique

    # 4. Field visits
    df_visits = pd.read_csv(PATHS.field_visits_csv)
    assert len(df_visits) == 5578
    assert df_visits["visit_id"].is_unique
    assert df_visits["address_id"].nunique() == 1477

    # 5. Localities and Towns
    df_loc = pd.read_csv(PATHS.localities_csv)
    assert len(df_loc) == 36
    assert df_loc["locality_id"].is_unique

    df_towns = pd.read_csv(PATHS.towns_csv)
    assert len(df_towns) == 3
    assert set(df_towns["town_id"]) == {"T1", "T2", "T3"}

    # 6. Landmarks POI
    df_poi = pd.read_csv(PATHS.landmarks_poi_csv)
    assert len(df_poi) == 240
    assert df_poi["poi_id"].is_unique


def test_official_benchmark_partition():
    """Validate 100 surveyed addresses split into 66 train, 19 val, 15 test with zero overlap."""
    df_bench = load_benchmark_ground_truth()
    assert len(df_bench) == 100

    train_ids = set(df_bench[df_bench["split"] == "train"]["address_id"])
    val_ids = set(df_bench[df_bench["split"].isin(["validation", "val"])]["address_id"])
    test_ids = set(df_bench[df_bench["split"] == "test"]["address_id"])

    assert len(train_ids) == 66, f"Expected 66 train, got {len(train_ids)}"
    assert len(val_ids) == 19, f"Expected 19 val, got {len(val_ids)}"
    assert len(test_ids) == 15, f"Expected 15 test, got {len(test_ids)}"

    # Zero overlap check
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val splits overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test splits overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test splits overlap!"


def test_benchmark_visited_breakdown():
    """Verify visited vs cold-start breakdown in benchmark (49 visited, 51 unvisited)."""
    df_bench = load_benchmark_ground_truth()
    visited_count = int(df_bench["is_visited"].sum())
    unvisited_count = int((~df_bench["is_visited"]).sum())

    assert visited_count == 49, f"Expected 49 visited benchmark addresses, got {visited_count}"
    assert unvisited_count == 51, f"Expected 51 unvisited benchmark addresses, got {unvisited_count}"


def test_commercial_baseline_reproduction():
    """Verify commercial baseline reproduction: median ~ 376.4m, P90 ~ 839.2m, max ~ 4.8km."""
    report = evaluate_commercial_baseline(write_report=False)

    overall = report["overall"]
    median_error = overall["median_error_m"]
    p90_error = overall["p90_error_m"]
    max_error = overall["max_error_m"]

    # Assert within documented tolerance
    assert abs(median_error - 376.4) <= 0.5, f"Baseline median {median_error} deviates from 376.4m"
    assert abs(p90_error - 839.2) <= 2.0, f"Baseline P90 {p90_error} deviates from 839.2m"
    assert 4500.0 <= max_error <= 5000.0, f"Baseline max error {max_error} deviates from ~4.8km"
