"""Tests for GPS Trajectory Mining, Tortuosity Guards, and Doorstep Medoid Estimation.

Authority: PS3_MODEL_SPEC.md Section 4, PS3_DATA_CONTRACT.md.
"""

import numpy as np
import pandas as pd
import pytest

from neurogeo.trajectory_miner import (
    compute_path_tortuosity,
    compute_radius_of_gyration,
    filter_gps_jumps,
    compute_spatial_medoid,
    compute_address_doorstep_medoid,
)
from neurogeo.sanitizer import sanitize_visits


def test_straight_line_tortuosity():
    """Verify that a straight line path yields tortuosity exactly 1.0."""
    x = [0.0, 10.0, 20.0, 30.0, 40.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.0]
    tau = compute_path_tortuosity(x, y)
    assert abs(tau - 1.0) < 1e-4, f"Straight line path should have tortuosity 1.0, got {tau}"


def test_tortuosity_stationary_jitter_guard():
    """Verify that stationary jitter (net displacement <= 15m) returns neutral tau = 1.0."""
    # Person pacing around a 5m room, total travel = 40m, but net displacement = 2m
    x = [0.0, 5.0, 0.0, 5.0, 2.0]
    y = [0.0, 0.0, 0.0, 0.0, 1.0]
    # Net displacement is sqrt(2^2 + 1^2) = 2.23m <= 15.0m
    tau = compute_path_tortuosity(x, y, min_displacement_m=15.0)
    assert tau == 1.0, f"Stationary jitter should be guarded to 1.0, got {tau}"


def test_tortuosity_too_few_points_guard():
    """Verify that fewer than 3 GPS points returns neutral tau = 1.0."""
    # 2 points
    tau_2 = compute_path_tortuosity([0.0, 50.0], [0.0, 50.0], min_points=3)
    assert tau_2 == 1.0

    # 1 point
    tau_1 = compute_path_tortuosity([0.0], [0.0], min_points=3)
    assert tau_1 == 1.0


def test_tortuosity_poor_accuracy_guard():
    """Verify that high GPS error (> 50m) returns neutral tau = 1.0."""
    x = [0.0, 20.0, 50.0, 100.0]
    y = [0.0, 10.0, 30.0, 80.0]
    acc = [65.0, 70.0, 80.0, 75.0]  # mean > 50m
    tau = compute_path_tortuosity(x, y, accuracy_m=acc, max_accuracy_m=50.0)
    assert tau == 1.0


def test_gps_jump_filter():
    """Verify that instantaneous multi-path velocity jumps (> 40 m/s) are filtered out."""
    # 3 points recorded every 30 seconds
    # Point 1: (0, 0) at t=0
    # Point 2: (5000, 5000) at t=30s -> speed = 7071m / 30s = 235 m/s (anomalous jump)
    # Point 3: (30, 40) at t=60s -> speed from Point 1 = 50m / 60s = 0.83 m/s (normal)
    x = np.array([0.0, 5000.0, 30.0])
    y = np.array([0.0, 5000.0, 40.0])
    ts = pd.to_datetime(["2026-04-01 10:00:00", "2026-04-01 10:00:30", "2026-04-01 10:01:00"]).to_numpy()

    fx, fy, _, num_jumps = filter_gps_jumps(x, y, ts, max_speed_mps=40.0)

    assert num_jumps == 1
    assert len(fx) == 2
    assert fx[0] == 0.0 and fx[1] == 30.0
    assert fy[0] == 0.0 and fy[1] == 40.0


def test_radius_of_gyration_non_negative():
    """Verify that radius of gyration is non-negative and deterministic."""
    x = [10.0, 20.0, 30.0, 40.0]
    y = [15.0, 25.0, 35.0, 45.0]
    rg = compute_radius_of_gyration(x, y)
    assert rg > 0.0
    assert abs(rg - compute_radius_of_gyration(x, y)) < 1e-6

    # Single point must be 0
    assert compute_radius_of_gyration([10.0], [10.0]) == 0.0


def test_spatial_medoid_determinism():
    """Verify that spatial medoid selection is deterministic and chooses a true cluster member."""
    x = np.array([10.0, 11.0, 12.0, 100.0])
    y = np.array([10.0, 11.0, 12.0, 100.0])
    med_x, med_y = compute_spatial_medoid(x, y)

    # Medoid among candidate points should be inside the cluster (11, 11), not at (100, 100)
    assert med_x in [10.0, 11.0, 12.0]
    assert med_y in [10.0, 11.0, 12.0]
    assert (med_x, med_y) == compute_spatial_medoid(x, y)


def test_doorstep_medoid_outlier_rejection():
    """Verify that a fraudulent/low-integrity outlier does NOT dominate the doorstep medoid."""
    # Address A1 has:
    # 2 clean positive doorstep visits near (100, 100) with unique photos (I ~ 0.9)
    # 1 fake visit at (5000, 5000) with photo collision count = 26 (I < 1e-10)
    df_visits = pd.DataFrame([
        {
            "visit_id": "V1",
            "address_id": "A1",
            "outcome": "met_family",
            "photo_hash": "hash_unique_1",
            "dwell_s": 120,
            "gps_accuracy_m": 12.0,
            "checkin_x": 100.0,
            "checkin_y": 100.0,
        },
        {
            "visit_id": "V2",
            "address_id": "A1",
            "outcome": "locked_premises",
            "photo_hash": "hash_unique_2",
            "dwell_s": 90,
            "gps_accuracy_m": 15.0,
            "checkin_x": 104.0,
            "checkin_y": 102.0,
        },
        {
            "visit_id": "V_FAKE",
            "address_id": "A1",
            "outcome": "met_borrower",
            "photo_hash": "ph_duplicate_cluster",  # will have 26 collisions below
            "dwell_s": 15,
            "gps_accuracy_m": 80.0,
            "checkin_x": 5000.0,
            "checkin_y": 5000.0,
        },
    ] + [
        # Add 25 more visits reusing 'ph_duplicate_cluster' to trigger fraud collision penalty
        {
            "visit_id": f"V_DUMMY_{i}",
            "address_id": f"A_OTHER_{i}",
            "outcome": "address_not_traceable",
            "photo_hash": "ph_duplicate_cluster",
            "dwell_s": 10,
            "gps_accuracy_m": 30.0,
            "checkin_x": 0.0,
            "checkin_y": 0.0,
        }
        for i in range(25)
    ])

    sanitized = sanitize_visits(df_visits)
    medoid_info = compute_address_doorstep_medoid("A1", sanitized, min_integrity=0.50)

    assert medoid_info is not None
    assert medoid_info["has_credible_doorstep"] is True
    # The medoid must be near (100, 100), NOT near (5000, 5000)
    assert abs(medoid_info["doorstep_medoid_x"] - 100.0) < 10.0
    assert abs(medoid_info["doorstep_medoid_y"] - 100.0) < 10.0
    assert "V_FAKE" not in medoid_info["contributing_visit_ids"]
    assert set(medoid_info["contributing_visit_ids"]) == {"V1", "V2"}
