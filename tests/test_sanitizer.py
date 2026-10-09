"""Tests for Operational Visit Sanitizer & Integrity Scoring.

Authority: PS3_MODEL_SPEC.md Sections 2 and 3, PS3_DATA_CONTRACT.md.
"""

import numpy as np
import pandas as pd
import pytest

from neurogeo.sanitizer import (
    compute_visit_integrity,
    compute_spatial_evidence_strength,
    compute_photo_collision_counts,
    sanitize_visits,
    OUTCOME_WEIGHTS,
)


def test_integrity_score_strictly_bounded():
    """Verify that I(v) is strictly bounded in [0.0, 1.0] across edge-case inputs."""
    # Test across extreme cases: negative dwell, huge speed, zero collision, massive collision
    collision_counts = [0, 1, 2, 5, 26, 100, 1000]
    dwell_times = [-100.0, 0.0, 10.0, 45.0, 60.0, 600.0, 10000.0]
    speeds = [-50.0, 0.0, 10.0, 30.0, 60.0, 120.0, 500.0]

    for c in collision_counts:
        for d in dwell_times:
            for s in speeds:
                score = compute_visit_integrity(c, d, s)
                assert 0.0 <= score <= 1.0, f"Integrity out of bounds for c={c}, d={d}, s={s}: {score}"


def test_clean_visit_vs_degraded_visit_integrity():
    """Verify that a clean visit has significantly higher integrity than a degraded visit."""
    # Clean visit: unique photo (c=1), 2 mins dwell (120s), normal bike speed (15 km/h)
    clean_score = compute_visit_integrity(collision_counts=1, dwell_s=120.0, speed_kmh=15.0)

    # Degraded visit: 5 photo collisions, 5s drive-by dwell, high speed (80 km/h)
    degraded_score = compute_visit_integrity(collision_counts=5, dwell_s=5.0, speed_kmh=80.0)

    assert clean_score > 0.85, f"Expected clean score > 0.85, got {clean_score}"
    assert degraded_score < 0.05, f"Expected degraded score < 0.05, got {degraded_score}"
    assert clean_score > (10 * degraded_score), "Clean visit should dominate degraded visit"


def test_photo_collision_penalty_scale():
    """Verify that duplicate photo hashes incur the exact exponential penalty."""
    # For unique photo (c=1): photo factor is exp(0) = 1.0
    # Dwell fixed at high value (300s) and speed=0 to isolate photo decay
    score_1 = compute_visit_integrity(collision_counts=1, dwell_s=300.0, speed_kmh=0.0, lambda_photo=1.0)
    score_2 = compute_visit_integrity(collision_counts=2, dwell_s=300.0, speed_kmh=0.0, lambda_photo=1.0)
    score_5 = compute_visit_integrity(collision_counts=5, dwell_s=300.0, speed_kmh=0.0, lambda_photo=1.0)
    score_26 = compute_visit_integrity(collision_counts=26, dwell_s=300.0, speed_kmh=0.0, lambda_photo=1.0)

    assert abs(score_1 - 1.0) < 0.01
    assert abs(score_2 - np.exp(-1.0)) < 0.01  # exp(-1) ~= 0.368
    assert abs(score_5 - np.exp(-4.0)) < 0.01  # exp(-4) ~= 0.0183
    assert score_26 < 1e-10, f"FA009-style 26-collision should be < 1e-10, got {score_26}"


def test_missing_photo_hash_handling():
    """Verify that missing/null photo hashes are not lumped into a shared collision group."""
    df_sample = pd.DataFrame([
        {"visit_id": "V1", "photo_hash": None},
        {"visit_id": "V2", "photo_hash": np.nan},
        {"visit_id": "V3", "photo_hash": ""},
        {"visit_id": "V4", "photo_hash": "hash_shared"},
        {"visit_id": "V5", "photo_hash": "hash_shared"},
    ])

    counts = compute_photo_collision_counts(df_sample)

    # Missing hashes must receive count = 1 (no collision penalty)
    assert counts.iloc[0] == 1, "None photo hash should have count 1"
    assert counts.iloc[1] == 1, "NaN photo hash should have count 1"
    assert counts.iloc[2] == 1, "Empty string photo hash should have count 1"

    # Shared hashes must receive count = 2
    assert counts.iloc[3] == 2
    assert counts.iloc[4] == 2


def test_evidence_strength_decay_with_gps_accuracy():
    """Verify that S(v) decreases monotonically with poorer GPS accuracy."""
    accuracies = [5.0, 15.0, 25.0, 50.0, 80.0]
    strengths = [compute_spatial_evidence_strength(acc, "met_family", sigma_gps=25.0) for acc in accuracies]

    for i in range(len(strengths) - 1):
        assert strengths[i] > strengths[i + 1], f"S(v) failed monotonic decrease: {strengths}"
    assert strengths[0] > 0.80
    assert strengths[-1] < 0.05


def test_outcome_weights_frozen_specification():
    """Verify that outcome weights strictly follow PS3_MODEL_SPEC.md."""
    acc = 10.0  # constant accuracy
    s_family = compute_spatial_evidence_strength(acc, "met_family")
    s_locked = compute_spatial_evidence_strength(acc, "locked_premises")
    s_cash = compute_spatial_evidence_strength(acc, "cash_collected")
    s_borrower = compute_spatial_evidence_strength(acc, "met_borrower")
    s_shifted = compute_spatial_evidence_strength(acc, "neighbour_says_shifted")
    s_nobody = compute_spatial_evidence_strength(acc, "no_such_person")
    s_untraceable = compute_spatial_evidence_strength(acc, "address_not_traceable")

    assert s_family > s_locked > s_cash > s_borrower > s_shifted
    assert s_nobody == 0.0
    assert s_untraceable == 0.0


def test_neighbour_says_shifted_is_address_state_only():
    """Verify that neighbour_says_shifted is flagged as address-state, NOT positive doorstep evidence."""
    df_sample = pd.DataFrame([
        {"visit_id": "V1", "address_id": "A1", "outcome": "met_family", "photo_hash": "h1", "dwell_s": 100, "gps_accuracy_m": 10},
        {"visit_id": "V2", "address_id": "A1", "outcome": "neighbour_says_shifted", "photo_hash": "h2", "dwell_s": 100, "gps_accuracy_m": 10},
    ])

    sanitized = sanitize_visits(df_sample)

    v1 = sanitized.iloc[0]
    v2 = sanitized.iloc[1]

    assert bool(v1["is_doorstep_confirmed"]) is True
    assert bool(v1["is_address_state_shifted"]) is False

    assert bool(v2["is_doorstep_confirmed"]) is False, "neighbour_says_shifted must NOT be doorstep confirmed"
    assert bool(v2["is_address_state_shifted"]) is True, "neighbour_says_shifted must be address-state evidence"
    assert bool(v2["is_credible_doorstep"]) is False
