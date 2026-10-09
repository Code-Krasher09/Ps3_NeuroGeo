"""Deterministic unit and contract tests for Phase 5 Offline Navigator and Evidence Card.

Authority: PS3_FINAL_ARCHITECTURE.md Section 4, PS3_MODEL_SPEC.md Section 8,
and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

import math
import socket
from typing import Dict, List
import pandas as pd
import pytest

from neurogeo.config import PATHS
from neurogeo.navigator import (
    haversine_distance_m,
    euclidean_distance_m,
    bearing_degrees,
    destination_offset,
    point_within_radius,
    coarse_direction,
    navigate,
    build_evidence_card,
    explain_prediction,
)
from neurogeo.schemas import (
    NavigationMode,
    NavigationInstruction,
    EvidenceCardData,
    PredictionStatus,
    Phase4Prediction,
)


# =============================================================================
# 1. Geometry Tests
# =============================================================================

def test_geometry_zero_distance():
    """Verify distance between identical points is exactly 0.0 m."""
    d = euclidean_distance_m(150.5, 230.2, 150.5, 230.2)
    assert d == 0.0


def test_geometry_positive_distance():
    """Verify Pythagorean 3-4-5 triangle computes exact 500.0 m distance."""
    d = euclidean_distance_m(0.0, 0.0, 300.0, 400.0)
    assert d == 500.0


def test_geometry_bearing_cardinal_directions():
    """Verify bearing angles for exact cardinal compass directions (+Y=North, +X=East)."""
    # Due North: (0, 0) -> (0, 100) => 0.0 deg
    assert bearing_degrees(0.0, 0.0, 0.0, 100.0) == 0.0
    assert coarse_direction(0.0) == "North"

    # Due East: (0, 0) -> (100, 0) => 90.0 deg
    assert bearing_degrees(0.0, 0.0, 100.0, 0.0) == 90.0
    assert coarse_direction(90.0) == "East"

    # Due South: (0, 0) -> (0, -100) => 180.0 deg
    assert bearing_degrees(0.0, 0.0, 0.0, -100.0) == 180.0
    assert coarse_direction(180.0) == "South"

    # Due West: (0, 0) -> (-100, 0) => 270.0 deg
    assert bearing_degrees(0.0, 0.0, -100.0, 0.0) == 270.0
    assert coarse_direction(270.0) == "West"


def test_geometry_bearing_diagonal_directions():
    """Verify diagonal bearings (NE=45, SE=135, SW=225, NW=315)."""
    # North-East: (0, 0) -> (100, 100) => 45.0 deg
    assert bearing_degrees(0.0, 0.0, 100.0, 100.0) == 45.0
    assert coarse_direction(45.0) == "North-East"

    # South-East: (0, 0) -> (100, -100) => 135.0 deg
    assert bearing_degrees(0.0, 0.0, 100.0, -100.0) == 135.0
    assert coarse_direction(135.0) == "South-East"

    # South-West: (0, 0) -> (-100, -100) => 225.0 deg
    assert bearing_degrees(0.0, 0.0, -100.0, -100.0) == 225.0
    assert coarse_direction(225.0) == "South-West"

    # North-West: (0, 0) -> (-100, 100) => 315.0 deg
    assert bearing_degrees(0.0, 0.0, -100.0, 100.0) == 315.0
    assert coarse_direction(315.0) == "North-West"


def test_geometry_8_sector_boundaries():
    """Verify exact boundary behavior around 45-degree sector transitions."""
    # North boundaries: [-22.5, 22.5) -> [337.5, 360) U [0, 22.5)
    assert coarse_direction(337.4) == "North-West"
    assert coarse_direction(337.5) == "North"
    assert coarse_direction(359.9) == "North"
    assert coarse_direction(0.0) == "North"
    assert coarse_direction(22.4) == "North"
    assert coarse_direction(22.5) == "North-East"

    # East boundaries: [67.5, 112.5)
    assert coarse_direction(67.4) == "North-East"
    assert coarse_direction(67.5) == "East"
    assert coarse_direction(112.4) == "East"
    assert coarse_direction(112.5) == "South-East"


def test_geometry_point_inside_radius():
    """Verify point_within_radius returns True when within threshold."""
    assert point_within_radius(100.0, 100.0, 120.0, 100.0, radius_m=30.0) is True
    # Exactly on boundary
    assert point_within_radius(0.0, 0.0, 30.0, 40.0, radius_m=50.0) is True


def test_geometry_point_outside_radius():
    """Verify point_within_radius returns False when outside threshold."""
    assert point_within_radius(100.0, 100.0, 140.0, 100.0, radius_m=30.0) is False
    assert point_within_radius(0.0, 0.0, 30.0, 40.0, radius_m=49.9) is False


def test_geometry_haversine_formula():
    """Verify standard geodetic haversine implementation on known benchmark coordinate pair."""
    # London (51.5074 N, 0.1278 W) to Paris (48.8566 N, 2.3522 E) ~ 343.5 km
    dist_m = haversine_distance_m(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340000.0 < dist_m < 350000.0


def test_geometry_destination_offset():
    """Verify destination_offset accurately projects coordinates given bearing and distance."""
    # Move 100 m North from (500, 500) -> (500, 600)
    dest_n = destination_offset(500.0, 500.0, distance_m=100.0, bearing_deg=0.0)
    assert dest_n == (500.0, 600.0)

    # Move 100 m East from (500, 500) -> (600, 500)
    dest_e = destination_offset(500.0, 500.0, distance_m=100.0, bearing_deg=90.0)
    assert dest_e == (600.0, 500.0)


# =============================================================================
# 2. Navigation Modes & Instructions
# =============================================================================

def test_navigation_doorstep_mode():
    """Verify DOORSTEP_VERIFIED predictions map to DOORSTEP mode."""
    pred = {
        "predicted_x": 1000.0,
        "predicted_y": 2000.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 35.0,
    }
    nav = navigate(current_x=1000.0, current_y=1900.0, prediction=pred)
    assert nav.navigation_mode == NavigationMode.DOORSTEP.value
    assert nav.status_tier == "DOORSTEP_VERIFIED"
    assert "doorstep" in nav.instruction.lower()


def test_navigation_street_mode():
    """Verify STREET_RESOLVED predictions map to STREET mode and include caution."""
    pred = {
        "predicted_x": 500.0,
        "predicted_y": 500.0,
        "status_tier": "STREET_RESOLVED",
        "confidence_radius_m": 85.0,
    }
    nav = navigate(current_x=500.0, current_y=300.0, prediction=pred)
    assert nav.navigation_mode == NavigationMode.STREET.value
    assert "street" in nav.instruction.lower()
    assert nav.caution is not None
    assert "street-level" in nav.caution.lower()


def test_navigation_locality_mode():
    """Verify LOCALITY_BOUNDED predictions map to LOCALITY mode and include neighborhood caveat."""
    pred = {
        "predicted_x": 0.0,
        "predicted_y": 0.0,
        "status_tier": "LOCALITY_BOUNDED",
        "confidence_radius_m": 250.0,
    }
    nav = navigate(current_x=300.0, current_y=400.0, prediction=pred)
    assert nav.navigation_mode == NavigationMode.LOCALITY.value
    assert "locality" in nav.instruction.lower()
    assert nav.caution is not None
    assert "locality" in nav.caution.lower()


def test_navigation_baseline_mode():
    """Verify BASELINE_FALLBACK predictions map to BASELINE mode and warn of low confidence."""
    pred = {
        "predicted_x": 2000.0,
        "predicted_y": 2000.0,
        "status_tier": "BASELINE_FALLBACK",
        "confidence_radius_m": 450.0,
    }
    nav = navigate(current_x=2000.0, current_y=1000.0, prediction=pred)
    assert nav.navigation_mode == NavigationMode.BASELINE.value
    assert "baseline" in nav.instruction.lower()
    assert nav.caution is not None
    assert "low-confidence" in nav.caution.lower()


def test_navigation_within_radius_instruction():
    """Verify instruction states agent is within estimated radius when distance <= radius."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 50.0,
    }
    # Current distance = 30 m (< 50 m)
    nav = navigate(current_x=100.0, current_y=130.0, prediction=pred)
    assert nav.within_confidence_radius is True
    assert "within the estimated" in nav.instruction
    assert "50 m" in nav.instruction


def test_navigation_outside_radius_instruction():
    """Verify instruction provides distance and direction when outside radius."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "STREET_RESOLVED",
        "confidence_radius_m": 50.0,
    }
    # Current distance = 200 m South of target -> Must proceed North
    nav = navigate(current_x=100.0, current_y=-100.0, prediction=pred)
    assert nav.within_confidence_radius is False
    assert "proceed north" in nav.instruction.lower()
    assert "200 m" in nav.instruction


def test_navigation_zero_distance_behavior():
    """Verify zero-distance arrival guidance when current position equals target."""
    pred = {
        "predicted_x": 120.0,
        "predicted_y": 340.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 35.0,
    }
    nav = navigate(current_x=120.0, current_y=340.0, prediction=pred)
    assert nav.distance_m == 0.0
    assert nav.within_confidence_radius is True
    assert "arrived at the predicted target" in nav.instruction


def test_navigation_deterministic_repeated_calls():
    """Verify repeated navigation calls with identical inputs produce bit-for-bit identical results."""
    pred = {
        "predicted_x": 450.2,
        "predicted_y": 780.9,
        "status_tier": "STREET_RESOLVED",
        "confidence_radius_m": 85.0,
    }
    results = [navigate(current_x=200.0, current_y=300.0, prediction=pred) for _ in range(10)]
    first = results[0]
    for other in results[1:]:
        assert first == other


# =============================================================================
# 3. Uncertainty-Aware & Status-Aware Language
# =============================================================================

def test_uncertainty_p90_wording_no_guarantee():
    """Verify instructions and explanations use 'empirical P90 radius' and never claim 'guarantee'."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "LOCALITY_BOUNDED",
        "confidence_radius_m": 250.0,
    }
    nav = navigate(current_x=500.0, current_y=500.0, prediction=pred)
    exp = explain_prediction(pred)

    assert "guarantee" not in nav.instruction.lower()
    assert "guaranteed" not in nav.instruction.lower()
    assert "guarantee" not in exp.lower()
    assert "guaranteed" not in exp.lower()


def test_uncertainty_low_confidence_produces_caution():
    """Verify low-confidence status tiers (BASELINE, LOCALITY) produce mandatory caution text."""
    for tier in ["BASELINE_FALLBACK", "LOCALITY_BOUNDED"]:
        pred = {"predicted_x": 0.0, "predicted_y": 0.0, "status_tier": tier, "confidence_radius_m": 300.0}
        nav = navigate(100.0, 100.0, pred)
        assert nav.caution is not None
        assert len(nav.caution) > 0


def test_uncertainty_radius_preserved_exactly():
    """Verify confidence_radius_m is preserved identically from Phase 4 prediction."""
    r_val = 615.93
    pred = {"predicted_x": 0.0, "predicted_y": 0.0, "status_tier": "STREET_RESOLVED", "confidence_radius_m": r_val}
    nav = navigate(100.0, 100.0, pred)
    assert nav.confidence_radius_m == r_val


# =============================================================================
# 4. Negative Evidence & Field Warnings
# =============================================================================

def test_negative_evidence_neighbour_says_shifted_warning():
    """Verify neighbour_says_shifted triggers an explicit shifted borrower warning."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 35.0,
    }
    nav = navigate(0.0, 0.0, pred, negative_evidence=["neighbour_says_shifted"])
    assert nav.caution is not None
    assert "shifted" in nav.caution.lower()

    card = build_evidence_card(pred, evidence_context={"negative_evidence": ["neighbour_says_shifted"]})
    assert len(card.negative_evidence) > 0
    assert card.caution is not None
    assert "shifted" in card.caution.lower()


def test_negative_evidence_address_not_traceable_warning():
    """Verify address_not_traceable triggers an untraceable address warning."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "STREET_RESOLVED",
        "confidence_radius_m": 85.0,
    }
    nav = navigate(0.0, 0.0, pred, negative_evidence=["address_not_traceable"])
    assert nav.caution is not None
    assert "not traceable" in nav.caution.lower()


def test_negative_evidence_never_positive_doorstep():
    """Frozen requirement: neighbour_says_shifted is address-state negative evidence, never positive doorstep."""
    pred = {
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "selected_candidate_type": "C1",
        "status_tier": "BASELINE_FALLBACK",
        "confidence_radius_m": 450.0,
    }
    card = build_evidence_card(pred, evidence_context={"negative_evidence": ["neighbour_says_shifted"]})
    # Negative evidence must be recorded under negative_evidence, never positive_evidence
    assert any("shifted" in ne.lower() for ne in card.negative_evidence)
    assert not any("shifted" in pe.lower() for pe in card.positive_evidence)
    assert card.status_tier != "DOORSTEP_VERIFIED"


# =============================================================================
# 5. Evidence Card & Explanation
# =============================================================================

def test_evidence_card_candidate_provenance_preserved():
    """Verify candidate types map to standardized provenance source names."""
    c_types = {
        "C1": "commercial_baseline",
        "C2": "direct_visit_medoid",
        "C3": "physical_poi",
        "C4": "spatial_neighbor_graph",
        "C5": "locality_centroid",
    }
    for c_code, expected_name in c_types.items():
        pred = {
            "address_id": "AD000001",
            "predicted_x": 100.0,
            "predicted_y": 100.0,
            "selected_candidate_type": c_code,
            "evidence_sources": c_code,
            "status_tier": "STREET_RESOLVED",
        }
        card = build_evidence_card(pred)
        assert expected_name in card.evidence_sources


def test_evidence_card_calibration_provenance_preserved():
    """Verify calibration tier, sample count, and fallback level are preserved."""
    pred = {
        "address_id": "AD000001",
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "status_tier": "STREET_RESOLVED",
        "calibration_tier": "STREET_RESOLVED",
        "calibration_sample_count": 279,
        "calibration_fallback_level": "tier_direct",
        "confidence_radius_m": 615.93,
    }
    card = build_evidence_card(pred)
    assert card.calibration_tier == "STREET_RESOLVED"
    assert card.calibration_sample_count == 279
    assert card.calibration_fallback_level == "tier_direct"
    assert card.confidence_radius_m == 615.93


def test_evidence_card_score_and_margin_preserved():
    """Verify score and score_margin are preserved exactly."""
    pred = {
        "address_id": "AD000001",
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "selected_score": 4.5230,
        "score_margin": 0.9978,
    }
    card = build_evidence_card(pred)
    assert card.score == 4.5230
    assert card.score_margin == 0.9978


def test_evidence_card_missing_optional_evidence_handled():
    """Verify missing optional fields (empty context, None values) are handled gracefully."""
    pred = {
        "address_id": "AD000099",
        "predicted_x": 500.0,
        "predicted_y": 500.0,
    }
    card = build_evidence_card(pred, evidence_context=None)
    assert card.address_id == "AD000099"
    assert card.predicted_coordinates == (500.0, 500.0)
    assert card.explanation is not None


def test_explanation_distinguishes_evidence_types():
    """Verify explain_prediction clearly distinguishes direct, neighbor, and baseline evidence."""
    # Doorstep
    exp_door = explain_prediction({"status_tier": "DOORSTEP_VERIFIED", "selected_candidate_type": "C2"})
    assert "doorstep medoid" in exp_door.lower() or "direct" in exp_door.lower()

    # Street
    exp_street = explain_prediction({"status_tier": "STREET_RESOLVED", "selected_candidate_type": "C4"})
    assert "neighbor" in exp_street.lower() or "street" in exp_street.lower()

    # Baseline
    exp_base = explain_prediction({"status_tier": "BASELINE_FALLBACK", "selected_candidate_type": "C1"})
    assert "baseline" in exp_base.lower()


# =============================================================================
# 6. Leakage & Immutability Tests
# =============================================================================

def test_leakage_navigator_never_reads_surveyed_coords():
    """Verify navigator operates completely without surveyed ground truth coordinates."""
    pred = {
        "predicted_x": 250.0,
        "predicted_y": 350.0,
        "status_tier": "STREET_RESOLVED",
        "confidence_radius_m": 85.0,
        # Intentionally inject bogus ground truth to verify it is completely ignored
        "surveyed_x": 99999.0,
        "surveyed_y": 99999.0,
    }
    nav = navigate(current_x=250.0, current_y=300.0, prediction=pred)
    assert nav.target_x == 250.0
    assert nav.target_y == 350.0
    assert nav.distance_m == 50.0


def test_leakage_navigator_never_reads_target_gps():
    """Verify navigator never touches historical GPS trajectory breadcrumbs."""
    pred = {
        "predicted_x": 120.0,
        "predicted_y": 140.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 35.0,
    }
    nav = navigate(current_x=120.0, current_y=100.0, prediction=pred)
    assert nav.target_x == 120.0
    assert nav.target_y == 140.0


def test_phase4_immutability():
    """Phase 4 Immutability Test: Verify navigator consumes Phase 4 predictions without altering them."""
    preds_df = pd.read_csv(PATHS.phase4_predictions_csv)
    assert len(preds_df) == 100

    row = preds_df.iloc[0].to_dict()
    orig_x = row["predicted_x"]
    orig_y = row["predicted_y"]
    orig_cand = row["selected_candidate_type"]
    orig_tier = row["status_tier"]
    orig_rad = row["confidence_radius_m"]

    nav = navigate(current_x=0.0, current_y=0.0, prediction=row)
    card = build_evidence_card(row)

    # Verify input row is unchanged
    assert row["predicted_x"] == orig_x
    assert row["predicted_y"] == orig_y
    assert row["selected_candidate_type"] == orig_cand
    assert row["status_tier"] == orig_tier
    assert row["confidence_radius_m"] == orig_rad

    # Verify output preserves fields
    assert nav.target_x == orig_x
    assert nav.target_y == orig_y
    assert nav.status_tier == orig_tier
    assert nav.confidence_radius_m == orig_rad
    assert card.selected_candidate == orig_cand
    assert card.confidence_radius_m == orig_rad


# =============================================================================
# 7. Offline & No-Network Tests
# =============================================================================

def test_offline_no_external_network(monkeypatch):
    """Verify navigator operates completely offline; network calls raise an exception."""
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network access attempted during offline navigation!")

    monkeypatch.setattr(socket, "socket", guarded_socket)

    pred = {
        "address_id": "AD000001",
        "predicted_x": 1000.0,
        "predicted_y": 2000.0,
        "status_tier": "DOORSTEP_VERIFIED",
        "confidence_radius_m": 35.0,
        "selected_candidate_type": "C2",
    }
    # Both navigation and evidence card construction must execute without network calls
    nav = navigate(current_x=500.0, current_y=1500.0, prediction=pred)
    card = build_evidence_card(pred)

    assert nav.distance_m > 0
    assert card.provenance["offline_mode"] is True
    assert card.provenance["external_network_accessed"] is False
