"""Focused unit and regression tests for Production Calibrator.

Authority: Prompt 30 Phase E requirements.
Tests:
- resolved and unresolved landmark anchors
- ambiguous POI resolution
- mentioned-POI exclusion
- collocated-POI exclusion
- no eligible nearby POIs
- spatial-consistency rejection
- locality fallback
- normalized-residual calibration
- candidate-specific calibration lookup
- insufficient-sample fallback
- unknown candidate fallback
- minimum and maximum radius enforcement
- unchanged C2 behavior
- target closure and leakage protections
- stable output schema and provenance
"""

import math
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
import pytest

from neurogeo.calibrator import (
    assign_status_tier,
    calibrate_prediction,
    calibrate_prediction_production,
    compute_landmark_spatial_scale,
    find_nearest_unmentioned_poi,
    fit_empirical_calibration_quantiles,
    get_calibrated_radius_with_fallback,
)
from neurogeo.config import DEFAULT_CONFIG, PATHS, ModelConfig
from neurogeo.schemas import CalibrationResult, Phase4Prediction, PredictionStatus


@pytest.fixture
def mock_df_pois() -> pd.DataFrame:
    return pd.DataFrame([
        {"poi_id": "LM0001", "town_id": "T1", "name": "Main Temple", "landmark_type": "temple", "x": 100.0, "y": 100.0},
        {"poi_id": "LM0002", "town_id": "T1", "name": "Collocated Stall", "landmark_type": "shop", "x": 102.0, "y": 101.0},  # ~2.2m from LM0001 (< 5m)
        {"poi_id": "LM0003", "town_id": "T1", "name": "Local School", "landmark_type": "school", "x": 180.0, "y": 160.0},     # 100m from LM0001
        {"poi_id": "LM0004", "town_id": "T1", "name": "Town Hospital", "landmark_type": "hospital", "x": 600.0, "y": 700.0},  # ~781m from LM0001
        {"poi_id": "LM0005", "town_id": "T2", "name": "Wrong Town POI", "landmark_type": "bank", "x": 150.0, "y": 150.0},
    ])


@pytest.fixture
def mock_df_localities() -> pd.DataFrame:
    return pd.DataFrame([
        {"locality_id": "T1-L01", "town_id": "T1", "locality_name": "Locality 1", "centroid_x": 120.0, "centroid_y": 120.0},
        {"locality_id": "T1-L02", "town_id": "T1", "locality_name": "Locality 2", "centroid_x": 650.0, "centroid_y": 680.0},
    ])


@pytest.fixture
def mock_calibration_dict() -> Dict[str, Any]:
    return {
        "tiers": {
            "DOORSTEP_VERIFIED": {"radius_m": 35.0, "quantile": 0.90, "sample_count": 50, "fallback_level": "direct_doorstep"},
            "STREET_RESOLVED": {"radius_m": 500.0, "quantile": 0.90, "sample_count": 100, "fallback_level": "tier_direct"},
            "LOCALITY_BOUNDED": {"radius_m": 650.0, "quantile": 0.90, "sample_count": 80, "fallback_level": "tier_direct"},
            "BASELINE_FALLBACK": {"radius_m": 900.0, "quantile": 0.90, "sample_count": 60, "fallback_level": "tier_direct"},
        },
        "candidates": {
            "C1": {"candidate_type": "C1", "radius_m": 1650.0, "quantile": 0.90, "sample_count": 50, "fallback_level": "candidate_p90"},
            "C2": {"candidate_type": "C2", "radius_m": 35.0, "quantile": 0.90, "sample_count": 0, "fallback_level": "direct_doorstep"},
            "C3": {"candidate_type": "C3", "radius_m": 430.0, "quantile": 0.90, "sample_count": 80, "fallback_level": "candidate_p90"},
            "C4": {"candidate_type": "C4", "radius_m": 710.0, "quantile": 0.90, "sample_count": 60, "fallback_level": "candidate_p90"},
            "C5": {"candidate_type": "C5", "radius_m": 610.0, "quantile": 0.90, "sample_count": 70, "fallback_level": "candidate_p90"},
        },
        "landmark_geometry": {
            "target_quantile": 0.90,
            "sample_count": 90,
            "pooled_quantile_factor": 5.0,
            "candidate_quantile_factors": {"C3": 4.5, "C1": 5.0, "C4": 5.0},
            "candidate_sample_counts": {"C3": 80, "C1": 5, "C4": 5},
        },
        "localities": {
            "T1-L01": {"locality_id": "T1-L01", "radius_m": 480.0, "quantile": 0.90, "sample_count": 25},
            "T1-L02": {"locality_id": "T1-L02", "radius_m": 550.0, "quantile": 0.90, "sample_count": 20},
        },
    }


# -----------------------------------------------------------------------------
# 1. Resolved and Unresolved Landmark Anchors
# -----------------------------------------------------------------------------
def test_resolved_landmark_anchor_activates_geometry(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """Resolved landmark with nearby unmentioned POI activates landmark_geometry calibration."""
    entity = {
        "poi_resolution_status": "resolved",
        "landmark_poi_id": "LM0001",
        "locality_id_resolved": "T1-L01",
    }
    pred_coord = (110.0, 105.0)  # ~11.2m from LM0001 (consistent)

    res = calibrate_prediction_production(
        selected_candidate_type="C3",
        predicted_coord=pred_coord,
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=mock_calibration_dict,
        parsed_entity=entity,
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
    )

    assert isinstance(res, CalibrationResult)
    assert res.calibration_method == "landmark_geometry"
    assert res.primary_landmark_poi_id == "LM0001"
    assert res.nearest_unmentioned_poi_id == "LM0003"  # LM0002 excluded due to collocation (<5m)
    assert res.geometry_scale_m is not None
    assert res.geometry_scale_m >= 15.0
    assert res.fallback_level == "candidate_specific_geom"  # C3 has N=80 >= 15
    assert res.radius_m == min(max(round(4.5 * res.geometry_scale_m, 2), 15.0), 2500.0)


def test_unresolved_landmark_anchor_falls_back(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """Unresolved landmark bypasses Level 1 and falls back to candidate P90."""
    entity = {
        "poi_resolution_status": "unresolved",
        "landmark_poi_id": None,
        "locality_id_resolved": "T1-L01",
    }
    pred_coord = (110.0, 105.0)

    res = calibrate_prediction_production(
        selected_candidate_type="C4",
        predicted_coord=pred_coord,
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=mock_calibration_dict,
        parsed_entity=entity,
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
    )

    assert res.calibration_method == "candidate_p90"
    assert res.fallback_level == "candidate_p90"
    assert res.radius_m == 710.0
    assert "landmark_unresolved" in str(res.fallback_reason)


# -----------------------------------------------------------------------------
# 2. Ambiguous POI Resolution
# -----------------------------------------------------------------------------
def test_ambiguous_poi_resolution_falls_back(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """Ambiguous POI match cannot fabricate an anchor; falls back to candidate P90."""
    entity = {
        "poi_resolution_status": "ambiguous",
        "landmark_poi_id": "LM0001",  # resolver marked ambiguous
        "locality_id_resolved": "T1-L01",
    }

    res = calibrate_prediction_production(
        selected_candidate_type="C1",
        predicted_coord=(200.0, 200.0),
        status_tier=PredictionStatus.BASELINE_FALLBACK,
        calibration_dict=mock_calibration_dict,
        parsed_entity=entity,
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
    )

    assert res.calibration_method == "candidate_p90"
    assert res.radius_m == 1650.0
    assert "landmark_ambiguous" in str(res.fallback_reason)


# -----------------------------------------------------------------------------
# 3. Mentioned-POI and Collocated-POI Exclusion
# -----------------------------------------------------------------------------
def test_mentioned_and_collocated_poi_exclusion(mock_df_pois, mock_df_localities):
    """Primary anchor is excluded, and collocated POIs (<5.0m) are strictly excluded."""
    # Anchor LM0001 at (100, 100)
    # LM0002 is at (102, 101) -> distance sqrt(4+1) = 2.236m (< 5.0m)
    # LM0003 is at (180, 160) -> distance sqrt(6400+3600) = 100m
    d_min, unmentioned_id, pool_status = find_nearest_unmentioned_poi(
        primary_poi_id="LM0001",
        town_id="T1",
        locality_id="T1-L01",
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
        collocation_distance_threshold_m=5.0,
    )

    assert unmentioned_id != "LM0001", "Anchor itself must never be returned"
    assert unmentioned_id != "LM0002", "Collocated POI (<5m) must be excluded"
    assert unmentioned_id == "LM0003"
    assert abs(d_min - 100.0) < 0.1
    assert pool_status == "locality_pool"


# -----------------------------------------------------------------------------
# 4. No Eligible Nearby POIs
# -----------------------------------------------------------------------------
def test_no_eligible_nearby_pois():
    """When no other POIs exist in the town, marks geometry path ineligible."""
    solo_poi_df = pd.DataFrame([
        {"poi_id": "LM9999", "town_id": "T1", "name": "Solo Landmark", "landmark_type": "temple", "x": 50.0, "y": 50.0}
    ])
    d_min, unmentioned_id, pool_status = find_nearest_unmentioned_poi(
        primary_poi_id="LM9999",
        town_id="T1",
        locality_id="T1-L01",
        df_pois=solo_poi_df,
    )

    assert d_min is None
    assert unmentioned_id is None
    assert pool_status == "no_eligible_unmentioned_pois"


# -----------------------------------------------------------------------------
# 5. Spatial-Consistency Rejection
# -----------------------------------------------------------------------------
def test_spatial_consistency_rejection(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """If prediction is >1000m from resolved anchor, geometry path is rejected."""
    scale, status = compute_landmark_spatial_scale(
        predicted_coord=(1500.0, 100.0),  # 1400m from anchor (100, 100)
        anchor_coord=(100.0, 100.0),
        nearest_unmentioned_distance_m=100.0,
        max_consistency_distance_m=1000.0,
    )
    assert scale is None
    assert status == "prediction_anchor_spatial_inconsistency"

    # End-to-end calibrator rejection
    entity = {
        "poi_resolution_status": "resolved",
        "landmark_poi_id": "LM0001",
        "locality_id_resolved": "T1-L01",
    }
    res = calibrate_prediction_production(
        selected_candidate_type="C3",
        predicted_coord=(1500.0, 100.0),
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=mock_calibration_dict,
        parsed_entity=entity,
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
    )
    # Rejected geometry path falls back to candidate P90
    assert res.calibration_method == "candidate_p90"
    assert res.fallback_reason == "prediction_anchor_spatial_inconsistency"


# -----------------------------------------------------------------------------
# 6. Locality Fallback
# -----------------------------------------------------------------------------
def test_locality_fallback(mock_calibration_dict):
    """When landmark geometry and candidate P90 are unavailable, uses locality empirical P90."""
    cal_dict_no_cand = {
        "tiers": mock_calibration_dict["tiers"],
        "candidates": {},  # candidate P90 unavailable
        "localities": mock_calibration_dict["localities"],
    }
    entity = {
        "poi_resolution_status": "unresolved",
        "locality_id_resolved": "T1-L01",
    }

    res = calibrate_prediction_production(
        selected_candidate_type="C5",
        predicted_coord=(120.0, 120.0),
        status_tier=PredictionStatus.LOCALITY_BOUNDED,
        calibration_dict=cal_dict_no_cand,
        parsed_entity=entity,
    )

    assert res.calibration_method == "locality_empirical"
    assert res.fallback_level == "locality_specific"
    assert res.radius_m == 480.0
    assert res.sample_count == 25


# -----------------------------------------------------------------------------
# 7. Normalized Residual Calibration and Quantile Scaling
# -----------------------------------------------------------------------------
def test_normalized_residual_scaling(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """Verifies formula G_i = max(||P - L_0||, d_min, 15m) and radius = Q * G_i."""
    # Scale when ||P - L_0|| = 30m, d_min = 100m -> G_i = max(30, 100, 15) = 100m
    g_scale, status = compute_landmark_spatial_scale(
        predicted_coord=(130.0, 100.0),
        anchor_coord=(100.0, 100.0),
        nearest_unmentioned_distance_m=100.0,
    )
    assert g_scale == 100.0
    assert status == "scale_computed"

    # Scale when ||P - L_0|| = 200m, d_min = 100m -> G_i = max(200, 100, 15) = 200m
    g_scale_2, _ = compute_landmark_spatial_scale(
        predicted_coord=(300.0, 100.0),
        anchor_coord=(100.0, 100.0),
        nearest_unmentioned_distance_m=100.0,
    )
    assert g_scale_2 == 200.0


# -----------------------------------------------------------------------------
# 8. Candidate-Specific Calibration Lookup
# -----------------------------------------------------------------------------
def test_candidate_specific_lookup(mock_calibration_dict):
    """Verifies C1, C3, C4, C5 each retrieve their own candidate P90 radius."""
    for c_type, expected_r in [("C1", 1650.0), ("C3", 430.0), ("C4", 710.0), ("C5", 610.0)]:
        res = calibrate_prediction_production(
            selected_candidate_type=c_type,
            predicted_coord=(0.0, 0.0),
            status_tier=PredictionStatus.LOCALITY_BOUNDED,
            calibration_dict=mock_calibration_dict,
        )
        assert res.calibration_method == "candidate_p90"
        assert res.radius_m == expected_r


# -----------------------------------------------------------------------------
# 9. Insufficient-Sample and Unknown Candidate Fallback
# -----------------------------------------------------------------------------
def test_insufficient_sample_and_unknown_candidate_fallback(mock_calibration_dict):
    """Sparse or unknown candidates safely fall back to evidence tier."""
    sparse_cal = {
        "tiers": mock_calibration_dict["tiers"],
        "candidates": {
            "C1": {"candidate_type": "C1", "radius_m": 1200.0, "sample_count": 3},  # < 15 samples
        },
    }
    res = calibrate_prediction_production(
        selected_candidate_type="C1",
        predicted_coord=(0.0, 0.0),
        status_tier=PredictionStatus.BASELINE_FALLBACK,
        calibration_dict=sparse_cal,
    )
    assert res.calibration_method == "evidence_tier"
    assert res.radius_m == 900.0

    # Unknown candidate
    res_unk = calibrate_prediction_production(
        selected_candidate_type="C99",
        predicted_coord=(0.0, 0.0),
        status_tier=PredictionStatus.LOCALITY_BOUNDED,
        calibration_dict=mock_calibration_dict,
    )
    assert res_unk.calibration_method == "evidence_tier"
    assert res_unk.radius_m == 650.0


# -----------------------------------------------------------------------------
# 10. Minimum and Maximum Radius Enforcement
# -----------------------------------------------------------------------------
def test_min_and_max_radius_enforcement(mock_df_pois, mock_df_localities):
    """Enforces 15.0m floor and 2,500.0m cap AFTER calculation."""
    # Extremely small scale -> clamped to 15.0m
    cal_tiny = {
        "tiers": {"STREET_RESOLVED": {"radius_m": 5.0, "sample_count": 50, "fallback_level": "tier_direct"}},
        "candidates": {"C3": {"radius_m": 4.0, "sample_count": 50}},
    }
    res_tiny = calibrate_prediction_production(
        selected_candidate_type="C3",
        predicted_coord=(0.0, 0.0),
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=cal_tiny,
    )
    assert res_tiny.radius_m == 15.0, "Radius must not drop below 15.0m floor"

    # Extremely large scale -> clamped to 2500.0m
    cal_huge = {
        "tiers": {"BASELINE_FALLBACK": {"radius_m": 5000.0, "sample_count": 50, "fallback_level": "tier_direct"}},
        "candidates": {"C1": {"radius_m": 4500.0, "sample_count": 50}},
    }
    res_huge = calibrate_prediction_production(
        selected_candidate_type="C1",
        predicted_coord=(0.0, 0.0),
        status_tier=PredictionStatus.BASELINE_FALLBACK,
        calibration_dict=cal_huge,
    )
    assert res_huge.radius_m == 2500.0, "Radius must not exceed 2500.0m cap"


# -----------------------------------------------------------------------------
# 11. Unchanged C2 Direct-Doorstep Behavior
# -----------------------------------------------------------------------------
def test_unchanged_c2_behavior(mock_df_pois, mock_df_localities, mock_calibration_dict):
    """C2 direct-doorstep is strictly 35.0m and never overridden by geometry or coarser models."""
    entity = {
        "poi_resolution_status": "resolved",
        "landmark_poi_id": "LM0001",
        "locality_id_resolved": "T1-L01",
    }
    # Even if landmark is resolved, C2 is strictly direct doorstep!
    res_c2 = calibrate_prediction_production(
        selected_candidate_type="C2",
        predicted_coord=(105.0, 105.0),
        status_tier=PredictionStatus.DOORSTEP_VERIFIED,
        calibration_dict=mock_calibration_dict,
        parsed_entity=entity,
        df_pois=mock_df_pois,
        df_localities=mock_df_localities,
    )
    assert res_c2.radius_m == 35.0
    assert res_c2.calibration_method == "direct_doorstep"
    assert res_c2.calibration_tier == PredictionStatus.DOORSTEP_VERIFIED.value
    assert res_c2.fallback_level == "direct_doorstep"

    # has_credible_doorstep=True also preserves 35.0m
    res_cred = calibrate_prediction_production(
        selected_candidate_type="C4",
        predicted_coord=(105.0, 105.0),
        status_tier=PredictionStatus.DOORSTEP_VERIFIED,
        calibration_dict=mock_calibration_dict,
        has_credible_doorstep=True,
    )
    assert res_cred.radius_m == 35.0
    assert res_cred.calibration_method == "direct_doorstep"


# -----------------------------------------------------------------------------
# 12. Backward Compatibility of calibrate_prediction Wrapper
# -----------------------------------------------------------------------------
def test_calibrate_prediction_backward_compatibility(mock_calibration_dict):
    """Legacy call signature calibrate_prediction(status_tier, calibration_dict) works identically."""
    rad, conf, fb, cnt = calibrate_prediction(
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=mock_calibration_dict,
    )
    assert rad == 500.0
    assert conf == 0.90
    assert fb == "tier_direct"
    assert cnt == 100

    # Modern call returning CalibrationResult
    res = calibrate_prediction(
        status_tier=PredictionStatus.STREET_RESOLVED,
        calibration_dict=mock_calibration_dict,
        selected_candidate_type="C3",
        return_result_object=True,
    )
    assert isinstance(res, CalibrationResult)
    assert res.radius_m == 430.0
    assert res.calibration_method == "candidate_p90"
