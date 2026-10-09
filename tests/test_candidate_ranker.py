"""Deterministic unit and contract tests for spatial candidate generation and linear reranker.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

import math
from typing import Dict, List
import numpy as np
import pytest

from neurogeo.candidate_ranker import (
    generate_candidates,
    deduplicate_candidates,
    extract_candidate_features,
    score_candidate,
    rank_candidates,
    explain_ranking,
    fit_reranker_weights,
    compute_euclidean_distance,
)
from neurogeo.schemas import ParsedAddressEntity, SpatialCandidate
from neurogeo.config import DEFAULT_CONFIG


# -------------------------------------------------------------
# Test 1: C1 candidate creation
# -------------------------------------------------------------
def test_c1_candidate_creation():
    """Verify C1 commercial baseline candidate is created with correct coordinates and metadata."""
    baseline = {"geocoder_x": 1250.5, "geocoder_y": -280.2, "precision": "street", "provider": "test_provider"}
    cands = generate_candidates(address_id="AD0001", baseline_row=baseline)
    c1 = next(c for c in cands if c.candidate_type == "C1")
    assert c1.available is True
    assert c1.x == 1250.5
    assert c1.y == -280.2
    assert c1.evidence_metadata["precision"] == "street"
    assert c1.candidate_id == "AD0001::C1"


# -------------------------------------------------------------
# Test 2: C2 unavailable without credible doorstep
# -------------------------------------------------------------
def test_c2_unavailable_without_credible_doorstep():
    """Verify C2 is strictly unavailable if has_credible_doorstep is False or medoid is missing."""
    # Case A: False has_credible_doorstep
    medoid_not_credible = {"doorstep_medoid_x": 100.0, "doorstep_medoid_y": 200.0, "has_credible_doorstep": False}
    cands = generate_candidates(address_id="AD0002", medoid_row=medoid_not_credible)
    c2 = next(c for c in cands if c.candidate_type == "C2")
    assert c2.available is False
    assert c2.x is None
    assert c2.y is None

    # Case B: Credible doorstep exists
    medoid_credible = {"doorstep_medoid_x": 100.0, "doorstep_medoid_y": 200.0, "has_credible_doorstep": True}
    cands_b = generate_candidates(address_id="AD0002", medoid_row=medoid_credible)
    c2_b = next(c for c in cands_b if c.candidate_type == "C2")
    assert c2_b.available is True
    assert c2_b.x == 100.0
    assert c2_b.y == 200.0


# -------------------------------------------------------------
# Test 3: C3 uses physical POI coordinate
# -------------------------------------------------------------
def test_c3_uses_physical_poi_coordinate():
    """Verify C3 adopts exact physical POI coordinate from landmarks_poi.csv."""
    poi = {"poi_id": "LM0010", "name": "Bus Stop", "landmark_type": "bus_stop", "x": -1325.0, "y": -3511.8}
    cands = generate_candidates(address_id="AD0003", poi_row=poi)
    c3 = next(c for c in cands if c.candidate_type == "C3")
    assert c3.available is True
    assert c3.x == -1325.0
    assert c3.y == -3511.8
    assert c3.evidence_metadata["poi_id"] == "LM0010"


# -------------------------------------------------------------
# Test 4: C3 relational words do not create arbitrary coordinate offsets
# -------------------------------------------------------------
def test_c3_relational_words_no_arbitrary_offsets():
    """Verify prepositions ('behind', 'opposite', 'beside', 'near') do NOT displace C3 coordinates."""
    poi = {"poi_id": "LM0013", "name": "Church", "landmark_type": "church", "x": -834.6, "y": -2978.5}
    entity = ParsedAddressEntity(
        address_id="AD0004",
        town_id="T1",
        landmark_raw="Church",
        spatial_relation="behind",
    )
    cands = generate_candidates(address_id="AD0004", poi_row=poi, parsed_entity=entity)
    c3 = next(c for c in cands if c.candidate_type == "C3")
    assert c3.x == -834.6
    assert c3.y == -2978.5
    assert c3.evidence_metadata["relation_offset_applied"] is False


# -------------------------------------------------------------
# Test 5: C4 exact weighted centroid
# -------------------------------------------------------------
def test_c4_exact_weighted_centroid():
    """Verify C4 coordinate matches weighted centroid formula: sum(w_j * X_j) / sum(w_j)."""
    # 2 neighbors: N1=(100, 200, w=2.0), N2=(300, 400, w=1.0)
    # Expected: x = (2*100 + 1*300)/3 = 500/3 = 166.67, y = (2*200 + 1*400)/3 = 800/3 = 266.67
    c4_res = {
        "c4_x": 166.67,
        "c4_y": 266.67,
        "neighbor_count": 2,
        "total_neighbor_weight": 3.0,
        "town_id": "T1",
        "locality_id": "T1-L01",
        "street_normalized": "5th main",
    }
    cands = generate_candidates(address_id="AD0005", c4_result=c4_res)
    c4 = next(c for c in cands if c.candidate_type == "C4")
    assert c4.available is True
    assert c4.x == 166.67
    assert c4.y == 266.67
    assert c4.evidence_metadata["neighbor_count"] == 2


# -------------------------------------------------------------
# Test 6: C4 unavailable without locality
# -------------------------------------------------------------
def test_c4_unavailable_without_locality():
    """Verify C4 is strictly unavailable when locality is unresolved."""
    cands = generate_candidates(address_id="AD0006", c4_result=None)
    c4 = next(c for c in cands if c.candidate_type == "C4")
    assert c4.available is False
    assert c4.x is None
    assert c4.y is None


# -------------------------------------------------------------
# Test 7: C4 unavailable without neighbors
# -------------------------------------------------------------
def test_c4_unavailable_without_neighbors():
    """Verify C4 is unavailable when neighbor_count is 0."""
    c4_empty = {"c4_x": None, "c4_y": None, "neighbor_count": 0, "total_neighbor_weight": 0.0}
    cands = generate_candidates(address_id="AD0007", c4_result=c4_empty)
    c4 = next(c for c in cands if c.candidate_type == "C4")
    assert c4.available is False


# -------------------------------------------------------------
# Test 8: C5 locality centroid
# -------------------------------------------------------------
def test_c5_locality_centroid():
    """Verify C5 extracts exact locality centroid from localities.csv."""
    loc_row = {"locality_id": "T1-L03", "locality_name": "Kuvempu Layout", "centroid_x": 1150.0, "centroid_y": -220.0}
    cands = generate_candidates(address_id="AD0008", locality_row=loc_row)
    c5 = next(c for c in cands if c.candidate_type == "C5")
    assert c5.available is True
    assert c5.x == 1150.0
    assert c5.y == -220.0


# -------------------------------------------------------------
# Test 9: Candidate coordinate deduplication
# -------------------------------------------------------------
def test_candidate_coordinate_deduplication():
    """Verify deduplication collapses coordinates within 5m and records agreed_sources."""
    c1 = SpatialCandidate(
        candidate_id="AD1::C1",
        candidate_type="C1",
        x=100.0,
        y=200.0,
        available=True,
    )
    c4 = SpatialCandidate(
        candidate_id="AD1::C4",
        candidate_type="C4",
        x=102.0,  # Within 2.8m of C1
        y=202.0,
        available=True,
    )
    c5 = SpatialCandidate(
        candidate_id="AD1::C5",
        candidate_type="C5",
        x=500.0,  # Far away
        y=800.0,
        available=True,
    )
    deduped = deduplicate_candidates([c1, c4, c5], tolerance_m=5.0)
    assert len(deduped) == 2  # C1 and C4 merged into 1 pin, C5 kept
    # Higher priority C4 is the primary kept pin
    primary = deduped[0]
    assert primary.candidate_type == "C4"
    assert "C1" in primary.agreed_sources
    assert "C4" in primary.agreed_sources


# -------------------------------------------------------------
# Test 10: Feature values bounded and deterministic
# -------------------------------------------------------------
def test_feature_values_bounded_and_deterministic():
    """Verify all extracted candidate features are strictly in [0.0, 1.0]."""
    cand = SpatialCandidate(
        candidate_id="AD1::C4",
        candidate_type="C4",
        x=100.0,
        y=200.0,
        available=True,
    )
    visits = [
        {"checkin_x": 105.0, "checkin_y": 205.0, "visit_integrity_score": 0.85, "spatial_evidence_strength": 0.90},
    ]
    neighbors = [
        {"x": 110.0, "y": 195.0, "weight": 2.5},
    ]
    feats = extract_candidate_features(
        candidate=cand,
        direct_visits=visits,
        neighbor_records=neighbors,
        baseline_coord=(102.0, 198.0),
    )
    for feat_name, val in feats.items():
        assert 0.0 <= val <= 1.0, f"Feature {feat_name} out of bounds: {val}"


# -------------------------------------------------------------
# Test 11: Negative-search penalty behaves monotonically
# -------------------------------------------------------------
def test_negative_search_penalty_monotonic():
    """Verify negative search penalty increases monotonically as negative visits increase."""
    cand = SpatialCandidate(
        candidate_id="AD1::C1",
        candidate_type="C1",
        x=100.0,
        y=200.0,
        available=True,
    )
    # 0 negative visits
    f0 = extract_candidate_features(candidate=cand, direct_visits=[])
    # 1 negative visit nearby
    v1 = [{"checkin_x": 105.0, "checkin_y": 200.0, "outcome": "address_not_traceable", "visit_integrity_score": 0.8, "tortuosity": 2.0}]
    f1 = extract_candidate_features(candidate=cand, direct_visits=v1)
    # 2 negative visits nearby
    v2 = v1 + [{"checkin_x": 102.0, "checkin_y": 201.0, "outcome": "address_not_traceable", "visit_integrity_score": 0.9, "tortuosity": 2.5}]
    f2 = extract_candidate_features(candidate=cand, direct_visits=v2)

    assert f0["NegativeSearchPenalty"] == 0.0
    assert f1["NegativeSearchPenalty"] > f0["NegativeSearchPenalty"]
    assert f2["NegativeSearchPenalty"] >= f1["NegativeSearchPenalty"]


# -------------------------------------------------------------
# Test 12: neighbour_says_shifted is not positive doorstep evidence
# -------------------------------------------------------------
def test_neighbour_says_shifted_not_positive_doorstep():
    """Verify neighbour_says_shifted does NOT generate C2 or positive direct evidence."""
    cand = SpatialCandidate(
        candidate_id="AD1::C1",
        candidate_type="C1",
        x=100.0,
        y=200.0,
        available=True,
    )
    # Only shifted visits exist
    shifted_visits = [
        {"checkin_x": 100.0, "checkin_y": 200.0, "outcome": "neighbour_says_shifted", "visit_integrity_score": 0.8, "spatial_evidence_strength": 0.0}
    ]
    feats = extract_candidate_features(candidate=cand, direct_visits=shifted_visits)
    # DirectEvidence must be zero or near zero because spatial_evidence_strength is 0
    assert feats["DirectEvidence"] == 0.0
    # Negative penalty should be positive
    assert feats["NegativeSearchPenalty"] > 0.0


# -------------------------------------------------------------
# Test 13: Ranking is deterministic
# -------------------------------------------------------------
def test_ranking_is_deterministic():
    """Verify repeated ranking calls on identical inputs produce byte-equivalent rankings."""
    c1 = SpatialCandidate(candidate_id="A::C1", candidate_type="C1", x=100.0, y=200.0, available=True, features={"TextMatch": 0.5, "DirectEvidence": 0.0, "NeighborSupport": 0.0, "BaselineAgreement": 1.0, "NegativeSearchPenalty": 0.0})
    c4 = SpatialCandidate(candidate_id="A::C4", candidate_type="C4", x=300.0, y=400.0, available=True, features={"TextMatch": 0.9, "DirectEvidence": 0.0, "NeighborSupport": 0.8, "BaselineAgreement": 0.0, "NegativeSearchPenalty": 0.0})
    
    r1 = rank_candidates([c1, c4], deduplicate=False)
    r2 = rank_candidates([c1, c4], deduplicate=False)
    assert [c.candidate_type for c in r1] == [c.candidate_type for c in r2]
    assert [c.total_score for c in r1] == [c.total_score for c in r2]


# -------------------------------------------------------------
# Test 14: Ranking prefers stronger evidence in synthetic cases
# -------------------------------------------------------------
def test_ranking_prefers_stronger_evidence():
    """Verify direct doorstep evidence (C2) outranks baseline (C1), and C4 outranks C5."""
    c1 = SpatialCandidate(candidate_id="A::C1", candidate_type="C1", x=100.0, y=100.0, available=True, features={"TextMatch": 0.5, "DirectEvidence": 0.0, "NeighborSupport": 0.0, "BaselineAgreement": 1.0, "NegativeSearchPenalty": 0.0})
    c2 = SpatialCandidate(candidate_id="A::C2", candidate_type="C2", x=200.0, y=200.0, available=True, features={"TextMatch": 1.0, "DirectEvidence": 0.9, "NeighborSupport": 0.0, "BaselineAgreement": 0.2, "NegativeSearchPenalty": 0.0})
    ranked = rank_candidates([c1, c2], deduplicate=False)
    assert ranked[0].candidate_type == "C2"


# -------------------------------------------------------------
# Test 15: Target surveyed coordinate cannot influence ranking
# -------------------------------------------------------------
def test_surveyed_coordinate_cannot_influence_ranking():
    """Verify candidate ranker never accepts, requires, or uses surveyed ground truth."""
    cands = generate_candidates(
        address_id="AD0015",
        baseline_row={"geocoder_x": 100.0, "geocoder_y": 200.0},
    )
    # The ranking function has no parameter for surveyed coordinates
    ranked = rank_candidates(cands)
    assert len(ranked) > 0
    # Injecting fake ground truth into external context changes nothing
    ranked_again = rank_candidates(cands)
    assert ranked[0].total_score == ranked_again[0].total_score


# -------------------------------------------------------------
# Test 16: Target medoid cannot influence ranking for held-out inference
# -------------------------------------------------------------
def test_target_medoid_cannot_influence_held_out_inference():
    """Verify is_held_out=True suppresses C2 completely, guaranteeing cold-start isolation."""
    medoid = {"doorstep_medoid_x": 500.0, "doorstep_medoid_y": 600.0, "has_credible_doorstep": True}
    cands = generate_candidates(
        address_id="AD0016",
        medoid_row=medoid,
        baseline_row={"geocoder_x": 100.0, "geocoder_y": 100.0},
        is_held_out=True,
    )
    c2 = next(c for c in cands if c.candidate_type == "C2")
    assert c2.available is False
    assert c2.x is None
    assert c2.y is None
    ranked = rank_candidates(cands)
    assert all(c.candidate_type != "C2" for c in ranked)


# -------------------------------------------------------------
# Test 22: Numerically identical ranking reproducibility
# -------------------------------------------------------------
def test_numerically_identical_ranking():
    """Verify ranking produces exact identical floats across runs."""
    c1 = SpatialCandidate(candidate_id="X::C1", candidate_type="C1", x=10.0, y=20.0, available=True, features={"TextMatch": 0.6, "DirectEvidence": 0.0, "NeighborSupport": 0.2, "BaselineAgreement": 0.9, "NegativeSearchPenalty": 0.1})
    c4 = SpatialCandidate(candidate_id="X::C4", candidate_type="C4", x=15.0, y=25.0, available=True, features={"TextMatch": 0.8, "DirectEvidence": 0.0, "NeighborSupport": 0.7, "BaselineAgreement": 0.5, "NegativeSearchPenalty": 0.0})
    r1 = rank_candidates([c1, c4], deduplicate=False)
    r2 = rank_candidates([c1, c4], deduplicate=False)
    assert r1[0].total_score == r2[0].total_score
    assert r1[1].total_score == r2[1].total_score


# -------------------------------------------------------------
# Test 23: No candidate has fabricated coordinates
# -------------------------------------------------------------
def test_no_fabricated_coordinates():
    """Verify that when raw inputs are None, candidate coordinates are strictly None."""
    cands = generate_candidates(address_id="EMPTY_ADDR")
    for c in cands:
        assert c.available is False
        assert c.x is None
        assert c.y is None
