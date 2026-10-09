"""Unit tests for Locality-Bounded POI Resolver and Landmark Disambiguator.

Authority: PS3_FINAL_ARCHITECTURE.md Section 2, PS3_DATA_CONTRACT.md Section 2 (Proof 1),
and PS3_MODEL_SPEC.md Section 5 (Candidate C3).
Test Requirements: Items 15 through 24 of Phase 2 Specification.
"""

import pandas as pd
import pytest

from neurogeo.config import PATHS
from neurogeo.poi_resolver import (
    _compute_poi_locality_assignments,
    compute_landmark_type_similarities,
    compute_text_poi_similarity,
    resolve_poi,
    resolve_poi_detailed,
)


@pytest.fixture
def df_pois():
    """Load official landmarks reference table."""
    return pd.read_csv(PATHS.landmarks_poi_csv)


@pytest.fixture
def df_localities():
    """Load official localities reference table."""
    return pd.read_csv(PATHS.localities_csv)


# ---------------------------------------------------------------------------
# Test 15: Correct exact POI resolves
# ---------------------------------------------------------------------------
def test_correct_exact_poi_resolves(df_pois, df_localities):
    # In T3-L05 (Lakeview Phase 2), Community Hall is LM0258
    result = resolve_poi_detailed(
        extracted_landmark_text="Community Hall",
        town_id="T3",
        locality_id="T3-L05",
        df_pois=df_pois,
        df_localities=df_localities,
    )
    assert result.status == "resolved"
    assert result.poi_id == "LM0258"
    assert result.x is not None
    assert result.y is not None
    assert result.confidence >= 0.90


# ---------------------------------------------------------------------------
# Test 16: Minor spelling variation resolves correctly when confidence sufficient
# ---------------------------------------------------------------------------
def test_minor_spelling_variation_resolves(df_pois, df_localities):
    # 'kalyana mantapa' in T3-L05 matches Community Hall (LM0258)
    result = resolve_poi_detailed(
        extracted_landmark_text="kalyana mantapa",
        town_id="T3",
        locality_id="T3-L05",
        df_pois=df_pois,
        df_localities=df_localities,
    )
    assert result.status == "resolved"
    assert result.poi_id == "LM0258"
    assert result.confidence >= 0.65


# ---------------------------------------------------------------------------
# Test 17: Wrong-town POI is not selected when town is known
# ---------------------------------------------------------------------------
def test_wrong_town_poi_not_selected(df_pois, df_localities):
    # Look for a POI specifying town T1; POI candidates must strictly be from T1
    result = resolve_poi_detailed(
        extracted_landmark_text="Bus Stop",
        town_id="T1",
        locality_id="T1-L01",
        df_pois=df_pois,
        df_localities=df_localities,
    )
    assert result.status == "resolved"
    poi_row = df_pois[df_pois["poi_id"] == result.poi_id].iloc[0]
    assert poi_row["town_id"] == "T1"
    assert poi_row["town_id"] != "T2"
    assert poi_row["town_id"] != "T3"


# ---------------------------------------------------------------------------
# Test 18: Wrong-locality POI is not preferred over compatible candidate
# ---------------------------------------------------------------------------
def test_wrong_locality_poi_not_preferred(df_pois, df_localities):
    # Bus Stop in T1-L01 vs Bus Stop in T1-L02
    res_l01 = resolve_poi_detailed("Bus Stop", "T1", "T1-L01", df_pois, df_localities)
    res_l02 = resolve_poi_detailed("Bus Stop", "T1", "T1-L02", df_pois, df_localities)
    
    assert res_l01.status == "resolved"
    assert res_l02.status == "resolved"
    # Scoping by locality must pick distinct POIs
    assert res_l01.poi_id != res_l02.poi_id


# ---------------------------------------------------------------------------
# Test 19: Ambiguous POIs are explicitly marked ambiguous
# ---------------------------------------------------------------------------
def test_ambiguous_pois_explicitly_marked(df_pois, df_localities):
    # Without locality scoping, generic "Bus Stop" occurs in multiple localities in T1
    result = resolve_poi_detailed(
        extracted_landmark_text="Bus Stop",
        town_id="T1",
        locality_id=None,  # No locality
        df_pois=df_pois,
        df_localities=df_localities,
    )
    # Must report ambiguous, NOT arbitrarily pick one
    assert result.status == "ambiguous"
    assert result.poi_id is None
    assert len(result.candidate_matches) >= 2


# ---------------------------------------------------------------------------
# Test 20: Unresolvable POIs remain unresolved
# ---------------------------------------------------------------------------
def test_unresolvable_pois_remain_unresolved(df_pois, df_localities):
    # Fictitious landmark that does not exist in the town
    result = resolve_poi_detailed(
        extracted_landmark_text="Eiffel Tower Aerospace Center",
        town_id="T1",
        locality_id="T1-L01",
        df_pois=df_pois,
        df_localities=df_localities,
    )
    assert result.status == "unresolved"
    assert result.poi_id is None


# ---------------------------------------------------------------------------
# Test 21: Structural uniqueness is not treated as automatic proof of correctness
# ---------------------------------------------------------------------------
def test_structural_uniqueness_not_proof_of_correctness(df_pois, df_localities):
    # Even though a locality has a unique Petrol Bunk, if the mention is completely dissimilar,
    # the resolver MUST NOT force a match simply because the POI is unique!
    result = resolve_poi_detailed(
        extracted_landmark_text="Random Unknown Grocery Stall",
        town_id="T1",
        locality_id="T1-L01",
        df_pois=df_pois,
        df_localities=df_localities,
    )
    assert result.status == "unresolved"
    assert result.poi_id is None


# ---------------------------------------------------------------------------
# Test 22: Relational terms do not modify POI coordinates
# ---------------------------------------------------------------------------
def test_relational_terms_do_not_modify_poi_coords(df_pois, df_localities):
    # Physical coordinates of LM0258
    true_row = df_pois[df_pois["poi_id"] == "LM0258"].iloc[0]
    expected_x = float(true_row["x"])
    expected_y = float(true_row["y"])
    
    # Resolve POI with tuple wrapper
    resolved = resolve_poi("Community Hall", "T3", "T3-L05", df_pois, df_localities)
    assert resolved is not None
    poi_id, res_x, res_y = resolved
    
    assert poi_id == "LM0258"
    # Strict invariant: exactly matches physical catalog coordinates (0 heuristic offset)
    assert res_x == expected_x
    assert res_y == expected_y


# ---------------------------------------------------------------------------
# Test 23: Resolver is deterministic
# ---------------------------------------------------------------------------
def test_resolver_determinism(df_pois, df_localities):
    r1 = resolve_poi_detailed("Govt School", "T2", "T2-L03", df_pois, df_localities)
    r2 = resolve_poi_detailed("Govt School", "T2", "T2-L03", df_pois, df_localities)
    assert r1.status == r2.status
    assert r1.poi_id == r2.poi_id
    assert r1.confidence == r2.confidence
    assert r1.x == r2.x
    assert r1.y == r2.y


# ---------------------------------------------------------------------------
# Test 24: No surveyed coordinates accessed during Phase 2 inference
# ---------------------------------------------------------------------------
def test_no_surveyed_coords_accessed(monkeypatch, df_pois, df_localities):
    # Ensure surveyed_addresses.csv is never accessed or imported by Phase 2 code
    import sys
    
    # Poison surveyed_addresses file reading
    def poisoned_read_csv(*args, **kwargs):
        path_str = str(args[0]) if args else ""
        if "surveyed_addresses" in path_str:
            raise PermissionError("LEAKAGE ALERT: Surveyed coordinates accessed during Phase 2 inference!")
        return pd.read_csv_orig(*args, **kwargs)
        
    pd.read_csv_orig = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", poisoned_read_csv)
    
    # Executing resolver and parser must not trigger poisoned read
    res = resolve_poi_detailed("Ganesh Temple", "T1", "T1-L04", df_pois, df_localities)
    assert res.status in ("resolved", "unresolved")
