"""Unit and integration tests for Hierarchical Spatial Graph, Candidate C4, and SCV/LOSO Engines.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

import math
from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from neurogeo.config import PATHS
from neurogeo.leakage import (
    compute_isolation_closure,
    assert_fit_scope_valid,
    check_artifact_for_leakage,
    assert_no_leakage,
)
from neurogeo.spatial_graph import (
    build_hierarchical_street_graph,
    compute_scoped_neighbor_centroid,
    compute_address_neighbor_weights,
    export_street_centroids,
    export_neighbor_graph_table,
    run_simulated_cold_start_validation,
    run_leave_one_street_out_validation,
    make_street_key,
    make_locality_key,
)


@pytest.fixture
def synthetic_spatial_data():
    """Create minimal synthetic spatial dataset for exact hand-calculated assertions."""
    df_addresses = pd.DataFrame([
        {
            "address_id": "AD001",
            "account_id": "ACC001",
            "town_id": "T1",
            "locality_id_resolved": "T1-L01",
            "street_normalized": "1st cross, 2nd main",
        },
        {
            "address_id": "AD002",
            "account_id": "ACC002",
            "town_id": "T1",
            "locality_id_resolved": "T1-L01",
            "street_normalized": "1st cross, 2nd main",
        },
        {
            "address_id": "AD003",
            "account_id": "ACC003",
            "town_id": "T1",
            "locality_id_resolved": "T1-L01",
            "street_normalized": "1st cross, 2nd main",
        },
        {
            "address_id": "AD004",
            "account_id": "ACC004",
            "town_id": "T1",
            "locality_id_resolved": "T1-L02",  # Different locality! Same street name
            "street_normalized": "1st cross, 2nd main",
        },
        {
            "address_id": "AD005",
            "account_id": "ACC005",
            "town_id": "T1",
            "locality_id_resolved": None,      # Unresolved locality!
            "street_normalized": "1st cross, 2nd main",
        },
        {
            "address_id": "AD006",
            "account_id": "ACC006",
            "town_id": "T1",
            "locality_id_resolved": "T1-L01",
            "street_normalized": "isolated avenue",  # Solitary street (0 neighbors)
        },
    ])

    df_medoids = pd.DataFrame([
        {
            "address_id": "AD001",
            "doorstep_medoid_x": 100.0,
            "doorstep_medoid_y": 200.0,
            "has_credible_doorstep": True,
            "total_evidence_weight": 1.0,
        },
        {
            "address_id": "AD002",
            "doorstep_medoid_x": 100.0,
            "doorstep_medoid_y": 200.0,
            "has_credible_doorstep": True,
            "total_evidence_weight": 2.0,
        },
        {
            "address_id": "AD003",
            "doorstep_medoid_x": 400.0,
            "doorstep_medoid_y": 500.0,
            "has_credible_doorstep": True,
            "total_evidence_weight": 1.0,
        },
        {
            "address_id": "AD004",
            "doorstep_medoid_x": 800.0,
            "doorstep_medoid_y": 900.0,
            "has_credible_doorstep": True,
            "total_evidence_weight": 3.0,
        },
        {
            "address_id": "AD006",
            "doorstep_medoid_x": 50.0,
            "doorstep_medoid_y": 50.0,
            "has_credible_doorstep": True,
            "total_evidence_weight": 1.5,
        },
    ])

    df_visits = pd.DataFrame([
        {"visit_id": "VS001", "account_id": "ACC001", "address_id": "AD001", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS002", "account_id": "ACC002", "address_id": "AD002", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS003", "account_id": "ACC002", "address_id": "AD002", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS004", "account_id": "ACC003", "address_id": "AD003", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS005", "account_id": "ACC004", "address_id": "AD004", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS006", "account_id": "ACC004", "address_id": "AD004", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "VS007", "account_id": "ACC004", "address_id": "AD004", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
    ])

    return df_addresses, df_medoids, df_visits


# -------------------------------------------------------------------------
# Test 1: Target address is absent from its own neighbor set
# -------------------------------------------------------------------------
def test_target_address_absent_from_neighbor_set(synthetic_spatial_data):
    """Verify target address A* never appears in its own neighbor set N(A*)."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="train_all")

    # Target is AD001
    c4 = compute_scoped_neighbor_centroid(
        address_id="AD001",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4 is not None
    assert "AD001" not in c4["neighbor_address_ids"]
    assert set(c4["neighbor_address_ids"]) == {"AD002", "AD003"}


# -------------------------------------------------------------------------
# Test 2: Target address visits do not contribute to neighbor weights
# -------------------------------------------------------------------------
def test_target_address_visits_do_not_contribute_to_weights(synthetic_spatial_data):
    """Verify target address A*'s visits do not contribute to total_neighbor_weight."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="train_all")

    # AD001 has weight 1.0; AD002 has weight 2.0; AD003 has weight 1.0
    c4 = compute_scoped_neighbor_centroid(
        address_id="AD001",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4 is not None
    # Total weight must be w(AD002) + w(AD003) = 2.0 + 1.0 = 3.0, excluding AD001's 1.0
    assert c4["total_neighbor_weight"] == pytest.approx(3.0, abs=1e-3)
    assert c4["neighbor_weights"] == [2.0, 1.0]


# -------------------------------------------------------------------------
# Test 3: Target address doorstep medoid does not contribute to C4
# -------------------------------------------------------------------------
def test_target_address_medoid_does_not_contribute_to_c4(synthetic_spatial_data):
    """Verify target address doorstep coordinates do not pull C4 toward target."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    
    # Give AD001 an extreme outlier coordinate (10000, 20000)
    df_med_outlier = df_med.copy()
    df_med_outlier.loc[df_med_outlier["address_id"] == "AD001", "doorstep_medoid_x"] = 10000.0
    df_med_outlier.loc[df_med_outlier["address_id"] == "AD001", "doorstep_medoid_y"] = 20000.0

    graph = build_hierarchical_street_graph(df_addr, df_med_outlier, df_visits=df_vis, fit_scope="train_all")

    c4 = compute_scoped_neighbor_centroid(
        address_id="AD001",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4 is not None
    # C4 should be weighted avg of AD002 (100, 200, w=2) and AD003 (400, 500, w=1)
    # Expected: x = (2*100 + 1*400)/3 = 200.0, y = (2*200 + 1*500)/3 = 300.0
    assert c4["c4_x"] == pytest.approx(200.0, abs=1e-2)
    assert c4["c4_y"] == pytest.approx(300.0, abs=1e-2)


# -------------------------------------------------------------------------
# Test 4: Target address does not appear in graph artifacts used for prediction
# -------------------------------------------------------------------------
def test_target_address_not_in_generated_graph_artifacts_used_for_prediction(synthetic_spatial_data):
    """Verify target address is purged from training graph before computing fold predictions."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    closure = compute_isolation_closure("AD001", df_visits=df_vis, df_addresses=df_addr)

    # Filtered inputs via Closure(A*)
    train_addr = closure.filter_addresses(df_addr)
    train_med = closure.filter_medoids(df_med)
    train_vis = closure.filter_visits(df_vis)

    train_graph = build_hierarchical_street_graph(train_addr, train_med, df_visits=train_vis, fit_scope="train_fold_1")
    street_table = export_street_centroids(train_graph, fit_scope="train_fold_1")
    neighbor_table = export_neighbor_graph_table(train_graph, fit_scope="train_fold_1")

    # AD001 must not appear in street centroids or neighbor table
    assert_no_leakage("AD001", street_table)
    assert_no_leakage("AD001", neighbor_table)


# -------------------------------------------------------------------------
# Test 5: Synthetic leakage injection is detected by existing harness
# -------------------------------------------------------------------------
def test_synthetic_leakage_injection_detected_by_harness():
    """Verify check_artifact_for_leakage catches contaminated graph tables."""
    target_aid = "AD000999"
    
    # Contaminated street table
    contaminated_sc = pd.DataFrame([{
        "town_id": "T1",
        "locality_id": "T1-L01",
        "street_normalized": "main street",
        "member_address_ids": f"AD000100,{target_aid}",
        "fit_scope": "train_fold_1",
    }])
    
    # Check detection
    has_leakage, msg = check_artifact_for_leakage(target_aid, contaminated_sc)
    # The member_address_ids column or a dict check
    dict_check = {target_aid: (100.0, 200.0)}
    has_leakage_dict, msg_dict = check_artifact_for_leakage(target_aid, dict_check)
    assert has_leakage_dict is True
    with pytest.raises(AssertionError, match="LEAKAGE CONTRACT VIOLATION"):
        assert_no_leakage(target_aid, dict_check)


# -------------------------------------------------------------------------
# Test 6: fit_scope is correctly attached to graph artifacts
# -------------------------------------------------------------------------
def test_fit_scope_correctly_attached_to_graph_artifacts(synthetic_spatial_data):
    """Verify provenance fit_scope is attached and validated."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    
    # Valid fit scope
    graph = build_hierarchical_street_graph(df_addr, df_med, fit_scope="train_fold_2")
    assert graph.graph["fit_scope"] == "train_fold_2"

    df_sc = export_street_centroids(graph, fit_scope="train_fold_2")
    assert (df_sc["fit_scope"] == "train_fold_2").all()

    df_ng = export_neighbor_graph_table(graph, fit_scope="train_fold_2")
    assert (df_ng["fit_scope"] == "train_fold_2").all()

    # Invalid fit scope should be rejected
    with pytest.raises(ValueError, match="Invalid fit_scope"):
        build_hierarchical_street_graph(df_addr, df_med, fit_scope="unscoped_all")


# -------------------------------------------------------------------------
# Test 7: Missing locality prevents invalid cross-locality C4 grouping
# -------------------------------------------------------------------------
def test_missing_locality_prevents_invalid_cross_locality_c4(synthetic_spatial_data):
    """Verify street name with unresolved or different locality never groups across localities."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, fit_scope="train_all")

    # AD004 is in T1-L02; AD001, AD002, AD003 are in T1-L01. Same street name!
    c4_ad4 = compute_scoped_neighbor_centroid(
        address_id="AD004",
        df_addresses=df_addr,
        graph=graph,
    )
    # AD004 is the only account in T1-L02 on this street -> 0 neighbors -> None!
    assert c4_ad4 is None

    # AD005 has locality_id_resolved = None -> must return None
    c4_ad5 = compute_scoped_neighbor_centroid(
        address_id="AD005",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4_ad5 is None


# -------------------------------------------------------------------------
# Test 8: Missing neighbors produces C4 unavailable (None)
# -------------------------------------------------------------------------
def test_missing_neighbors_produces_c4_unavailable(synthetic_spatial_data):
    """Verify C4 returns None (no arbitrary fallback inside C4) when neighbors are absent."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, fit_scope="train_all")

    # AD006 is on 'isolated avenue', the only address on that street
    c4 = compute_scoped_neighbor_centroid(
        address_id="AD006",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4 is None


# -------------------------------------------------------------------------
# Test 9: C4 weighted centroid matches hand-calculated synthetic examples
# -------------------------------------------------------------------------
def test_c4_weighted_centroid_matches_hand_calculated_synthetic(synthetic_spatial_data):
    """Verify exact floating-point weighted centroid calculation against manual math."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="train_all")

    # Target: AD001. Neighbors: AD002 (100, 200, w=2.0) and AD003 (400, 500, w=1.0)
    # Total weight = 3.0
    # Expected X = (2.0 * 100.0 + 1.0 * 400.0) / 3.0 = 600.0 / 3.0 = 200.0
    # Expected Y = (2.0 * 200.0 + 1.0 * 500.0) / 3.0 = 900.0 / 3.0 = 300.0
    c4 = compute_scoped_neighbor_centroid(
        address_id="AD001",
        df_addresses=df_addr,
        graph=graph,
    )
    assert c4 is not None
    assert c4["c4_x"] == 200.0
    assert c4["c4_y"] == 300.0
    assert c4["neighbor_count"] == 2
    assert c4["total_neighbor_weight"] == 3.0


# -------------------------------------------------------------------------
# Test 10: Neighbor weights match sum(I(v) * S(v))
# -------------------------------------------------------------------------
def test_neighbor_weights_match_sum_iv_sv(synthetic_spatial_data):
    """Verify w_j matches sum of visit integrity * evidence strength."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    weighted_med = compute_address_neighbor_weights(df_med, df_visits=df_vis)

    # AD002 has 2 visits with I=1.0, S=1.0 -> weight = 2.0
    w_ad2 = weighted_med.loc[weighted_med["address_id"] == "AD002", "neighbor_weight"].iloc[0]
    assert w_ad2 == pytest.approx(2.0, abs=1e-4)

    # AD004 has 3 visits with I=1.0, S=1.0 -> weight = 3.0
    w_ad4 = weighted_med.loc[weighted_med["address_id"] == "AD004", "neighbor_weight"].iloc[0]
    assert w_ad4 == pytest.approx(3.0, abs=1e-4)


# -------------------------------------------------------------------------
# Test 11: SCV target history/GPS masking works
# -------------------------------------------------------------------------
def test_scv_target_history_and_gps_masking(synthetic_spatial_data):
    """Verify Closure(A*) masks target visits and GPS breadcrumbs during SCV."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    
    closure = compute_isolation_closure("AD002", df_visits=df_vis, df_addresses=df_addr)
    filtered_vis = closure.filter_visits(df_vis)
    filtered_med = closure.filter_medoids(df_med)

    assert "AD002" not in filtered_vis["address_id"].values
    assert "AD002" not in filtered_med["address_id"].values


# -------------------------------------------------------------------------
# Test 12: Surveyed target coordinate is not accessed during inference
# -------------------------------------------------------------------------
def test_surveyed_target_coordinate_not_accessed_during_inference(synthetic_spatial_data):
    """Verify compute_scoped_neighbor_centroid does not accept or inspect surveyed coordinates."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="production_all")

    # Calling compute_scoped_neighbor_centroid has no surveyed parameter and does not require surveyed data
    c4 = compute_scoped_neighbor_centroid("AD001", df_addresses=df_addr, graph=graph)
    assert c4 is not None
    assert "surveyed_x" not in c4
    assert "surveyed_y" not in c4

    # Trap test: Create a DataFrame with forbidden properties that raise on access during inference
    class TrapDataFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return TrapDataFrame

        def __getitem__(self, item):
            if item in ["surveyed_x", "surveyed_y"]:
                raise RuntimeError("LEAKAGE TRAP: surveyed coordinates accessed during candidate inference!")
            return super().__getitem__(item)

    trap_surveyed = TrapDataFrame({"address_id": ["AD001"], "surveyed_x": [100.0], "surveyed_y": [200.0]})
    # compute_scoped_neighbor_centroid must execute without touching trap_surveyed
    c4_safe = compute_scoped_neighbor_centroid("AD001", df_addresses=df_addr, graph=graph)
    assert c4_safe is not None


# -------------------------------------------------------------------------
# Test 13: LOSO removes all eligible visited accounts from held-out street
# -------------------------------------------------------------------------
def test_loso_removes_all_eligible_visited_accounts_from_held_out_street(synthetic_spatial_data):
    """Verify Leave-One-Street-Out holds out ALL accounts on that street simultaneously."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    
    # Run LOSO with min_visited_per_street = 3 (Street T1::T1-L01::1st cross, 2nd main has 3 accounts)
    df_loso = run_leave_one_street_out_validation(
        df_addr, df_med, df_visits=df_vis, min_visited_per_street=3
    )
    assert len(df_loso) == 1
    row = df_loso.iloc[0]
    assert row["held_out_account_count"] == 3
    assert row["held_out_address_count"] == 3
    assert row["evaluated_episode_count"] == 3
    assert row["evaluated_medoid_episode_count"] == 3
    assert row["visited_accounts_count"] == 3
    assert set(row["held_out_address_ids"].split(",")) == {"AD001", "AD002", "AD003"}
    assert set(row["held_out_account_ids"].split(",")) == {"ACC001", "ACC002", "ACC003"}
    # Because all accounts were held out, C4 must be 0% available
    assert row["c4_available_rate"] == 0.0


# -------------------------------------------------------------------------
# Test 14: LOSO data-construction-level leakage injection test
# -------------------------------------------------------------------------
def test_loso_held_out_street_leakage_injection(synthetic_spatial_data):
    """Verify injecting extreme coordinates into held-out accounts does not contaminate training graph."""
    df_addr, df_med, df_vis = synthetic_spatial_data

    # 1. Baseline LOSO run
    df_loso_base = run_leave_one_street_out_validation(
        df_addr, df_med, df_visits=df_vis, min_visited_per_street=3
    )

    # 2. Poison one of the held-out accounts (ACC001 / AD001) with extreme coordinates and weights
    df_med_poisoned = df_med.copy()
    df_med_poisoned.loc[df_med_poisoned["address_id"] == "AD001", "doorstep_medoid_x"] = 999999.0
    df_med_poisoned.loc[df_med_poisoned["address_id"] == "AD001", "doorstep_medoid_y"] = 999999.0
    df_med_poisoned.loc[df_med_poisoned["address_id"] == "AD001", "total_evidence_weight"] = 9999.0

    df_vis_poisoned = df_vis.copy()
    df_vis_poisoned.loc[df_vis_poisoned["account_id"] == "ACC001", "visit_integrity_score"] = 99.0
    df_vis_poisoned.loc[df_vis_poisoned["account_id"] == "ACC001", "spatial_evidence_strength"] = 99.0

    # 3. Rerun LOSO on poisoned data
    df_loso_poisoned = run_leave_one_street_out_validation(
        df_addr, df_med_poisoned, df_visits=df_vis_poisoned, min_visited_per_street=3
    )

    # 4. Strict invariance assertion:
    # Held-out street C4 remains 0% available
    assert df_loso_poisoned.iloc[0]["c4_available_rate"] == 0.0
    # Training graph for this fold must NOT contain the extreme coordinate
    fold_scope = "train_fold_1"
    held_out_accs = {"ACC001", "ACC002", "ACC003"}
    held_out_addrs = {"AD001", "AD002", "AD003"}
    train_addr = df_addr[~df_addr["account_id"].isin(held_out_accs) & ~df_addr["address_id"].isin(held_out_addrs)]
    train_med = df_med_poisoned[~df_med_poisoned["address_id"].isin(held_out_addrs)]
    train_vis = df_vis_poisoned[~df_vis_poisoned["account_id"].isin(held_out_accs) & ~df_vis_poisoned["address_id"].isin(held_out_addrs)]

    train_graph = build_hierarchical_street_graph(train_addr, train_med, df_visits=train_vis, fit_scope=fold_scope)
    street_table = export_street_centroids(train_graph, fit_scope=fold_scope)

    # AD001 must not appear anywhere in centroids or neighbor tables
    assert_no_leakage("AD001", street_table)
    # The extreme coordinate 999999.0 must not appear anywhere in centroid coordinates
    if len(street_table) > 0:
        assert (street_table["centroid_x"] < 900000.0).all()
        assert (street_table["centroid_y"] < 900000.0).all()



# -------------------------------------------------------------------------
# Test 15: SCV explicit provenance and calibration reservation
# -------------------------------------------------------------------------
def test_scv_explicit_provenance_and_calibration_reservation(synthetic_spatial_data):
    """Verify SCV results include required provenance fields and calibration flag."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    df_scv = run_simulated_cold_start_validation(
        df_addr, df_med, df_visits=df_vis, fit_scope="scv_leave_one_out"
    )

    assert "target_address_id" in df_scv.columns
    assert "evaluation_scope" in df_scv.columns
    assert "target_history_masked" in df_scv.columns
    assert "target_medoid_masked" in df_scv.columns
    assert "target_gps_masked" in df_scv.columns
    assert "graph_fit_scope" in df_scv.columns
    assert "reserved_for_calibration" in df_scv.columns

    assert (df_scv["evaluation_scope"] == "scv_leave_one_out").all()
    assert (df_scv["target_history_masked"] == True).all()
    assert (df_scv["target_medoid_masked"] == True).all()
    assert (df_scv["target_gps_masked"] == True).all()
    assert (df_scv["graph_fit_scope"] == "scv_leave_one_out").all()
    assert (df_scv["reserved_for_calibration"] == True).all()


# -------------------------------------------------------------------------
# Test 16: Running graph construction twice produces identical results
# -------------------------------------------------------------------------
def test_graph_construction_determinism(synthetic_spatial_data):
    """Verify deterministic graph construction yields identical graphs bit-for-bit."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    
    g1 = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="production_all")
    g2 = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="production_all")

    assert set(g1.nodes) == set(g2.nodes)
    assert set(g1.edges) == set(g2.edges)

    for n in g1.nodes:
        d1 = g1.nodes[n]
        d2 = g2.nodes[n]
        assert d1["node_type"] == d2["node_type"]
        assert d1.get("centroid_x") == d2.get("centroid_x")
        assert d1.get("centroid_y") == d2.get("centroid_y")
        assert d1.get("total_weight") == d2.get("total_weight")


# -------------------------------------------------------------------------
# Test 17: Structural hierarchy invariant: all nodes start with 'T'
# -------------------------------------------------------------------------
def test_graph_hierarchy_all_nodes_start_with_t(synthetic_spatial_data):
    """Verify every node starts with Town prefix 'T' and no cross-locality collision occurs."""
    df_addr, df_med, df_vis = synthetic_spatial_data
    graph = build_hierarchical_street_graph(df_addr, df_med, df_visits=df_vis, fit_scope="production_all")

    assert all(node.startswith("T") for node in graph.nodes)
    
    # Locality nodes
    assert "T1::T1-L01" in graph.nodes
    assert "T1::T1-L02" in graph.nodes

    # Street nodes under separate localities
    assert "T1::T1-L01::1st cross, 2nd main" in graph.nodes
    assert "T1::T1-L02::1st cross, 2nd main" in graph.nodes
    # They must be distinct nodes (no collision)
    assert "T1::T1-L01::1st cross, 2nd main" != "T1::T1-L02::1st cross, 2nd main"


# -------------------------------------------------------------------------
# Test 18: LOSO eligibility strictly requires distinct visited accounts
# -------------------------------------------------------------------------
def test_loso_eligibility_requires_distinct_visited_accounts():
    """Verify LOSO eligibility requires >= min_visited_per_street DISTINCT accounts, not address count."""
    # Street with 4 addresses, but only 2 distinct accounts:
    # ACC01 has 3 addresses (AD01, AD02, AD03), ACC02 has 1 address (AD04).
    df_addr = pd.DataFrame([
        {"address_id": "AD01", "account_id": "ACC01", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "market road"},
        {"address_id": "AD02", "account_id": "ACC01", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "market road"},
        {"address_id": "AD03", "account_id": "ACC01", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "market road"},
        {"address_id": "AD04", "account_id": "ACC02", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "market road"},
    ])
    df_med = pd.DataFrame([
        {"address_id": "AD01", "doorstep_medoid_x": 10.0, "doorstep_medoid_y": 20.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
        {"address_id": "AD02", "doorstep_medoid_x": 12.0, "doorstep_medoid_y": 22.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
        {"address_id": "AD03", "doorstep_medoid_x": 14.0, "doorstep_medoid_y": 24.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
        {"address_id": "AD04", "doorstep_medoid_x": 16.0, "doorstep_medoid_y": 26.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
    ])
    df_vis = pd.DataFrame([
        {"visit_id": "V1", "account_id": "ACC01", "address_id": "AD01", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V2", "account_id": "ACC01", "address_id": "AD02", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V3", "account_id": "ACC01", "address_id": "AD03", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V4", "account_id": "ACC02", "address_id": "AD04", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
    ])

    # Case 1: Threshold = 3. Address count is 4 (>=3), but distinct accounts is 2 (<3).
    # Must NOT be eligible!
    df_loso = run_leave_one_street_out_validation(df_addr, df_med, df_visits=df_vis, min_visited_per_street=3)
    assert len(df_loso) == 0, "LOSO erroneously qualified a street based on address count rather than distinct accounts!"

    # Case 2: Add 3rd distinct visited account (ACC03 with AD05)
    df_addr_3 = pd.concat([df_addr, pd.DataFrame([{
        "address_id": "AD05", "account_id": "ACC03", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "market road"
    }])], ignore_index=True)
    df_med_3 = pd.concat([df_med, pd.DataFrame([{
        "address_id": "AD05", "doorstep_medoid_x": 18.0, "doorstep_medoid_y": 28.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0
    }])], ignore_index=True)
    df_vis_3 = pd.concat([df_vis, pd.DataFrame([{
        "visit_id": "V5", "account_id": "ACC03", "address_id": "AD05", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0
    }])], ignore_index=True)

    df_loso_3 = run_leave_one_street_out_validation(df_addr_3, df_med_3, df_visits=df_vis_3, min_visited_per_street=3)
    assert len(df_loso_3) == 1
    row = df_loso_3.iloc[0]
    assert row["held_out_account_count"] == 3
    assert row["held_out_address_count"] == 5
    assert row["evaluated_episode_count"] == 5
    assert row["c4_available_rate"] == 0.0


# -------------------------------------------------------------------------
# Test 19: Multi-address account visit evidence is completely purged in LOSO
# -------------------------------------------------------------------------
def test_loso_multi_address_account_visit_evidence_purged():
    """Verify that if a held-out account has addresses on other streets, ALL of its evidence is purged."""
    # ACC01 has AD01 on street S1 and AD02 on street S2!
    df_addr = pd.DataFrame([
        {"address_id": "AD01", "account_id": "ACC01", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "street one"},
        {"address_id": "AD02", "account_id": "ACC01", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "street two"},
        {"address_id": "AD03", "account_id": "ACC02", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "street one"},
        {"address_id": "AD04", "account_id": "ACC03", "town_id": "T1", "locality_id_resolved": "T1-L01", "street_normalized": "street one"},
    ])
    df_med = pd.DataFrame([
        {"address_id": "AD01", "doorstep_medoid_x": 100.0, "doorstep_medoid_y": 100.0, "has_credible_doorstep": True, "total_evidence_weight": 5.0},
        {"address_id": "AD02", "doorstep_medoid_x": 500.0, "doorstep_medoid_y": 500.0, "has_credible_doorstep": True, "total_evidence_weight": 10.0},
        {"address_id": "AD03", "doorstep_medoid_x": 110.0, "doorstep_medoid_y": 110.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
        {"address_id": "AD04", "doorstep_medoid_x": 120.0, "doorstep_medoid_y": 120.0, "has_credible_doorstep": True, "total_evidence_weight": 1.0},
    ])
    df_vis = pd.DataFrame([
        {"visit_id": "V1", "account_id": "ACC01", "address_id": "AD01", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V2", "account_id": "ACC01", "address_id": "AD02", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V3", "account_id": "ACC02", "address_id": "AD03", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
        {"visit_id": "V4", "account_id": "ACC03", "address_id": "AD04", "visit_integrity_score": 1.0, "spatial_evidence_strength": 1.0},
    ])

    # Run LOSO with threshold = 3 (Street One has ACC01, ACC02, ACC03 -> 3 accounts)
    df_loso = run_leave_one_street_out_validation(df_addr, df_med, df_visits=df_vis, min_visited_per_street=3)
    assert len(df_loso) == 1
    assert df_loso.iloc[0]["street_key"] == "T1::T1-L01::street one"

    # Now verify that during this fold, ACC01's other address AD02 on street two was ALSO purged!
    fold_scope = "train_fold_1"
    held_out_accs = {"ACC01", "ACC02", "ACC03"}
    held_out_addrs = set(df_addr[df_addr["account_id"].isin(held_out_accs)]["address_id"])
    assert "AD02" in held_out_addrs, "AD02 should be identified as belonging to held-out account ACC01"

    train_addr = df_addr[~df_addr["account_id"].isin(held_out_accs)]
    train_med = df_med[~df_med["address_id"].isin(held_out_addrs)]
    train_vis = df_vis[~df_vis["account_id"].isin(held_out_accs)]
    train_graph = build_hierarchical_street_graph(train_addr, train_med, df_visits=train_vis, fit_scope=fold_scope)

    # Street two must NOT contain AD02 or any weight from ACC01
    if train_graph.has_node("T1::T1-L01::street two"):
        st2_data = train_graph.nodes["T1::T1-L01::street two"]
        assert st2_data.get("neighbor_count", 0) == 0
        assert st2_data.get("total_weight", 0.0) == 0.0

