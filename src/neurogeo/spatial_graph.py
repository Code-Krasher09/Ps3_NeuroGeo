"""Hierarchical spatial graph, neighbor propagation, and SCV/LOSO validation engines.

Authority: NEUROGEO_FINAL_ARCHITECTURE.md, NEUROGEO_MODEL_SPEC.md, and NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from neurogeo.config import PATHS
from neurogeo.leakage import (
    IsolationClosure,
    compute_isolation_closure,
    assert_fit_scope_valid,
    validate_fit_scope,
    check_artifact_for_leakage,
    assert_no_leakage,
)
from neurogeo.schemas import ScopedNeighborCentroidResult


def make_street_key(town_id: str, locality_id: str, street_normalized: str) -> str:
    """Construct canonical hierarchical street key: Town::Locality::Street."""
    t = str(town_id).strip()
    loc = str(locality_id).strip()
    st = str(street_normalized).strip().lower()
    return f"{t}::{loc}::{st}"


def make_locality_key(town_id: str, locality_id: str) -> str:
    """Construct canonical hierarchical locality key: Town::Locality."""
    t = str(town_id).strip()
    loc = str(locality_id).strip()
    return f"{t}::{loc}"


def compute_address_neighbor_weights(
    df_medoids: pd.DataFrame,
    df_visits: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute deterministic neighbor weights w_j = sum_{v in V(a_j)} I(v) * S(v).
    
    If df_visits is supplied, sums visit_integrity_score * spatial_evidence_strength directly.
    Otherwise uses precomputed total_evidence_weight from df_medoids.
    """
    df_weights = df_medoids.copy()
    if df_visits is not None and "visit_integrity_score" in df_visits.columns and "spatial_evidence_strength" in df_visits.columns:
        # Sum I(v) * S(v) per address
        v_copy = df_visits.copy()
        iv = v_copy["visit_integrity_score"].to_numpy(dtype=np.float64)
        sv = v_copy["spatial_evidence_strength"].to_numpy(dtype=np.float64)
        v_copy["iv_sv"] = np.maximum(iv * sv, 0.0)
        
        summed = v_copy.groupby("address_id")["iv_sv"].sum().reset_index()
        summed.rename(columns={"iv_sv": "computed_neighbor_weight"}, inplace=True)
        
        df_weights = df_weights.merge(summed, on="address_id", how="left")
        df_weights["neighbor_weight"] = df_weights["computed_neighbor_weight"].fillna(
            df_weights.get("total_evidence_weight", 0.0)
        )
    else:
        if "total_evidence_weight" in df_weights.columns:
            df_weights["neighbor_weight"] = df_weights["total_evidence_weight"].astype(float)
        else:
            df_weights["neighbor_weight"] = 1.0

    return df_weights


def build_hierarchical_street_graph(
    df_addresses: pd.DataFrame,
    df_medoids: pd.DataFrame,
    df_visits: Optional[pd.DataFrame] = None,
    fit_scope: str = "production_all",
) -> nx.DiGraph:
    """Construct hierarchically scoped directed graph (Town -> Locality -> Street).
    
    Enforces the structural invariant:
    - All node IDs start with 'T' (e.g. T1, T1::T1-L01, T1::T1-L01::block d, 5th road).
    - No cross-locality hub node collisions (streets are scoped by town and locality).
    - Validates fit_scope provenance against evaluation contract.
    """
    assert_fit_scope_valid("neighbor_graph", fit_scope)

    # 1. Prepare weights
    df_weighted_medoids = compute_address_neighbor_weights(df_medoids, df_visits=df_visits)

    # 2. Join addresses with medoids
    addr_cols = ["address_id", "town_id", "locality_id_resolved", "street_normalized"]
    if "locality_id_resolved" not in df_addresses.columns and "locality_id" in df_addresses.columns:
        df_addresses = df_addresses.rename(columns={"locality_id": "locality_id_resolved"})
        
    df_addr_sub = df_addresses[[c for c in addr_cols if c in df_addresses.columns]].copy()
    
    merged = df_addr_sub.merge(df_weighted_medoids, on="address_id", how="inner")
    
    # 3. Filter valid scoped addresses
    valid_mask = (
        merged["town_id"].notna() & (merged["town_id"] != "") &
        merged["locality_id_resolved"].notna() & (merged["locality_id_resolved"] != "") &
        merged["street_normalized"].notna() & (merged["street_normalized"] != "")
    )
    df_valid = merged[valid_mask].copy()

    # 4. Build NetworkX DiGraph
    graph = nx.DiGraph(fit_scope=fit_scope)

    # Add all unique towns
    unique_towns = df_valid["town_id"].unique()
    for tid in sorted(unique_towns):
        town_node = str(tid).strip()
        graph.add_node(town_node, node_type="town", town_id=town_node)

    # Group by street key
    df_valid["street_key"] = df_valid.apply(
        lambda r: make_street_key(r["town_id"], r["locality_id_resolved"], r["street_normalized"]),
        axis=1
    )
    df_valid["locality_key"] = df_valid.apply(
        lambda r: make_locality_key(r["town_id"], r["locality_id_resolved"]),
        axis=1
    )

    # Group by locality
    for (town_id, loc_id), grp_loc in df_valid.groupby(["town_id", "locality_id_resolved"]):
        town_node = str(town_id).strip()
        loc_node = make_locality_key(town_id, loc_id)
        if not graph.has_node(loc_node):
            graph.add_node(loc_node, node_type="locality", town_id=town_node, locality_id=str(loc_id).strip())
            graph.add_edge(town_node, loc_node)

    # Group by street
    for (town_id, loc_id, street_norm), grp_street in df_valid.groupby(
        ["town_id", "locality_id_resolved", "street_normalized"]
    ):
        town_node = str(town_id).strip()
        loc_node = make_locality_key(town_id, loc_id)
        street_node = make_street_key(town_id, loc_id, street_norm)

        # Filter to credible doorsteps for spatial neighbor anchors
        cred_subset = grp_street[grp_street["has_credible_doorstep"] == True]

        neighbor_accounts: List[Dict[str, Any]] = []
        total_w = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for _, row in cred_subset.iterrows():
            w = float(row.get("neighbor_weight", row.get("total_evidence_weight", 1.0)))
            if w <= 0:
                continue
            x = float(row["doorstep_medoid_x"])
            y = float(row["doorstep_medoid_y"])
            aid = str(row["address_id"])

            neighbor_accounts.append({
                "address_id": aid,
                "x": round(x, 2),
                "y": round(y, 2),
                "weight": round(w, 4),
            })
            total_w += w
            weighted_x += w * x
            weighted_y += w * y

        n_count = len(neighbor_accounts)
        centroid_x = round(weighted_x / total_w, 2) if total_w > 0 else None
        centroid_y = round(weighted_y / total_w, 2) if total_w > 0 else None

        graph.add_node(
            street_node,
            node_type="street",
            town_id=town_node,
            locality_id=str(loc_id).strip(),
            street_normalized=str(street_norm).strip(),
            street_key=street_node,
            fit_scope=fit_scope,
            neighbor_accounts=neighbor_accounts,
            neighbor_count=n_count,
            total_weight=round(total_w, 4),
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )
        graph.add_edge(loc_node, street_node)

    # Structural Assertion: All node IDs start with 'T'
    assert all(node.startswith("T") for node in graph.nodes), (
        "Structural invariant violated: All graph nodes must start with Town prefix 'T'."
    )

    return graph


# Alias for backwards compatibility
build_neighbor_graph = build_hierarchical_street_graph


def compute_scoped_neighbor_centroid(
    address_id: str,
    town_id: Optional[str] = None,
    locality_id: Optional[str] = None,
    street_normalized: Optional[str] = None,
    df_neighbor_visits: Optional[pd.DataFrame] = None,
    df_addresses: Optional[pd.DataFrame] = None,
    df_medoids: Optional[pd.DataFrame] = None,
    graph: Optional[nx.DiGraph] = None,
    fit_scope: str = "production_all",
) -> Optional[Dict[str, Any]]:
    """Compute Candidate C4 weighted neighbor centroid: sum(w_j * X_j) / sum(w_j).
    
    Adheres strictly to NEUROGEO_MODEL_SPEC.md Section 5 and evaluation zero-leakage contract:
    - Excludes target address A* and all target-derived visit observations.
    - Uses credible visited neighbors only (has_credible_doorstep == True).
    - Requires matching town_id, resolved locality_id, and normalized_street.
    - If locality is unresolvable or street is empty, C4 is explicitly unavailable (None).
    - If zero credible neighbors exist, returns None (no arbitrary fallback inside C4).
    """
    # 1. Resolve scope parameters if missing from target entity table
    target_aid = str(address_id).strip()
    
    if (town_id is None or locality_id is None or street_normalized is None) and df_addresses is not None:
        addr_match = df_addresses[df_addresses["address_id"] == target_aid]
        if len(addr_match) > 0:
            row = addr_match.iloc[0]
            if town_id is None:
                town_id = str(row.get("town_id", "")).strip() or None
            if locality_id is None:
                loc_val = row.get("locality_id_resolved") or row.get("locality_id")
                locality_id = str(loc_val).strip() if pd.notna(loc_val) and str(loc_val).strip() != "" else None
            if street_normalized is None:
                st_val = row.get("street_normalized")
                street_normalized = str(st_val).strip() if pd.notna(st_val) and str(st_val).strip() != "" else None

    # Guard: Town, locality, and street are strictly mandatory for C4
    if not town_id or not locality_id or not street_normalized:
        return None

    target_t = str(town_id).strip()
    target_loc = str(locality_id).strip()
    target_st = str(street_normalized).strip().lower()

    # 2. Extract neighbor accounts from graph or dataframes
    neighbors: List[Dict[str, Any]] = []

    if graph is not None:
        street_node = make_street_key(target_t, target_loc, target_st)
        if graph.has_node(street_node):
            node_data = graph.nodes[street_node]
            raw_neighbors = node_data.get("neighbor_accounts", [])
            # Filter out target address A*
            for n in raw_neighbors:
                if str(n["address_id"]) != target_aid and n["weight"] > 0:
                    neighbors.append(n)
        else:
            return None

    elif df_medoids is not None and df_addresses is not None:
        # Join on the fly
        loc_col = "locality_id_resolved" if "locality_id_resolved" in df_addresses.columns else "locality_id"
        matching_addrs = df_addresses[
            (df_addresses["town_id"] == target_t) &
            (df_addresses[loc_col] == target_loc) &
            (df_addresses["street_normalized"].str.lower() == target_st) &
            (df_addresses["address_id"] != target_aid)
        ]
        if len(matching_addrs) == 0:
            return None

        merged = matching_addrs.merge(df_medoids, on="address_id", how="inner")
        cred_subset = merged[merged["has_credible_doorstep"] == True]

        for _, row in cred_subset.iterrows():
            w = float(row.get("neighbor_weight", row.get("total_evidence_weight", 1.0)))
            if w <= 0:
                continue
            neighbors.append({
                "address_id": str(row["address_id"]),
                "x": float(row["doorstep_medoid_x"]),
                "y": float(row["doorstep_medoid_y"]),
                "weight": w,
            })

    elif df_neighbor_visits is not None:
        # df_neighbor_visits contains candidate neighbor records
        loc_col = "locality_id_resolved" if "locality_id_resolved" in df_neighbor_visits.columns else "locality_id"
        subset = df_neighbor_visits[
            (df_neighbor_visits["town_id"] == target_t) &
            (df_neighbor_visits[loc_col] == target_loc) &
            (df_neighbor_visits["street_normalized"].str.lower() == target_st) &
            (df_neighbor_visits["address_id"] != target_aid)
        ]
        if "has_credible_doorstep" in subset.columns:
            subset = subset[subset["has_credible_doorstep"] == True]

        for _, row in subset.iterrows():
            w = float(row.get("neighbor_weight", row.get("total_evidence_weight", 1.0)))
            if w <= 0:
                continue
            neighbors.append({
                "address_id": str(row["address_id"]),
                "x": float(row.get("doorstep_medoid_x", row.get("x", 0.0))),
                "y": float(row.get("doorstep_medoid_y", row.get("y", 0.0))),
                "weight": w,
            })

    # 3. Guard: If zero credible neighbors exist, return None
    if len(neighbors) == 0:
        return None

    # 4. Compute weighted centroid C4 = sum(w_j * X_j) / sum(w_j)
    total_w = sum(n["weight"] for n in neighbors)
    if total_w <= 0.0:
        return None

    c4_x = sum(n["weight"] * n["x"] for n in neighbors) / total_w
    c4_y = sum(n["weight"] * n["y"] for n in neighbors) / total_w

    return {
        "address_id": target_aid,
        "c4_x": round(float(c4_x), 2),
        "c4_y": round(float(c4_y), 2),
        "neighbor_count": int(len(neighbors)),
        "neighbor_address_ids": [n["address_id"] for n in neighbors],
        "neighbor_weights": [round(float(n["weight"]), 4) for n in neighbors],
        "total_neighbor_weight": round(float(total_w), 4),
        "town_id": target_t,
        "locality_id": target_loc,
        "street_normalized": target_st,
        "fit_scope": fit_scope,
    }


def export_street_centroids(
    graph: Optional[nx.DiGraph] = None,
    df_addresses: Optional[pd.DataFrame] = None,
    df_medoids: Optional[pd.DataFrame] = None,
    df_visits: Optional[pd.DataFrame] = None,
    fit_scope: str = "production_all",
) -> pd.DataFrame:
    """Export tabular street centroids for all qualified street nodes in the graph."""
    assert_fit_scope_valid("street_centroids", fit_scope)

    if graph is None:
        if df_addresses is None or df_medoids is None:
            raise ValueError("Must provide either a pre-built graph or (df_addresses, df_medoids).")
        graph = build_hierarchical_street_graph(
            df_addresses, df_medoids, df_visits=df_visits, fit_scope=fit_scope
        )

    records: List[Dict[str, Any]] = []

    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "street" and data.get("neighbor_count", 0) > 0:
            accounts = data.get("neighbor_accounts", [])
            member_ids = ",".join(n["address_id"] for n in accounts)
            records.append({
                "town_id": data["town_id"],
                "locality_id": data["locality_id"],
                "street_normalized": data["street_normalized"],
                "street_key": data["street_key"],
                "centroid_x": data["centroid_x"],
                "centroid_y": data["centroid_y"],
                "total_weight": data["total_weight"],
                "contributing_addresses_count": data["neighbor_count"],
                "member_address_ids": member_ids,
                "fit_scope": fit_scope,
            })

    df_out = pd.DataFrame(records)
    if len(df_out) > 0:
        df_out.sort_values(["town_id", "locality_id", "street_normalized"], inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    return df_out


def export_neighbor_graph_table(
    graph: Optional[nx.DiGraph] = None,
    df_addresses: Optional[pd.DataFrame] = None,
    df_medoids: Optional[pd.DataFrame] = None,
    df_visits: Optional[pd.DataFrame] = None,
    fit_scope: str = "production_all",
) -> pd.DataFrame:
    """Export tabular neighbor adjacency records (street_key <-> neighbor accounts)."""
    assert_fit_scope_valid("neighbor_graph", fit_scope)

    if graph is None:
        if df_addresses is None or df_medoids is None:
            raise ValueError("Must provide either a pre-built graph or (df_addresses, df_medoids).")
        graph = build_hierarchical_street_graph(
            df_addresses, df_medoids, df_visits=df_visits, fit_scope=fit_scope
        )

    records: List[Dict[str, Any]] = []

    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "street":
            accounts = data.get("neighbor_accounts", [])
            for acc in accounts:
                records.append({
                    "street_key": data["street_key"],
                    "town_id": data["town_id"],
                    "locality_id": data["locality_id"],
                    "street_normalized": data["street_normalized"],
                    "neighbor_address_id": acc["address_id"],
                    "doorstep_medoid_x": acc["x"],
                    "doorstep_medoid_y": acc["y"],
                    "neighbor_weight": acc["weight"],
                    "fit_scope": fit_scope,
                })

    df_out = pd.DataFrame(records)
    if len(df_out) > 0:
        df_out.sort_values(["street_key", "neighbor_address_id"], inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    return df_out


def run_simulated_cold_start_validation(
    df_addresses: pd.DataFrame,
    df_medoids: pd.DataFrame,
    df_visits: Optional[pd.DataFrame] = None,
    df_baseline: Optional[pd.DataFrame] = None,
    df_surveyed: Optional[pd.DataFrame] = None,
    fit_scope: str = "scv_leave_one_out",
) -> pd.DataFrame:
    """Execute Simulated Cold-Start Validation (SCV) across visited addresses.
    
    Zero-Leakage Protocol:
    - For each target address A*, target visits, GPS points, and doorstep medoid are masked.
    - C4 candidate is computed strictly from remaining eligible visited neighbors on the street.
    - Ground-truth comparison is performed strictly post-inference.
    - Evaluates error vs true doorstep medoid (and vs surveyed ground truth for benchmark addresses).
    """
    assert_fit_scope_valid("predictions", fit_scope)

    # 1. Prepare address and medoid lookup
    df_weighted_medoids = compute_address_neighbor_weights(df_medoids, df_visits=df_visits)
    
    addr_cols = ["address_id", "town_id", "locality_id_resolved", "street_normalized"]
    if "locality_id_resolved" not in df_addresses.columns and "locality_id" in df_addresses.columns:
        df_addresses = df_addresses.rename(columns={"locality_id": "locality_id_resolved"})
        
    df_addr_sub = df_addresses[[c for c in addr_cols if c in df_addresses.columns]].copy()
    
    merged = df_addr_sub.merge(df_weighted_medoids, on="address_id", how="inner")

    # Surveyed and baseline lookups
    surveyed_map: Dict[str, Tuple[float, float]] = {}
    if df_surveyed is not None and "address_id" in df_surveyed.columns:
        for _, s_row in df_surveyed.iterrows():
            surveyed_map[str(s_row["address_id"])] = (
                float(s_row["surveyed_x"]),
                float(s_row["surveyed_y"])
            )

    baseline_map: Dict[str, Tuple[float, float]] = {}
    if df_baseline is not None and "address_id" in df_baseline.columns:
        for _, b_row in df_baseline.iterrows():
            baseline_map[str(b_row["address_id"])] = (
                float(b_row["geocoder_x"]),
                float(b_row["geocoder_y"])
            )

    # Filter to credible addresses with valid street and locality to index neighbor pools
    valid_mask = (
        merged["town_id"].notna() & (merged["town_id"] != "") &
        merged["locality_id_resolved"].notna() & (merged["locality_id_resolved"] != "") &
        merged["street_normalized"].notna() & (merged["street_normalized"] != "")
    )
    
    # Credible doorstep pool for providing spatial neighbor evidence
    cred_pool = merged[valid_mask & (merged["has_credible_doorstep"] == True)].copy()
    cred_pool["street_key"] = cred_pool.apply(
        lambda r: make_street_key(r["town_id"], r["locality_id_resolved"], r["street_normalized"]),
        axis=1
    )

    # Fast indexed lookup of credible neighbors by street_key
    street_cred_records: Dict[str, List[Dict[str, Any]]] = {}
    street_cred_totals: Dict[str, Tuple[float, float, float]] = {}  # total_w, total_wx, total_wy

    for sk, group in cred_pool.groupby("street_key"):
        rec_list = []
        tot_w = 0.0
        tot_wx = 0.0
        tot_wy = 0.0
        for _, row in group.iterrows():
            w = float(row.get("neighbor_weight", row.get("total_evidence_weight", 1.0)))
            if w <= 0:
                continue
            x = float(row["doorstep_medoid_x"])
            y = float(row["doorstep_medoid_y"])
            aid = str(row["address_id"])
            rec_list.append({"address_id": aid, "x": x, "y": y, "weight": w})
            tot_w += w
            tot_wx += w * x
            tot_wy += w * y
        street_cred_records[sk] = rec_list
        street_cred_totals[sk] = (tot_w, tot_wx, tot_wy)

    # 2. Iterate through all visited episodes (1,477 rows in df_medoids)
    results: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        aid = str(row["address_id"])
        tid = str(row["town_id"]) if pd.notna(row["town_id"]) else ""
        loc_id = str(row["locality_id_resolved"]) if pd.notna(row["locality_id_resolved"]) else ""
        st_norm = str(row["street_normalized"]) if pd.notna(row["street_normalized"]) else ""
        
        med_x = float(row["doorstep_medoid_x"])
        med_y = float(row["doorstep_medoid_y"])
        has_credible = bool(row["has_credible_doorstep"])

        c4_available = False
        c4_x = np.nan
        c4_y = np.nan
        neighbor_count = 0
        total_neighbor_w = 0.0
        neighbor_ids_str = ""

        # Can C4 be formed? Requires resolved locality and street
        if loc_id and st_norm and tid:
            sk = make_street_key(tid, loc_id, st_norm)
            if sk in street_cred_records:
                all_neighbors = street_cred_records[sk]
                
                # Check if target is in the credible pool for this street
                target_in_pool = any(n["address_id"] == aid for n in all_neighbors)
                
                if target_in_pool:
                    # Subtract target's contribution (O(1) exact leave-one-out)
                    tot_w, tot_wx, tot_wy = street_cred_totals[sk]
                    target_rec = next(n for n in all_neighbors if n["address_id"] == aid)
                    rem_w = tot_w - target_rec["weight"]
                    
                    if rem_w > 0:
                        rem_count = len(all_neighbors) - 1
                        rem_wx = tot_wx - target_rec["weight"] * target_rec["x"]
                        rem_wy = tot_wy - target_rec["weight"] * target_rec["y"]
                        c4_available = True
                        c4_x = round(float(rem_wx / rem_w), 2)
                        c4_y = round(float(rem_wy / rem_w), 2)
                        neighbor_count = rem_count
                        total_neighbor_w = round(float(rem_w), 4)
                        neighbor_ids_str = ",".join(n["address_id"] for n in all_neighbors if n["address_id"] != aid)
                else:
                    # Target is not in pool (e.g. non-credible visit, or cold-start address)
                    tot_w, tot_wx, tot_wy = street_cred_totals[sk]
                    if tot_w > 0:
                        c4_available = True
                        c4_x = round(float(tot_wx / tot_w), 2)
                        c4_y = round(float(tot_wy / tot_w), 2)
                        neighbor_count = len(all_neighbors)
                        total_neighbor_w = round(float(tot_w), 4)
                        neighbor_ids_str = ",".join(n["address_id"] for n in all_neighbors)

        # Errors vs true empirical doorstep medoid
        error_vs_medoid = np.nan
        if c4_available:
            error_vs_medoid = round(float(math.hypot(c4_x - med_x, c4_y - med_y)), 2)

        # Baseline comparison
        base_x, base_y = baseline_map.get(aid, (np.nan, np.nan))
        base_err_vs_medoid = np.nan
        if pd.notna(base_x) and pd.notna(base_y):
            base_err_vs_medoid = round(float(math.hypot(base_x - med_x, base_y - med_y)), 2)

        # Surveyed ground truth comparison (strictly post-inference)
        is_surveyed = aid in surveyed_map
        surv_x, surv_y = surveyed_map.get(aid, (np.nan, np.nan))
        error_vs_surveyed = np.nan
        base_err_vs_surveyed = np.nan

        if is_surveyed:
            if c4_available:
                error_vs_surveyed = round(float(math.hypot(c4_x - surv_x, c4_y - surv_y)), 2)
            if pd.notna(base_x) and pd.notna(base_y):
                base_err_vs_surveyed = round(float(math.hypot(base_x - surv_x, base_y - surv_y)), 2)

        results.append({
            "target_address_id": aid,
            "address_id": aid,
            "town_id": tid,
            "locality_id_resolved": loc_id,
            "street_normalized": st_norm,
            "has_credible_doorstep": has_credible,
            "c4_available": c4_available,
            "c4_x": c4_x,
            "c4_y": c4_y,
            "neighbor_count": neighbor_count,
            "total_neighbor_weight": total_neighbor_w,
            "neighbor_address_ids": neighbor_ids_str,
            "ground_truth_medoid_x": round(med_x, 2),
            "ground_truth_medoid_y": round(med_y, 2),
            "error_vs_medoid_m": error_vs_medoid,
            "baseline_x": round(base_x, 2) if pd.notna(base_x) else np.nan,
            "baseline_y": round(base_y, 2) if pd.notna(base_y) else np.nan,
            "baseline_error_vs_medoid_m": base_err_vs_medoid,
            "is_surveyed_benchmark": is_surveyed,
            "surveyed_x": round(surv_x, 2) if pd.notna(surv_x) else np.nan,
            "surveyed_y": round(surv_y, 2) if pd.notna(surv_y) else np.nan,
            "error_vs_surveyed_m": error_vs_surveyed,
            "baseline_error_vs_surveyed_m": base_err_vs_surveyed,
            "evaluation_scope": "scv_leave_one_out",
            "target_history_masked": True,
            "target_medoid_masked": True,
            "target_gps_masked": True,
            "graph_fit_scope": "scv_leave_one_out",
            "reserved_for_calibration": True,
            "provenance_description": "leakage-safe SCV residuals reserved for empirical calibration",
        })

    df_scv = pd.DataFrame(results)
    return df_scv


def run_leave_one_street_out_validation(
    df_addresses: pd.DataFrame,
    df_medoids: pd.DataFrame,
    df_visits: Optional[pd.DataFrame] = None,
    df_localities: Optional[pd.DataFrame] = None,
    df_baseline: Optional[pd.DataFrame] = None,
    df_surveyed: Optional[pd.DataFrame] = None,
    min_visited_per_street: int = 5,
    fit_scope_prefix: str = "train_fold",
) -> pd.DataFrame:
    """Execute Leave-One-Street-Out (LOSO) spatial generalization validation.
    
    Evaluates spatial extrapolation when an entire street segment is unvisited:
    - Identifies qualified streets with >= min_visited_per_street DISTINCT visited accounts (account_id).
    - Simultaneously masks ALL visited accounts on the held-out street from the training graph
      (including all addresses, visits, and medoids belonging to those accounts across the entire portfolio).
    - Confirms that Candidate C4 is unavailable on the held-out street (0.0% availability).
    - Benchmarks fallback non-learned baselines (Locality Centroid C5 and Commercial Baseline C1).
    """
    df_addresses = df_addresses.copy()
    if "locality_id_resolved" not in df_addresses.columns and "locality_id" in df_addresses.columns:
        df_addresses = df_addresses.rename(columns={"locality_id": "locality_id_resolved"})

    # 1. Ensure account_id is populated in df_addresses
    if "account_id" not in df_addresses.columns:
        if df_visits is not None and "address_id" in df_visits.columns and "account_id" in df_visits.columns:
            addr_acc_map = (
                df_visits.dropna(subset=["address_id", "account_id"])
                .drop_duplicates(subset=["address_id"])
                .set_index("address_id")["account_id"]
                .to_dict()
            )
            df_addresses["account_id"] = df_addresses["address_id"].map(addr_acc_map)
        if "account_id" not in df_addresses.columns or df_addresses["account_id"].isna().any():
            try:
                from neurogeo.config import PATHS
                if PATHS.addresses_csv.exists():
                    df_raw_addr = pd.read_csv(PATHS.addresses_csv, usecols=["address_id", "account_id"])
                    raw_map = df_raw_addr.dropna().set_index("address_id")["account_id"].to_dict()
                    if "account_id" not in df_addresses.columns:
                        df_addresses["account_id"] = df_addresses["address_id"].map(raw_map)
                    else:
                        df_addresses["account_id"] = df_addresses["account_id"].fillna(
                            df_addresses["address_id"].map(raw_map)
                        )
            except Exception:
                pass
        if "account_id" not in df_addresses.columns:
            df_addresses["account_id"] = df_addresses["address_id"]
        else:
            df_addresses["account_id"] = df_addresses["account_id"].fillna(df_addresses["address_id"])

    # 2. Identify visited accounts (accounts with field visits or doorstep medoid evidence)
    visited_account_ids = set()
    if df_visits is not None and "account_id" in df_visits.columns:
        visited_account_ids.update(df_visits["account_id"].dropna().astype(str).unique())
    if df_medoids is not None and "address_id" in df_medoids.columns:
        med_addr_ids = set(df_medoids["address_id"].dropna().astype(str).unique())
        accs_with_med = (
            df_addresses[df_addresses["address_id"].astype(str).isin(med_addr_ids)]["account_id"]
            .dropna()
            .astype(str)
            .unique()
        )
        visited_account_ids.update(accs_with_med)

    # 3. Filter addresses with resolved town, locality, and street
    valid_mask = (
        df_addresses["town_id"].notna() & (df_addresses["town_id"] != "") &
        df_addresses["locality_id_resolved"].notna() & (df_addresses["locality_id_resolved"] != "") &
        df_addresses["street_normalized"].notna() & (df_addresses["street_normalized"] != "")
    )
    df_valid = df_addresses[valid_mask].copy()
    df_valid["street_key"] = df_valid.apply(
        lambda r: make_street_key(r["town_id"], r["locality_id_resolved"], r["street_normalized"]),
        axis=1
    )
    df_valid["account_id"] = df_valid["account_id"].astype(str)

    # 4. Identify qualified streets based on >= min_visited_per_street DISTINCT visited accounts
    df_valid_visited = df_valid[df_valid["account_id"].isin(visited_account_ids)]
    street_vis_acc_counts = df_valid_visited.groupby("street_key")["account_id"].nunique()
    eligible_streets = sorted(street_vis_acc_counts[street_vis_acc_counts >= min_visited_per_street].index.tolist())

    # Reference maps for evaluation
    loc_centroids: Dict[str, Tuple[float, float]] = {}
    if df_localities is not None and "locality_id" in df_localities.columns:
        for _, l_row in df_localities.iterrows():
            loc_centroids[str(l_row["locality_id"])] = (
                float(l_row["centroid_x"]),
                float(l_row["centroid_y"])
            )

    baseline_map: Dict[str, Tuple[float, float]] = {}
    if df_baseline is not None and "address_id" in df_baseline.columns:
        for _, b_row in df_baseline.iterrows():
            baseline_map[str(b_row["address_id"])] = (
                float(b_row["geocoder_x"]),
                float(b_row["geocoder_y"])
            )

    medoid_map: Dict[str, Tuple[float, float]] = {}
    if df_medoids is not None and "address_id" in df_medoids.columns:
        for _, m_row in df_medoids.iterrows():
            medoid_map[str(m_row["address_id"])] = (
                float(m_row["doorstep_medoid_x"]),
                float(m_row["doorstep_medoid_y"])
            )

    # 5. Run LOSO for each qualified street
    loso_records: List[Dict[str, Any]] = []

    for fold_idx, sk in enumerate(eligible_streets, start=1):
        fold_scope = f"{fit_scope_prefix}_{fold_idx}"

        # Identify ALL distinct visited account_ids on street S_j
        street_all_addrs = df_valid[df_valid["street_key"] == sk]
        held_out_accounts_on_street = set(
            street_all_addrs[street_all_addrs["account_id"].isin(visited_account_ids)]["account_id"].unique()
        )
        held_out_account_count = len(held_out_accounts_on_street)

        # Identify addresses belonging to those held-out accounts on street S_j
        held_out_eval_subset = street_all_addrs[street_all_addrs["account_id"].isin(held_out_accounts_on_street)]
        held_out_address_ids_on_street = set(held_out_eval_subset["address_id"].astype(str))
        held_out_address_count = len(held_out_address_ids_on_street)

        # Identify ALL addresses belonging to those held-out accounts across the entire portfolio
        all_held_out_account_addrs = set(
            df_addresses[df_addresses["account_id"].astype(str).isin(held_out_accounts_on_street)]["address_id"].astype(str)
        )

        # Identify all addresses on street S_j
        all_street_address_ids = set(street_all_addrs["address_id"].astype(str))

        # Combined purge set: all addresses of held-out accounts + all addresses on held-out street
        masked_address_ids = all_held_out_account_addrs | all_street_address_ids

        # Build fold-specific training datasets
        train_addresses = df_addresses[
            ~df_addresses["account_id"].astype(str).isin(held_out_accounts_on_street) &
            ~df_addresses["address_id"].astype(str).isin(masked_address_ids)
        ]
        train_medoids = df_medoids[~df_medoids["address_id"].astype(str).isin(masked_address_ids)]
        
        if df_visits is not None:
            vis_acc_mask = (
                df_visits["account_id"].astype(str).isin(held_out_accounts_on_street)
                if "account_id" in df_visits.columns
                else False
            )
            vis_addr_mask = df_visits["address_id"].astype(str).isin(masked_address_ids)
            train_visits = df_visits[~(vis_acc_mask | vis_addr_mask)]
        else:
            train_visits = None

        # Build fold-specific training graph
        train_graph = build_hierarchical_street_graph(
            train_addresses, train_medoids, df_visits=train_visits, fit_scope=fold_scope
        )

        # Zero-Leakage Assertion: Street S must have 0 neighbor accounts and no learned centroid in training graph
        if train_graph.has_node(sk):
            assert train_graph.nodes[sk].get("neighbor_count", 0) == 0, (
                f"LOSO LEAKAGE: Street {sk} retains neighbor accounts in training graph for fold {fold_scope}."
            )
            assert train_graph.nodes[sk].get("centroid_x") is None, (
                f"LOSO LEAKAGE: Street {sk} retains centroid_x in training graph for fold {fold_scope}."
            )
            assert train_graph.nodes[sk].get("centroid_y") is None, (
                f"LOSO LEAKAGE: Street {sk} retains centroid_y in training graph for fold {fold_scope}."
            )

        # Invariant: No masked address may appear in ANY neighbor accounts in ANY street node
        for node_id, node_data in train_graph.nodes(data=True):
            if node_data.get("node_type") == "street":
                for nbr in node_data.get("neighbor_accounts", []):
                    assert str(nbr["address_id"]) not in masked_address_ids, (
                        f"LOSO LEAKAGE: Masked address {nbr['address_id']} found in training graph node {node_id}!"
                    )

        # Evaluate addresses on held-out street
        c4_available_count = 0
        medoid_episodes_count = 0
        loc_errors: List[float] = []
        base_errors: List[float] = []

        first_row = held_out_eval_subset.iloc[0]
        tid = str(first_row["town_id"])
        loc_id = str(first_row["locality_id_resolved"])
        st_norm = str(first_row["street_normalized"])
        loc_x, loc_y = loc_centroids.get(loc_id, (np.nan, np.nan))

        for _, row in held_out_eval_subset.iterrows():
            aid = str(row["address_id"])

            # Attempt C4 in training graph
            c4_res = compute_scoped_neighbor_centroid(
                address_id=aid,
                town_id=tid,
                locality_id=loc_id,
                street_normalized=st_norm,
                graph=train_graph,
                fit_scope=fold_scope,
            )
            if c4_res is not None:
                c4_available_count += 1

            # Fallback error against true doorstep medoid (if available)
            if aid in medoid_map:
                medoid_episodes_count += 1
                mx, my = medoid_map[aid]

                # Fallback 1: Locality Centroid (C5) error
                if pd.notna(loc_x) and pd.notna(loc_y):
                    loc_errors.append(float(math.hypot(loc_x - mx, loc_y - my)))

                # Fallback 2: Commercial Baseline (C1) error
                bx, by = baseline_map.get(aid, (np.nan, np.nan))
                if pd.notna(bx) and pd.notna(by):
                    base_errors.append(float(math.hypot(bx - mx, by - my)))

        loso_records.append({
            "street_key": sk,
            "fold_id": fold_idx,
            "fit_scope": fold_scope,
            "town_id": tid,
            "locality_id": loc_id,
            "street_normalized": st_norm,
            "held_out_account_count": int(held_out_account_count),
            "held_out_address_count": int(held_out_address_count),
            "evaluated_episode_count": int(len(held_out_eval_subset)),
            "evaluated_medoid_episode_count": int(medoid_episodes_count),
            "visited_accounts_count": int(held_out_account_count),  # Backwards compatibility
            "held_out_account_ids": ",".join(sorted(held_out_accounts_on_street)),
            "held_out_address_ids": ",".join(sorted(held_out_address_ids_on_street)),
            "c4_available_rate": round(float(c4_available_count / len(held_out_eval_subset)), 4) if len(held_out_eval_subset) > 0 else 0.0,
            "locality_centroid_x": round(loc_x, 2) if pd.notna(loc_x) else np.nan,
            "locality_centroid_y": round(loc_y, 2) if pd.notna(loc_y) else np.nan,
            "median_error_vs_locality_m": round(float(np.median(loc_errors)), 2) if loc_errors else np.nan,
            "p90_error_vs_locality_m": round(float(np.quantile(loc_errors, 0.90)), 2) if loc_errors else np.nan,
            "median_error_vs_baseline_m": round(float(np.median(base_errors)), 2) if base_errors else np.nan,
            "p90_error_vs_baseline_m": round(float(np.quantile(base_errors, 0.90)), 2) if base_errors else np.nan,
        })

    df_loso = pd.DataFrame(loso_records)
    return df_loso

