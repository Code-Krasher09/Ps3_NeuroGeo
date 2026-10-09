"""HISTORICAL REPORT-ONLY ARTIFACT: Stage 2 (C6) Validation Evaluator (Archived).

Status: ARCHIVED / NOT EXECUTABLE AS ORIGINALLY DESIGNED.

Authority & Historical Context:
This script was implemented during Stage 2 to conduct an isolated, leakage-safe validation
evaluation comparing:
  - Condition 0: Baseline frozen predictions from reports/metrics/phase4_predictions.csv
  - Condition 1: Stage 1 parser active, candidates C1-C5 only
  - Condition 2: Stage 1 parser active + Candidate C6 (town-scoped landmark hypothesis)

Key Finding & Final Architectural Decision:
Across the 19 validation split addresses, Candidate C6 produced exactly 0 wins, 0 losses,
and 0 candidate selections. Because C6 introduced substantial resolver and scoring complexity
without providing incremental spatial accuracy over C3/C4, it was formally rejected and
completely excised from the production architecture (src/neurogeo).

Reason No Longer Executable:
Because Candidate C6 resolution (resolve_town_landmark_hypothesis) and C6 candidate generation
were cleanly removed from src/neurogeo in the final submission consolidation, this script is
no longer executable as originally designed. It must not be presented as a functioning C1-C5
evaluator or active comparison tool.

Authoritative Historical Artifacts:
The original evaluation outputs produced by this harness remain permanently preserved under
reports/eval_val_c6/:
  - reports/eval_val_c6/validation_predictions.csv
  - reports/eval_val_c6/validation_metrics_summary.json
  - reports/eval_val_c6/validation_paired_comparison.csv
  - reports/eval_val_c6/validation_candidate_diagnostics.csv
  - reports/eval_val_c6/evaluation_manifest.json

Do NOT execute this script in production. For all active production inference and evaluation,
use scripts/run_phase4.py and notebooks/01_master_pipeline.ipynb.
"""

from dataclasses import replace
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from neurogeo.config import PATHS, DEFAULT_CONFIG, PathConfig
from neurogeo.schemas import SpatialCandidate, PredictionStatus
from neurogeo.address_parser import parse_address
from neurogeo.candidate_ranker import (
    generate_candidates,
    extract_candidate_features,
    rank_candidates,
    compute_euclidean_distance,
)


# Note: Candidate C6 resolver logic was excised from src/neurogeo during consolidation.
from neurogeo.calibrator import assign_status_tier, calibrate_prediction
from neurogeo.spatial_graph import build_hierarchical_street_graph, compute_scoped_neighbor_centroid


def validate_output_dir(output_dir: Union[str, Path], paths: Optional[PathConfig] = None) -> Path:
    """Validate that the requested output directory is isolated and not protected."""
    p = paths or PATHS
    target = Path(output_dir).resolve()
    repo_root = p.repo_root.resolve()
    metrics_dir = p.metrics_dir.resolve()
    figures_dir = p.figures_dir.resolve()
    data_dir = p.data_dir.resolve()

    if target == repo_root:
        raise ValueError(f"Protected path violation: output directory cannot be repository root '{repo_root}'")
    if target == metrics_dir or metrics_dir in target.parents:
        raise ValueError(f"Protected path violation: output directory cannot be or be inside '{metrics_dir}'")
    if target == figures_dir or figures_dir in target.parents:
        raise ValueError(f"Protected path violation: output directory cannot be or be inside '{figures_dir}'")
    if target == data_dir or data_dir in target.parents:
        raise ValueError(f"Protected path violation: output directory cannot be or be inside '{data_dir}'")

    return target


def extract_validation_split_data(
    paths: Optional[PathConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], Set[str], Set[str]]:
    """Extract and validate the canonical 19-address validation split.

    Returns:
        (val_df, ground_truth_map, train_ids, test_ids)
    """
    p = paths or PATHS
    df_surv = pd.read_csv(p.surveyed_addresses_csv)
    df_raw_addr = pd.read_csv(p.addresses_csv)
    df_splits = pd.read_csv(p.splits_csv)

    # Merge to associate split with benchmark addresses
    merged = df_surv.merge(
        df_raw_addr[["address_id", "account_id", "town_id", "address_text"]],
        on="address_id",
        how="inner",
    )
    if len(merged) != len(df_surv):
        raise ValueError(f"Integrity error: {len(df_surv)} surveyed != {len(merged)} joined")

    merged = merged.merge(
        df_splits[["account_id", "split"]],
        on="account_id",
        how="inner",
    )
    if len(merged) != len(df_surv):
        raise ValueError(f"Split error: {len(df_surv)} surveyed != {len(merged)} with splits")

    train_df = merged[merged["split"] == "train"]
    val_df = merged[merged["split"] == "validation"].copy()
    test_df = merged[merged["split"] == "test"]

    train_ids = set(train_df["address_id"])
    val_ids = set(val_df["address_id"])
    test_ids = set(test_df["address_id"])

    # Strict invariant assertions
    if len(val_df) != 19:
        raise ValueError(f"Expected exactly 19 validation addresses, found {len(val_df)}")
    if val_df["address_id"].nunique() != 19:
        raise ValueError(f"Duplicate validation address IDs detected: {len(val_df)} rows, {val_df['address_id'].nunique()} unique")
    if len(val_ids & train_ids) > 0:
        raise ValueError(f"Leakage violation: {len(val_ids & train_ids)} addresses overlap between train and val")
    if len(val_ids & test_ids) > 0:
        raise ValueError(f"Leakage violation: {len(val_ids & test_ids)} addresses overlap between test and val")

    # Isolate ground truth surveyed coordinates into a separate dictionary
    ground_truth_map = {
        row["address_id"]: (float(row["surveyed_x"]), float(row["surveyed_y"]))
        for _, row in val_df.iterrows()
    }

    # Drop surveyed coordinates from the working dataframe to guarantee zero accidental leakage
    val_df_clean = val_df[["address_id", "account_id", "town_id", "address_text", "split"]].copy()

    return val_df_clean, ground_truth_map, train_ids, test_ids


def compute_subgroup_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute standard descriptive summary metrics for a list of evaluation records.

    Safely handles zero-sized subgroups and undefined metrics without fabricating values.
    """
    n = len(records)
    if n == 0:
        return {
            "sample_count": 0,
            "median_error_m": None,
            "mean_error_m": None,
            "p90_error_m": None,
            "pct_within_100m": {"pct": None, "numerator": 0, "denominator": 0},
            "pct_within_500m": {"pct": None, "numerator": 0, "denominator": 0},
            "empirical_coverage": {"pct": None, "numerator": 0, "denominator": 0},
            "status_tier_counts": {},
            "candidate_selection_counts": {},
        }

    errors = np.array([r["error_vs_surveyed_m"] for r in records if r["error_vs_surveyed_m"] is not None], dtype=float)
    covered = np.array([1 if r.get("is_covered") else 0 for r in records], dtype=int)

    within_100 = int(np.sum(errors <= 100.0))
    within_500 = int(np.sum(errors <= 500.0))
    covered_cnt = int(np.sum(covered))

    tier_counts: Dict[str, int] = {}
    for r in records:
        t = str(r.get("status_tier", "UNKNOWN"))
        tier_counts[t] = tier_counts.get(t, 0) + 1

    cand_counts: Dict[str, int] = {}
    for r in records:
        c = str(r.get("selected_candidate_type", "UNKNOWN"))
        cand_counts[c] = cand_counts.get(c, 0) + 1

    return {
        "sample_count": n,
        "median_error_m": round(float(np.median(errors)), 2) if len(errors) > 0 else None,
        "mean_error_m": round(float(np.mean(errors)), 2) if len(errors) > 0 else None,
        "p90_error_m": round(float(np.percentile(errors, 90)), 2) if len(errors) > 0 else None,
        "pct_within_100m": {
            "pct": round(float(within_100 / n * 100.0), 2),
            "numerator": within_100,
            "denominator": n,
        },
        "pct_within_500m": {
            "pct": round(float(within_500 / n * 100.0), 2),
            "numerator": within_500,
            "denominator": n,
        },
        "empirical_coverage": {
            "pct": round(float(covered_cnt / n * 100.0), 2),
            "numerator": covered_cnt,
            "denominator": n,
        },
        "status_tier_counts": tier_counts,
        "candidate_selection_counts": cand_counts,
    }


def compute_paired_comparison_summary(
    base_records: Dict[str, Dict[str, Any]],
    eval_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute paired address-level differences (Eval - Base)."""
    deltas = []
    wins = 0    # Eval error is smaller by > 1.0 m
    losses = 0  # Eval error is larger by > 1.0 m
    ties = 0    # |Eval - Base| <= 1.0 m

    common_ids = sorted(list(set(base_records.keys()) & set(eval_records.keys())))
    n = len(common_ids)
    if n == 0:
        return {
            "sample_count": 0,
            "wins": {"count": 0, "pct": None, "denominator": 0},
            "losses": {"count": 0, "pct": None, "denominator": 0},
            "ties": {"count": 0, "pct": None, "denominator": 0},
            "mean_delta_m": None,
            "median_delta_m": None,
        }

    for aid in common_ids:
        b_err = base_records[aid]["error_vs_surveyed_m"]
        e_err = eval_records[aid]["error_vs_surveyed_m"]
        if b_err is not None and e_err is not None:
            diff = round(float(e_err - b_err), 2)
            deltas.append(diff)
            if diff < -1.0:
                wins += 1
            elif diff > 1.0:
                losses += 1
            else:
                ties += 1

    d_arr = np.array(deltas, dtype=float)
    return {
        "sample_count": n,
        "wins": {"count": wins, "pct": round(float(wins / n * 100.0), 2), "denominator": n},
        "losses": {"count": losses, "pct": round(float(losses / n * 100.0), 2), "denominator": n},
        "ties": {"count": ties, "pct": round(float(ties / n * 100.0), 2), "denominator": n},
        "mean_delta_m": round(float(np.mean(d_arr)), 2) if len(d_arr) > 0 else None,
        "median_delta_m": round(float(np.median(d_arr)), 2) if len(d_arr) > 0 else None,
    }


def run_validation_evaluation(
    output_dir: Optional[Union[str, Path]] = None,
    paths: Optional[PathConfig] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Archived historical validation evaluator.

    This function is preserved for historical audit reference only and is no longer
    executable because Candidate C6 has been excised from the production architecture.
    """
    raise RuntimeError(
        "scripts/evaluate_validation.py is an archived historical report-only artifact. "
        "It is no longer executable as originally designed because Candidate C6 was rejected "
        "and removed from src/neurogeo during submission consolidation. "
        "Authoritative historical evaluation results remain preserved in reports/eval_val_c6/."
    )
    """Execute the three-condition validation evaluation cleanly and deterministically.

    Conditions:
    - Condition 0: Frozen baseline from reports/metrics/phase4_predictions.csv
    - Condition 1: Stage 1 active, candidates C1-C5 only (C6 disabled)
    - Condition 2: Stage 1 active, candidates C1-C6 enabled
    """
    p = paths or PATHS

    # 1. Output directory safety verification
    out_dir_path: Optional[Path] = None
    if not dry_run:
        target_dir = output_dir or (p.reports_dir / "eval_val_c6")
        out_dir_path = validate_output_dir(target_dir, paths=p)
        out_dir_path.mkdir(parents=True, exist_ok=True)

    # 2. Extract and assert the canonical 19 validation addresses
    val_df_clean, ground_truth_map, train_ids, test_ids = extract_validation_split_data(paths=p)
    val_ids = set(val_df_clean["address_id"])

    # 3. Ingest reference tables and frozen model parameters (READ-ONLY)
    df_raw_addr = pd.read_csv(p.addresses_csv)
    df_localities = pd.read_csv(p.localities_csv)
    df_pois = pd.read_csv(p.landmarks_poi_csv)
    df_base = pd.read_csv(p.baseline_geocodes_csv)
    df_medoids = pd.read_parquet(p.metrics_dir / "address_doorstep_medoids.parquet")
    df_vis = pd.read_parquet(p.metrics_dir / "visit_features.parquet")
    df_addr_ent = pd.read_parquet(p.address_entities_parquet)

    # Pre-index fast lookups
    baseline_map = df_base.set_index("address_id").to_dict("index")
    medoid_map = df_medoids.set_index("address_id").to_dict("index")
    poi_map = df_pois.set_index("poi_id").to_dict("index")
    loc_map = df_localities.set_index("locality_id").to_dict("index")
    visits_by_addr = df_vis.groupby("address_id").apply(lambda g: g.to_dict("records"), include_groups=False).to_dict()

    # Build hierarchical street graph under production_all fit scope
    df_addrs_parsed = df_raw_addr.merge(
        df_addr_ent[["address_id", "locality_id_resolved", "street_normalized"]],
        on="address_id",
        how="left",
    )
    graph = build_hierarchical_street_graph(
        df_addresses=df_addrs_parsed,
        df_medoids=df_medoids,
        df_visits=df_vis,
        fit_scope="production_all",
    )

    # Load frozen reranker weights
    with open(p.reranker_weights_json, "r", encoding="utf-8") as f:
        weights_data = json.load(f)
    fitted_weights = weights_data["weights"]

    # Load frozen calibration quantiles
    with open(p.calibration_quantiles_json, "r", encoding="utf-8") as f:
        cal_summary = json.load(f)

    # Load frozen baseline predictions
    df_base_preds_all = pd.read_csv(p.phase4_predictions_csv)
    df_base_val = df_base_preds_all[df_base_preds_all["address_id"].isin(val_ids)].copy()
    if len(df_base_val) != 19:
        raise ValueError(f"Expected 19 baseline validation predictions, found {len(df_base_val)}")

    # Map visited status for validation addresses
    visited_status_map = {
        aid: bool(aid in visits_by_addr and len(visits_by_addr[aid]) > 0)
        for aid in val_ids
    }

    # =========================================================================
    # Condition 0: Frozen Baseline Predictions
    # =========================================================================
    cond0_records: List[Dict[str, Any]] = []
    for _, row in df_base_val.iterrows():
        aid = row["address_id"]
        sx, sy = ground_truth_map[aid]
        pred_x, pred_y = float(row["predicted_x"]), float(row["predicted_y"])
        err_m = compute_euclidean_distance(pred_x, pred_y, sx, sy)
        rad_m = float(row["confidence_radius_m"])
        is_cov = bool(err_m <= rad_m)

        cond0_records.append({
            "address_id": aid,
            "condition": "Condition 0: Frozen Baseline",
            "condition_code": "cond0",
            "split": "validation",
            "is_visited": visited_status_map[aid],
            "predicted_x": round(pred_x, 2),
            "predicted_y": round(pred_y, 2),
            "selected_candidate_type": str(row["selected_candidate_type"]),
            "status_tier": str(row["status_tier"]),
            "confidence_radius_m": rad_m,
            "error_vs_surveyed_m": round(err_m, 2),
            "is_covered": is_cov,
            "selected_score": float(row.get("selected_score", 0.0)),
            "runner_up_score": float(row.get("runner_up_score", 0.0)),
            "score_margin": float(row.get("score_margin", 0.0)),
            "evidence_sources": str(row.get("evidence_sources", "")),
            "available_candidate_count": int(row.get("candidate_count", 0)),
            "available_candidate_types": str(row.get("evidence_sources", "")),
            "c6_available": False,
            "c6_reason": "not_in_baseline",
        })

    # =========================================================================
    # Recomputing Stage 1 Parsed Entities from Raw Address Text
    # =========================================================================
    recomputed_entities: Dict[str, Any] = {}
    for _, row in val_df_clean.iterrows():
        aid = row["address_id"]
        parsed = parse_address(
            address_id=aid,
            address_text=row["address_text"],
            town_id=row["town_id"],
            df_localities=df_localities,
            df_pois=df_pois,
        )
        recomputed_entities[aid] = {
            "address_id": aid,
            "address_text": row["address_text"],
            "town_id": row["town_id"],
            "pincode_resolved": parsed.pincode_resolved,
            "pincode_status": parsed.pincode_status,
            "locality_id_resolved": parsed.locality_id_resolved,
            "locality_name_raw": parsed.locality_name_raw,
            "street_normalized": parsed.street_normalized,
            "street_raw": parsed.street_raw,
            "landmark_poi_id": parsed.landmark_poi_id,
            "landmark_raw": parsed.landmark_raw,
            "spatial_relation": parsed.spatial_relation,
            "house_raw": parsed.house_raw,
            "locality_resolution_status": parsed.locality_resolution_status,
            "poi_resolution_status": parsed.poi_resolution_status,
        }

    # =========================================================================
    # Evaluation Loops for Condition 1 and Condition 2
    # =========================================================================
    cond1_records: List[Dict[str, Any]] = []
    cond2_records: List[Dict[str, Any]] = []

    diagnostics_rows: List[Dict[str, Any]] = []

    for _, row in val_df_clean.iterrows():
        aid = row["address_id"]
        sx, sy = ground_truth_map[aid]
        is_visited = visited_status_map[aid]

        b_row = baseline_map.get(aid)
        m_row = medoid_map.get(aid)
        v_list = visits_by_addr.get(aid, [])
        b_coord = (float(b_row["geocoder_x"]), float(b_row["geocoder_y"])) if b_row and pd.notna(b_row.get("geocoder_x")) else None

        parsed_dict = recomputed_entities[aid]
        p_id = parsed_dict.get("landmark_poi_id")
        p_row = poi_map.get(p_id) if pd.notna(p_id) else None
        l_id = parsed_dict.get("locality_id_resolved")
        l_row = loc_map.get(l_id) if pd.notna(l_id) else None

        # Hierarchical street graph neighbor centroid (C4)
        c4_res = compute_scoped_neighbor_centroid(
            address_id=aid,
            town_id=parsed_dict.get("town_id"),
            locality_id=parsed_dict.get("locality_id_resolved"),
            street_normalized=parsed_dict.get("street_normalized"),
            graph=graph,
            fit_scope="production_all",
        )

        # C6 resolver invocation
        c6_res = resolve_c6_candidate_for_row(
            row=parsed_dict,
            df_pois=df_pois,
            df_localities=df_localities,
        )

        # -------------------------------------------------------------
        # Condition 1: Stage 1 active, C1-C5 only (C6 explicitly disabled)
        # -------------------------------------------------------------
        cands_c1 = generate_candidates(
            address_id=aid,
            baseline_row=b_row,
            medoid_row=m_row,
            poi_row=p_row,
            c4_result=c4_res,
            c6_result=None,  # Explicitly disabled
            locality_row=l_row,
            parsed_entity=parsed_dict,
            fit_scope="production_all",
        )

        feated_c1 = []
        for c in cands_c1:
            if c.available and c.x is not None and c.y is not None:
                f = extract_candidate_features(
                    candidate=c,
                    parsed_entity=parsed_dict,
                    direct_visits=v_list,
                    baseline_coord=b_coord,
                )
                feated_c1.append(replace(c, features=f))

        ranked_c1 = rank_candidates(feated_c1, weights=fitted_weights, deduplicate=True)
        assert len(ranked_c1) > 0, f"No candidates for {aid} in Condition 1"
        sel_c1 = ranked_c1[0]
        rup_c1 = ranked_c1[1] if len(ranked_c1) > 1 else None
        margin_c1 = sel_c1.total_score - (rup_c1.total_score if rup_c1 else 0.0)

        has_cred = bool(m_row.get("has_credible_doorstep", False)) if m_row else False
        n_cnt = int(c4_res.get("neighbor_count", 0)) if c4_res else 0
        is_poi_res = bool(p_row is not None)
        b_prec = str(b_row.get("precision", "unknown")) if b_row else "unknown"

        tier_c1 = assign_status_tier(
            selected_candidate_type=sel_c1.candidate_type,
            has_credible_doorstep=has_cred,
            neighbor_count=n_cnt,
            is_poi_resolved=is_poi_res,
            baseline_precision=b_prec,
        )
        rad_c1, conf_c1, fb_c1, _ = calibrate_prediction(tier_c1, cal_summary)
        err_c1 = compute_euclidean_distance(sel_c1.x, sel_c1.y, sx, sy)
        cov_c1 = bool(err_c1 <= rad_c1)

        cond1_records.append({
            "address_id": aid,
            "condition": "Condition 1: Stage 1 Active, C1-C5 (C6 Disabled)",
            "condition_code": "cond1",
            "split": "validation",
            "is_visited": is_visited,
            "predicted_x": round(sel_c1.x, 2),
            "predicted_y": round(sel_c1.y, 2),
            "selected_candidate_type": sel_c1.candidate_type,
            "status_tier": tier_c1.value,
            "confidence_radius_m": rad_c1,
            "error_vs_surveyed_m": round(err_c1, 2),
            "is_covered": cov_c1,
            "selected_score": round(sel_c1.total_score, 4),
            "runner_up_score": round(rup_c1.total_score if rup_c1 else 0.0, 4),
            "score_margin": round(margin_c1, 4),
            "evidence_sources": ",".join(sel_c1.agreed_sources),
            "available_candidate_count": len([c for c in cands_c1 if c.available]),
            "available_candidate_types": ",".join([c.candidate_type for c in cands_c1 if c.available]),
            "c6_available": False,
            "c6_reason": "condition1_disabled",
        })

        # -------------------------------------------------------------
        # Condition 2: Stage 1 active, C1-C6 enabled
        # -------------------------------------------------------------
        cands_c2 = generate_candidates(
            address_id=aid,
            baseline_row=b_row,
            medoid_row=m_row,
            poi_row=p_row,
            c4_result=c4_res,
            c6_result=c6_res,  # Enabled
            locality_row=l_row,
            parsed_entity=parsed_dict,
            fit_scope="production_all",
        )

        feated_c2 = []
        for c in cands_c2:
            if c.available and c.x is not None and c.y is not None:
                f = extract_candidate_features(
                    candidate=c,
                    parsed_entity=parsed_dict,
                    direct_visits=v_list,
                    baseline_coord=b_coord,
                )
                feated_c2.append(replace(c, features=f))

        ranked_c2 = rank_candidates(feated_c2, weights=fitted_weights, deduplicate=True)
        assert len(ranked_c2) > 0, f"No candidates for {aid} in Condition 2"
        sel_c2 = ranked_c2[0]
        rup_c2 = ranked_c2[1] if len(ranked_c2) > 1 else None
        margin_c2 = sel_c2.total_score - (rup_c2.total_score if rup_c2 else 0.0)

        tier_c2 = assign_status_tier(
            selected_candidate_type=sel_c2.candidate_type,
            has_credible_doorstep=has_cred,
            neighbor_count=n_cnt,
            is_poi_resolved=is_poi_res,
            baseline_precision=b_prec,
        )
        rad_c2, conf_c2, fb_c2, _ = calibrate_prediction(tier_c2, cal_summary)
        err_c2 = compute_euclidean_distance(sel_c2.x, sel_c2.y, sx, sy)
        cov_c2 = bool(err_c2 <= rad_c2)

        c6_cand_obj = next((c for c in cands_c2 if c.candidate_type == "C6"), None)
        c6_avail = bool(c6_cand_obj.available) if c6_cand_obj else False
        c6_reason = str(c6_res.get("reason", "resolved" if c6_avail else "unknown"))

        cond2_records.append({
            "address_id": aid,
            "condition": "Condition 2: Stage 1 Active, C1-C6 Enabled",
            "condition_code": "cond2",
            "split": "validation",
            "is_visited": is_visited,
            "predicted_x": round(sel_c2.x, 2),
            "predicted_y": round(sel_c2.y, 2),
            "selected_candidate_type": sel_c2.candidate_type,
            "status_tier": tier_c2.value,
            "confidence_radius_m": rad_c2,
            "error_vs_surveyed_m": round(err_c2, 2),
            "is_covered": cov_c2,
            "selected_score": round(sel_c2.total_score, 4),
            "runner_up_score": round(rup_c2.total_score if rup_c2 else 0.0, 4),
            "score_margin": round(margin_c2, 4),
            "evidence_sources": ",".join(sel_c2.agreed_sources),
            "available_candidate_count": len([c for c in cands_c2 if c.available]),
            "available_candidate_types": ",".join([c.candidate_type for c in cands_c2 if c.available]),
            "c6_available": c6_avail,
            "c6_reason": c6_reason,
        })

        # Diagnostics row for C1-C6 candidate tracking
        for c in cands_c2:
            diagnostics_rows.append({
                "address_id": aid,
                "is_visited": is_visited,
                "candidate_type": c.candidate_type,
                "available": c.available,
                "selected_cond1": bool(sel_c1.candidate_type == c.candidate_type and c.candidate_type != "C6"),
                "selected_cond2": bool(sel_c2.candidate_type == c.candidate_type),
                "c6_reason": c6_reason if c.candidate_type == "C6" else None,
            })

    # =========================================================================
    # Aggregating Metrics Summary (Stratified by Visited vs Cold-Start)
    # =========================================================================
    cond0_dict = {r["address_id"]: r for r in cond0_records}
    cond1_dict = {r["address_id"]: r for r in cond1_records}
    cond2_dict = {r["address_id"]: r for r in cond2_records}

    metrics_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "validation_split_only_n19",
        "sample_size": 19,
        "visited_count": sum(1 for v in visited_status_map.values() if v),
        "cold_start_count": sum(1 for v in visited_status_map.values() if not v),
        "conditions": {
            "condition_0_baseline": {
                "overall": compute_subgroup_metrics(cond0_records),
                "visited": compute_subgroup_metrics([r for r in cond0_records if r["is_visited"]]),
                "cold_start": compute_subgroup_metrics([r for r in cond0_records if not r["is_visited"]]),
            },
            "condition_1_stage1_only": {
                "overall": compute_subgroup_metrics(cond1_records),
                "visited": compute_subgroup_metrics([r for r in cond1_records if r["is_visited"]]),
                "cold_start": compute_subgroup_metrics([r for r in cond1_records if not r["is_visited"]]),
            },
            "condition_2_stage1_plus_c6": {
                "overall": compute_subgroup_metrics(cond2_records),
                "visited": compute_subgroup_metrics([r for r in cond2_records if r["is_visited"]]),
                "cold_start": compute_subgroup_metrics([r for r in cond2_records if not r["is_visited"]]),
            },
        },
        "paired_comparisons": {
            "condition_1_vs_condition_0": compute_paired_comparison_summary(cond0_dict, cond1_dict),
            "condition_2_vs_condition_1": compute_paired_comparison_summary(cond1_dict, cond2_dict),
            "condition_2_vs_condition_0": compute_paired_comparison_summary(cond0_dict, cond2_dict),
        },
    }

    # =========================================================================
    # Constructing DataFrames for Tabular Export
    # =========================================================================
    all_pred_records = cond0_records + cond1_records + cond2_records
    df_all_preds = pd.DataFrame(all_pred_records)

    paired_rows: List[Dict[str, Any]] = []
    for aid in sorted(list(val_ids)):
        r0 = cond0_dict[aid]
        r1 = cond1_dict[aid]
        r2 = cond2_dict[aid]

        d1_0 = round(float(r1["error_vs_surveyed_m"] - r0["error_vs_surveyed_m"]), 2)
        d2_1 = round(float(r2["error_vs_surveyed_m"] - r1["error_vs_surveyed_m"]), 2)
        d2_0 = round(float(r2["error_vs_surveyed_m"] - r0["error_vs_surveyed_m"]), 2)

        def _outcome(delta: float) -> str:
            if delta < -1.0:
                return "win"
            if delta > 1.0:
                return "loss"
            return "tie"

        paired_rows.append({
            "address_id": aid,
            "is_visited": visited_status_map[aid],
            "cond0_selected": r0["selected_candidate_type"],
            "cond0_error_m": r0["error_vs_surveyed_m"],
            "cond1_selected": r1["selected_candidate_type"],
            "cond1_error_m": r1["error_vs_surveyed_m"],
            "cond2_selected": r2["selected_candidate_type"],
            "cond2_error_m": r2["error_vs_surveyed_m"],
            "delta_cond1_minus_cond0_m": d1_0,
            "delta_cond2_minus_cond1_m": d2_1,
            "delta_cond2_minus_cond0_m": d2_0,
            "cond1_vs_cond0_outcome": _outcome(d1_0),
            "cond2_vs_cond1_outcome": _outcome(d2_1),
            "cond2_vs_cond0_outcome": _outcome(d2_0),
        })
    df_paired = pd.DataFrame(paired_rows)

    df_diag = pd.DataFrame(diagnostics_rows)
    diag_summary = (
        df_diag.groupby(["candidate_type"])
        .agg(
            available_count=("available", "sum"),
            selected_cond1_count=("selected_cond1", "sum"),
            selected_cond2_count=("selected_cond2", "sum"),
        )
        .reset_index()
    )
    diag_summary["total_addresses"] = 19
    diag_summary["available_pct"] = (diag_summary["available_count"] / 19 * 100.0).round(2)

    # Manifest metadata
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "scripts/evaluate_validation.py",
        "evaluation_scope": "validation_split_only",
        "sample_count": 19,
        "splits_guarantee": {
            "validation_address_ids": sorted(list(val_ids)),
            "train_address_count": len(train_ids),
            "train_re_evaluated": False,
            "test_address_count": len(test_ids),
            "test_split_touched": False,
            "test_split_evaluated": False,
        },
        "inputs_loaded": {
            "addresses_csv": str(p.addresses_csv.resolve()),
            "surveyed_addresses_csv": str(p.surveyed_addresses_csv.resolve()),
            "splits_csv": str(p.splits_csv.resolve()),
            "localities_csv": str(p.localities_csv.resolve()),
            "landmarks_poi_csv": str(p.landmarks_poi_csv.resolve()),
            "baseline_geocodes_csv": str(p.baseline_geocodes_csv.resolve()),
            "address_doorstep_medoids_parquet": str((p.metrics_dir / "address_doorstep_medoids.parquet").resolve()),
            "visit_features_parquet": str((p.metrics_dir / "visit_features.parquet").resolve()),
            "reranker_weights_json": str(p.reranker_weights_json.resolve()),
            "calibration_quantiles_json": str(p.calibration_quantiles_json.resolve()),
            "phase4_predictions_csv": str(p.phase4_predictions_csv.resolve()),
        },
        "leakage_guarantees": [
            "Test split addresses were never loaded for evaluation, scoring, or metric computation.",
            "Reranker weights and calibration quantiles were frozen from baseline and not refitted.",
            "Surveyed ground truth coordinates were isolated from candidate generation and feature extraction.",
            "Target address field visits were never consumed by candidate C6.",
        ],
        "output_directory": str(out_dir_path.resolve()) if out_dir_path else "dry_run",
        "output_files": [
            "validation_predictions.csv",
            "validation_metrics_summary.json",
            "validation_paired_comparison.csv",
            "validation_candidate_diagnostics.csv",
            "evaluation_manifest.json",
        ] if out_dir_path else [],
    }

    # =========================================================================
    # Write Artifacts Exclusively to Isolated Directory
    # =========================================================================
    if not dry_run and out_dir_path is not None:
        p_preds = out_dir_path / "validation_predictions.csv"
        df_all_preds.to_csv(p_preds, index=False)

        p_summary = out_dir_path / "validation_metrics_summary.json"
        with open(p_summary, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        p_paired = out_dir_path / "validation_paired_comparison.csv"
        df_paired.to_csv(p_paired, index=False)

        p_diag = out_dir_path / "validation_candidate_diagnostics.csv"
        diag_summary.to_csv(p_diag, index=False)

        p_man = out_dir_path / "evaluation_manifest.json"
        with open(p_man, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return {
        "metrics_summary": metrics_summary,
        "manifest": manifest,
        "paired_comparison_count": len(df_paired),
        "total_prediction_records": len(df_all_preds),
        "output_directory": str(out_dir_path) if out_dir_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-Safe Validation Evaluator for NeuroGeo-Field")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PATHS.reports_dir / "eval_val_c6"),
        help="Isolated output directory to write evaluation artifacts (default: reports/eval_val_c6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute evaluation in-memory without writing any files to disk",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("NeuroGeo-Field: Isolated Validation Evaluator (Stage 1 + Stage 2)")
    print("=" * 75)
    print(f"Target Output Directory: {args.output_dir}")
    print(f"Dry Run Mode: {args.dry_run}")

    res = run_validation_evaluation(output_dir=args.output_dir, dry_run=args.dry_run)
    summary = res["metrics_summary"]

    print("\n[VALIDATION EVALUATION COMPLETE]")
    print(f"Sample Count: {summary['sample_size']} (Visited: {summary['visited_count']}, Cold-Start: {summary['cold_start_count']})")

    for c_key, c_name in [
        ("condition_0_baseline", "Condition 0 (Baseline)"),
        ("condition_1_stage1_only", "Condition 1 (Stage 1 Only)"),
        ("condition_2_stage1_plus_c6", "Condition 2 (Stage 1 + C6)"),
    ]:
        ov = summary["conditions"][c_key]["overall"]
        vis = summary["conditions"][c_key]["visited"]
        cs = summary["conditions"][c_key]["cold_start"]
        print(f"\n--- {c_name} ---")
        print(f"  Overall:    Median={ov['median_error_m']}m, P90={ov['p90_error_m']}m, <=100m={ov['pct_within_100m']['pct']}%, Cov={ov['empirical_coverage']['pct']}%")
        print(f"  Visited:    Median={vis['median_error_m']}m, P90={vis['p90_error_m']}m, <=100m={vis['pct_within_100m']['pct']}%")
        print(f"  Cold-Start: Median={cs['median_error_m']}m, P90={cs['p90_error_m']}m, <=100m={cs['pct_within_100m']['pct']}%")

    pair = summary["paired_comparisons"]["condition_2_vs_condition_0"]
    print("\n--- Paired Comparison: Condition 2 vs Condition 0 ---")
    print(f"  Wins: {pair['wins']['count']}/{pair['wins']['denominator']} ({pair['wins']['pct']}%)")
    print(f"  Losses: {pair['losses']['count']}/{pair['losses']['denominator']} ({pair['losses']['pct']}%)")
    print(f"  Ties: {pair['ties']['count']}/{pair['ties']['denominator']} ({pair['ties']['pct']}%)")
    print(f"  Mean Delta: {pair['mean_delta_m']}m, Median Delta: {pair['median_delta_m']}m")

    if res["output_directory"]:
        print(f"\nArtifacts successfully written to: {res['output_directory']}")
    else:
        print("\nDry run completed; zero files written.")


if __name__ == "__main__":
    main()
