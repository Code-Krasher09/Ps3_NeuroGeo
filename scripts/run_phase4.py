"""Phase 4 Runner: Candidate Generation/Reranking & Leakage-Safe Empirical Quantile Calibration.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md,
and PS3_IMPLEMENTATION_PLAN.md Phase 4.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from neurogeo.config import PATHS, DEFAULT_CONFIG
from neurogeo.schemas import SpatialCandidate
from neurogeo.spatial_graph import build_hierarchical_street_graph, compute_scoped_neighbor_centroid
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
from neurogeo.calibrator import (
    assign_status_tier,
    fit_empirical_calibration_quantiles,
    calibrate_prediction,
)
from neurogeo.evaluation import (
    compute_error_metrics,
    compute_calibration_coverage,
    load_benchmark_ground_truth,
)
from neurogeo.leakage import assert_fit_scope_valid, assert_no_leakage


def main() -> None:
    print("=" * 75)
    print("NeuroGeo-Field: Phase 4 Execution")
    print("Candidate Generation/Reranking & Leakage-Safe Empirical Quantile Calibration")
    print("=" * 75)
    print(f"Repository Root: {PATHS.repo_root.name}")
    print(f"Outputs Directory: {PATHS.metrics_dir.relative_to(PATHS.repo_root)}")

    t0 = time.time()

    # -------------------------------------------------------------
    # Step 1: Ingest Input Tables & Reference Datasets
    # -------------------------------------------------------------
    print("\n[Step 1/6] Ingesting Phase 1-3 artifacts and reference data...")
    df_raw_addr = pd.read_csv(PATHS.addresses_csv)
    df_addr_ent = pd.read_parquet(PATHS.address_entities_parquet)
    if "account_id" not in df_addr_ent.columns:
        df_addr_ent = df_addr_ent.merge(df_raw_addr[["address_id", "account_id"]], on="address_id", how="left")
    
    df_medoids = pd.read_parquet(PATHS.metrics_dir / "address_doorstep_medoids.parquet")
    df_vis = pd.read_parquet(PATHS.metrics_dir / "visit_features.parquet")
    df_localities = pd.read_csv(PATHS.localities_csv)
    df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
    df_base = pd.read_csv(PATHS.baseline_geocodes_csv)
    df_surv = pd.read_csv(PATHS.surveyed_addresses_csv)
    df_splits = pd.read_csv(PATHS.splits_csv)
    df_scv = pd.read_csv(PATHS.scv_results_csv)

    print(f"  * Portfolio addresses: {len(df_addr_ent):,} rows")
    print(f"  * Visited doorstep medoids: {len(df_medoids):,} rows")
    print(f"  * Visit features: {len(df_vis):,} rows")
    print(f"  * Commercial baseline priors: {len(df_base):,} rows")
    print(f"  * Localities: {len(df_localities):,} neighborhoods")
    print(f"  * POI Landmarks: {len(df_pois):,} reference features")
    print(f"  * Surveyed benchmark addresses: {len(df_surv):,} ground truth rows")
    print(f"  * SCV visited-address episodes: {len(df_scv):,} episodes ({df_scv['address_id'].nunique():,} unique addresses across 1,339 unique visited accounts)")

    # -------------------------------------------------------------
    # Step 2: Build Hierarchical Spatial Graph
    # -------------------------------------------------------------
    print("\n[Step 2/6] Building hierarchical spatial graph for candidate C4 extraction...")
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
    print(f"  * Graph constructed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # -------------------------------------------------------------
    # Step 3: Fit Linear Reranker Weights on Train (N=66) Only
    # -------------------------------------------------------------
    print("\n[Step 3/6] Fitting deterministic linear reranker weights on Train split (N=66)...")
    # Identify splits for surveyed benchmark
    df_benchmark = df_surv.merge(
        df_addr_ent[[
            "address_id", "account_id", "town_id", "locality_id_resolved",
            "street_normalized", "landmark_poi_id", "spatial_relation",
            "locality_name_raw", "landmark_raw",
        ]],
        on="address_id",
        how="left",
    )
    df_benchmark = df_benchmark.merge(df_splits, on="account_id", how="left")

    train_mask = df_benchmark["split"] == "train"
    val_mask = df_benchmark["split"] == "validation"
    test_mask = df_benchmark["split"] == "test"

    train_addrs = set(df_benchmark[train_mask]["address_id"])
    val_addrs = set(df_benchmark[val_mask]["address_id"])
    test_addrs = set(df_benchmark[test_mask]["address_id"])

    print(f"  * Benchmark Split counts: Train={len(train_addrs)}, Val={len(val_addrs)}, Test={len(test_addrs)}")
    assert len(train_addrs) == 66, f"Expected 66 train, got {len(train_addrs)}"
    assert len(val_addrs) == 19, f"Expected 19 val, got {len(val_addrs)}"
    assert len(test_addrs) == 15, f"Expected 15 test, got {len(test_addrs)}"

    # Pre-index fast lookups
    baseline_map = df_base.set_index("address_id").to_dict("index")
    medoid_map = df_medoids.set_index("address_id").to_dict("index")
    poi_map = df_pois.set_index("poi_id").to_dict("index")
    loc_map = df_localities.set_index("locality_id").to_dict("index")
    visits_by_addr = df_vis.groupby("address_id").apply(lambda g: g.to_dict("records"), include_groups=False).to_dict()

    train_episodes: List[Dict[str, Any]] = []
    for _, row in df_benchmark[train_mask].iterrows():
        aid = row["address_id"]
        sx, sy = row["surveyed_x"], row["surveyed_y"]

        b_row = baseline_map.get(aid)
        m_row = medoid_map.get(aid)
        p_id = row.get("landmark_poi_id")
        p_row = poi_map.get(p_id) if pd.notna(p_id) else None
        l_id = row.get("locality_id_resolved")
        l_row = loc_map.get(l_id) if pd.notna(l_id) else None

        c4_res = compute_scoped_neighbor_centroid(
            address_id=aid,
            town_id=row.get("town_id"),
            locality_id=row.get("locality_id_resolved"),
            street_normalized=row.get("street_normalized"),
            graph=graph,
            fit_scope="benchmark_train",
        )

        cands = generate_candidates(
            address_id=aid,
            baseline_row=b_row,
            medoid_row=m_row,
            poi_row=p_row,
            c4_result=c4_res,
            locality_row=l_row,
            parsed_entity=row.to_dict(),
            fit_scope="benchmark_train",
        )

        v_list = visits_by_addr.get(aid, [])
        b_coord = (b_row["geocoder_x"], b_row["geocoder_y"]) if b_row and pd.notna(b_row.get("geocoder_x")) else None

        feated_cands = []
        errors = {}
        for c in cands:
            if c.available and c.x is not None and c.y is not None:
                f = extract_candidate_features(
                    candidate=c,
                    parsed_entity=row.to_dict(),
                    direct_visits=v_list,
                    baseline_coord=b_coord,
                )
                c_feated = replace(c, features=f)
                feated_cands.append(c_feated)
                errors[c.candidate_type] = compute_euclidean_distance(c.x, c.y, sx, sy)

        train_episodes.append({
            "address_id": aid,
            "candidates": feated_cands,
            "errors": errors,
        })

    # Fit reranker weights strictly on train set
    fit_result = fit_reranker_weights(
        train_records=train_episodes,
        random_state=42,
    )
    fitted_weights = fit_result["weights"]
    print(f"  * Reranker weights fitted on Train (N=66):")
    for k, v in fitted_weights.items():
        print(f"      {k}: {v:.4f}")

    # Export reranker_weights.json
    out_weights_json = PATHS.reranker_weights_json
    weights_artifact = {
        "fit_scope": "benchmark_train",
        "evaluation_scope": "benchmark_train",
        "source_data_scope": "data/shared/splits.csv (train split N=66)",
        "weights": fitted_weights,
        "feature_definitions": {
            "TextMatch": "Token and entity compatibility in [0, 1]",
            "DirectEvidence": "Dwell and integrity-weighted direct visit kernel in [0, 1]",
            "NeighborSupport": "Hierarchical street neighbor support kernel in [0, 1]",
            "BaselineAgreement": "Distance-decay agreement with commercial baseline prior in [0, 1]",
            "NegativeSearchPenalty": "Search failure penalty (address_not_traceable, neighbour_says_shifted) in [0, 1]",
        },
        "sample_count": fit_result["sample_count"],
        "pairwise_comparisons": fit_result["pairwise_comparisons"],
        "loss": fit_result["loss"],
        "convergence": fit_result["convergence"],
        "method": fit_result["method"],
        "random_state": fit_result["random_state"],
    }
    with open(out_weights_json, "w", encoding="utf-8") as f:
        json.dump(weights_artifact, f, indent=2)
    print(f"  * Exported: {out_weights_json.relative_to(PATHS.repo_root)}")

    # -------------------------------------------------------------
    # Step 4: Fit Empirical Quantile Calibration on SCV Residuals
    # -------------------------------------------------------------
    print("\n[Step 4/6] Fitting Evidence-Tier Empirical Quantile Calibration on SCV residuals...")
    # Strictly exclude all benchmark addresses (train N=66, validation N=19, test N=15) from calibration!
    cal_summary = fit_empirical_calibration_quantiles(
        df_scv=df_scv,
        df_localities=df_localities,
        train_address_ids=train_addrs,
        validation_address_ids=val_addrs,
        test_address_ids=test_addrs,
        min_samples=15,
        target_quantile=0.90,
        fit_scope="calibration_oof",
    )

    print("  * Calibration Provenance & Benchmark Split Isolation:")
    print(f"      Total SCV visited-address episodes: {len(df_scv):,} across 1,339 unique accounts")
    print(f"      Excluded benchmark addresses: {cal_summary['excluded_benchmark_address_count']} (Train={len(train_addrs)}, Val={len(val_addrs)}, Test={len(test_addrs)})")
    print(f"      Retained SCV calibration episodes: {cal_summary['total_scv_episodes']}")
    print("  * Empirical Quantiles by Evidence Tier (Target Coverage ~90%):")
    for t_name, t_data in cal_summary["tiers"].items():
        print(
            f"      {t_name:20s}: R = {t_data['radius_m']:6.2f}m "
            f"(Episodes={t_data['unique_episode_count']:4d}, Residuals={t_data['residual_count']:4d}, "
            f"fallback={t_data['fallback_level']}, family={t_data.get('evidence_family')})"
        )
    print(f"  * Fallback distribution: {cal_summary['fallback_counts']}")

    # Export calibration_quantiles.json
    out_cal_json = PATHS.calibration_quantiles_json
    with open(out_cal_json, "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, indent=2)
    print(f"  * Exported: {out_cal_json.relative_to(PATHS.repo_root)}")

    # -------------------------------------------------------------
    # Step 5: Full Benchmark Candidate Generation, Ranking & Predictions
    # -------------------------------------------------------------
    print("\n[Step 5/6] Generating candidates, ranking, and predictions across all 100 benchmark addresses...")
    benchmark_records: List[Dict[str, Any]] = []
    all_candidate_diagnostics: List[Dict[str, Any]] = []

    # Track candidate availability
    cand_avail_counts = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}

    for _, row in df_benchmark.iterrows():
        aid = row["address_id"]
        sx, sy = row["surveyed_x"], row["surveyed_y"]
        split = row["split"]

        b_row = baseline_map.get(aid)
        m_row = medoid_map.get(aid)
        p_id = row.get("landmark_poi_id")
        p_row = poi_map.get(p_id) if pd.notna(p_id) else None
        l_id = row.get("locality_id_resolved")
        l_row = loc_map.get(l_id) if pd.notna(l_id) else None

        c4_res = compute_scoped_neighbor_centroid(
            address_id=aid,
            town_id=row.get("town_id"),
            locality_id=row.get("locality_id_resolved"),
            street_normalized=row.get("street_normalized"),
            graph=graph,
            fit_scope="production_all",
        )

        cands = generate_candidates(
            address_id=aid,
            baseline_row=b_row,
            medoid_row=m_row,
            poi_row=p_row,
            c4_result=c4_res,
            locality_row=l_row,
            parsed_entity=row.to_dict(),
            fit_scope="production_all",
        )

        for c in cands:
            if c.available:
                cand_avail_counts[c.candidate_type] = cand_avail_counts.get(c.candidate_type, 0) + 1

        v_list = visits_by_addr.get(aid, [])
        is_visited = len(v_list) > 0
        b_coord = (b_row["geocoder_x"], b_row["geocoder_y"]) if b_row and pd.notna(b_row.get("geocoder_x")) else None

        # Feature extraction
        feated_cands = []
        for c in cands:
            if c.available and c.x is not None and c.y is not None:
                f = extract_candidate_features(
                    candidate=c,
                    parsed_entity=row.to_dict(),
                    direct_visits=v_list,
                    baseline_coord=b_coord,
                )
                feated_cands.append(replace(c, features=f))

        # Rank candidates using fitted weights
        ranked = rank_candidates(feated_cands, weights=fitted_weights, deduplicate=True)
        assert len(ranked) > 0, f"No candidates available for {aid}"
        
        selected = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = selected.total_score - (runner_up.total_score if runner_up else 0.0)

        # Status tier assignment
        has_cred = bool(m_row.get("has_credible_doorstep", False)) if m_row else False
        n_cnt = int(c4_res.get("neighbor_count", 0)) if c4_res else 0
        is_poi_res = bool(p_row is not None)
        b_prec = str(b_row.get("precision", "unknown")) if b_row else "unknown"

        status_tier = assign_status_tier(
            selected_candidate_type=selected.candidate_type,
            has_credible_doorstep=has_cred,
            neighbor_count=n_cnt,
            is_poi_resolved=is_poi_res,
            baseline_precision=b_prec,
        )

        # Calibration
        rad_m, conf_level, fb_level, sample_cnt = calibrate_prediction(status_tier, cal_summary)

        # Error vs ground truth surveyed coordinate
        err_m = compute_euclidean_distance(selected.x, selected.y, sx, sy)
        is_covered = bool(err_m <= rad_m)

        # Record prediction
        eval_scope = f"eval_{split}" if split in {"validation", "test"} else "train_all"
        fit_scope = "benchmark_train"

        pred_rec = {
            "address_id": aid,
            "predicted_x": round(selected.x, 2),
            "predicted_y": round(selected.y, 2),
            "selected_candidate_type": selected.candidate_type,
            "status_tier": status_tier.value,
            "confidence_radius_m": rad_m,
            "confidence_level": conf_level,
            "candidate_count": len(ranked),
            "selected_score": round(selected.total_score, 4),
            "runner_up_score": round(runner_up.total_score if runner_up else 0.0, 4),
            "score_margin": round(margin, 4),
            "evidence_sources": ",".join(selected.agreed_sources),
            "calibration_tier": status_tier.value,
            "calibration_sample_count": sample_cnt,
            "calibration_fallback_level": fb_level,
            "fit_scope": fit_scope,
            "evaluation_scope": eval_scope,
            # Ground truth verification fields (evaluation only)
            "is_surveyed_benchmark": True,
            "split": split,
            "is_visited": is_visited,
            "surveyed_x": sx,
            "surveyed_y": sy,
            "error_vs_surveyed_m": round(err_m, 2),
            "is_covered": is_covered,
        }
        benchmark_records.append(pred_rec)

        # Record candidate diagnostics for parquet export
        for c in ranked:
            all_candidate_diagnostics.append({
                "address_id": aid,
                "candidate_id": c.candidate_id,
                "candidate_type": c.candidate_type,
                "x": c.x,
                "y": c.y,
                "available": c.available,
                "agreed_sources": ",".join(c.agreed_sources),
                "TextMatch": c.features.get("TextMatch", 0.0),
                "DirectEvidence": c.features.get("DirectEvidence", 0.0),
                "NeighborSupport": c.features.get("NeighborSupport", 0.0),
                "BaselineAgreement": c.features.get("BaselineAgreement", 0.0),
                "NegativeSearchPenalty": c.features.get("NegativeSearchPenalty", 0.0),
                "total_score": c.total_score,
                "rank": c.rank,
                "fit_scope": fit_scope,
                "evaluation_scope": eval_scope,
            })

    df_preds = pd.DataFrame(benchmark_records)
    df_cands_diag = pd.DataFrame(all_candidate_diagnostics)

    # Export predictions.csv and candidate_scores.parquet
    out_preds_csv = PATHS.phase4_predictions_csv
    df_preds.to_csv(out_preds_csv, index=False)
    print(f"  * Exported: {out_preds_csv.relative_to(PATHS.repo_root)} ({len(df_preds)} rows)")

    out_cands_parquet = PATHS.candidate_scores_parquet
    df_cands_diag.to_parquet(out_cands_parquet, index=False)
    print(f"  * Exported: {out_cands_parquet.relative_to(PATHS.repo_root)} ({len(df_cands_diag)} rows)")

    # -------------------------------------------------------------
    # Step 6: Execute Official Ablation Ladder (M0 - M6)
    # -------------------------------------------------------------
    print("\n[Step 6/6] Computing official Ablation Ladder (M0 -> M6)...")
    # Precompute raw visit check-in mean per address
    raw_visits = df_vis.groupby("address_id")[["checkin_x", "checkin_y"]].mean().to_dict("index")

    # For each benchmark address, prepare candidate coordinates
    ablation_rows = []
    
    # Store candidate coords per address
    addr_candidates: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {}
    for _, row in df_benchmark.iterrows():
        aid = row["address_id"]
        b_row = baseline_map.get(aid)
        m_row = medoid_map.get(aid)
        p_id = row.get("landmark_poi_id")
        p_row = poi_map.get(p_id) if pd.notna(p_id) else None
        
        c1_c = (float(b_row["geocoder_x"]), float(b_row["geocoder_y"])) if b_row and pd.notna(b_row.get("geocoder_x")) else None
        
        raw_v = raw_visits.get(aid)
        raw_c = (float(raw_v["checkin_x"]), float(raw_v["checkin_y"])) if raw_v else None
        
        c2_c = None
        if m_row and bool(m_row.get("has_credible_doorstep", False)) and pd.notna(m_row.get("doorstep_medoid_x")):
            c2_c = (float(m_row["doorstep_medoid_x"]), float(m_row["doorstep_medoid_y"]))
            
        c3_c = (float(p_row["x"]), float(p_row["y"])) if p_row and pd.notna(p_row.get("x")) else None
        
        c4_res = compute_scoped_neighbor_centroid(
            address_id=aid,
            town_id=row.get("town_id"),
            locality_id=row.get("locality_id_resolved"),
            street_normalized=row.get("street_normalized"),
            graph=graph,
        )
        c4_c = (float(c4_res["c4_x"]), float(c4_res["c4_y"])) if c4_res else None

        addr_candidates[aid] = {
            "c1": c1_c,
            "raw": raw_c,
            "c2": c2_c,
            "c3": c3_c,
            "c4": c4_c,
        }

    # Baseline M0 error per address
    m0_errors: Dict[str, float] = {}
    for _, row in df_benchmark.iterrows():
        aid = row["address_id"]
        c1 = addr_candidates[aid]["c1"]
        m0_errors[aid] = compute_euclidean_distance(c1[0], c1[1], row["surveyed_x"], row["surveyed_y"]) if c1 else np.nan

    split_map = df_benchmark.set_index("address_id")["split"].to_dict()
    vis_map = {row["address_id"]: bool(row["address_id"] in visits_by_addr and len(visits_by_addr[row["address_id"]]) > 0) for _, row in df_benchmark.iterrows()}

    # Stages definition
    stages = [
        ("M0", "Commercial Baseline Only (C1)"),
        ("M1", "Baseline + Raw Direct Visit Check-ins (Unweighted)"),
        ("M2", "Baseline + Integrity-Weighted Doorstep Medoids (C2)"),
        ("M3", "Baseline + Doorstep Medoids + NLP/POI Anchor (C3)"),
        ("M4", "Baseline + Medoids + POI + Spatial Graph Centroid (C4)"),
        ("M5", "Learned Deterministic Linear Reranker (w_neg=0)"),
        ("M6", "Full Deterministic Reranker + Negative Search Penalty"),
    ]

    w_no_neg = dict(fitted_weights)
    w_no_neg["w_neg"] = 0.0

    ablation_table_data = []

    for stage_id, stage_desc in stages:
        stage_errs: Dict[str, float] = {}
        avail_count = 0

        for _, row in df_benchmark.iterrows():
            aid = row["address_id"]
            sx, sy = row["surveyed_x"], row["surveyed_y"]
            c_dict = addr_candidates[aid]

            pred_coord = None
            if stage_id == "M0":
                pred_coord = c_dict["c1"]
            elif stage_id == "M1":
                pred_coord = c_dict["raw"] if c_dict["raw"] is not None else c_dict["c1"]
            elif stage_id == "M2":
                pred_coord = c_dict["c2"] if c_dict["c2"] is not None else c_dict["c1"]
            elif stage_id == "M3":
                pred_coord = c_dict["c2"] if c_dict["c2"] is not None else (c_dict["c3"] if c_dict["c3"] is not None else c_dict["c1"])
            elif stage_id == "M4":
                pred_coord = c_dict["c2"] if c_dict["c2"] is not None else (c_dict["c4"] if c_dict["c4"] is not None else (c_dict["c3"] if c_dict["c3"] is not None else c_dict["c1"]))
            elif stage_id == "M5":
                # Rerank with w_neg = 0
                feated = [c for c in all_candidate_diagnostics if c["address_id"] == aid]
                # Reconstruct candidate objects
                rec_cands = [
                    SpatialCandidate(
                        candidate_id=c["candidate_id"], candidate_type=c["candidate_type"],
                        x=c["x"], y=c["y"], available=c["available"],
                        features={
                            "TextMatch": c["TextMatch"], "DirectEvidence": c["DirectEvidence"],
                            "NeighborSupport": c["NeighborSupport"], "BaselineAgreement": c["BaselineAgreement"],
                            "NegativeSearchPenalty": c["NegativeSearchPenalty"],
                        },
                        agreed_sources=tuple(c["agreed_sources"].split(",")) if c["agreed_sources"] else (),
                    )
                    for c in feated
                ]
                r5 = rank_candidates(rec_cands, weights=w_no_neg, deduplicate=False)
                pred_coord = (r5[0].x, r5[0].y) if r5 else c_dict["c1"]
            elif stage_id == "M6":
                # Rerank with full weights
                pred_row = df_preds[df_preds["address_id"] == aid].iloc[0]
                pred_coord = (float(pred_row["predicted_x"]), float(pred_row["predicted_y"]))

            if pred_coord is not None:
                avail_count += 1
                err = compute_euclidean_distance(pred_coord[0], pred_coord[1], sx, sy)
                stage_errs[aid] = err
            else:
                stage_errs[aid] = np.nan

        # Compute summary metrics across segments
        all_e = pd.Series(stage_errs)
        train_e = pd.Series({aid: stage_errs[aid] for aid in all_e.index if split_map.get(aid) == "train"}).dropna()
        val_e = pd.Series({aid: stage_errs[aid] for aid in all_e.index if split_map.get(aid) == "validation"}).dropna()
        test_e = pd.Series({aid: stage_errs[aid] for aid in all_e.index if split_map.get(aid) == "test"}).dropna()
        vis_e = pd.Series({aid: stage_errs[aid] for aid in all_e.index if vis_map.get(aid, False)}).dropna()
        cold_e = pd.Series({aid: stage_errs[aid] for aid in all_e.index if not vis_map.get(aid, False)}).dropna()

        m0_median = pd.Series(m0_errors).median()
        curr_median = all_e.median()
        imp_vs_base = m0_median - curr_median

        ablation_table_data.append({
            "stage": stage_id,
            "description": stage_desc,
            "candidate_availability": f"{avail_count}/100",
            "overall_median_error_m": round(curr_median, 2),
            "overall_p90_error_m": round(all_e.quantile(0.9), 2),
            "overall_max_error_m": round(all_e.max(), 2),
            "improvement_vs_baseline_m": round(imp_vs_base, 2),
            "visited_median_error_m": round(vis_e.median(), 2),
            "visited_p90_error_m": round(vis_e.quantile(0.9), 2),
            "cold_start_median_error_m": round(cold_e.median(), 2),
            "cold_start_p90_error_m": round(cold_e.quantile(0.9), 2),
            "train_median_error_m": round(train_e.median(), 2),
            "val_median_error_m": round(val_e.median(), 2),
            "test_median_error_m": round(test_e.median(), 2),
        })

    df_ablation = pd.DataFrame(ablation_table_data)
    out_ablation_csv = PATHS.phase4_ablation_csv
    df_ablation.to_csv(out_ablation_csv, index=False)
    print(f"  * Exported: {out_ablation_csv.relative_to(PATHS.repo_root)}")

    # -------------------------------------------------------------
    # Step 7: Print Verification & Diagnostics Matrix
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE 4 ABLATION LADDER SUMMARY (M0 -> M6)")
    print("=" * 75)
    for _, r in df_ablation.iterrows():
        print(
            f"{r['stage']:4s} | {r['description'][:48]:48s} | "
            f"Median: {r['overall_median_error_m']:6.2f}m | "
            f"P90: {r['overall_p90_error_m']:6.2f}m | "
            f"Imp: +{r['improvement_vs_baseline_m']:6.2f}m"
        )

    print("\n" + "=" * 75)
    print("BENCHMARK SEGMENT PERFORMANCE COMPARISON")
    print("=" * 75)
    m0_row = df_ablation[df_ablation["stage"] == "M0"].iloc[0]
    m6_row = df_ablation[df_ablation["stage"] == "M6"].iloc[0]
    print(f"Segment                   | Commercial Baseline (M0) | NeuroGeo Phase 4 (M6) | Improvement")
    print("-" * 75)
    print(f"All Benchmark (N=100)     | {m0_row['overall_median_error_m']:6.2f}m / {m0_row['overall_p90_error_m']:6.2f}m   | {m6_row['overall_median_error_m']:6.2f}m / {m6_row['overall_p90_error_m']:6.2f}m  | +{m0_row['overall_median_error_m'] - m6_row['overall_median_error_m']:6.2f}m")
    print(f"Visited Addresses (N=49)  | {m0_row['visited_median_error_m']:6.2f}m / {m0_row['visited_p90_error_m']:6.2f}m   | {m6_row['visited_median_error_m']:6.2f}m / {m6_row['visited_p90_error_m']:6.2f}m  | +{m0_row['visited_median_error_m'] - m6_row['visited_median_error_m']:6.2f}m")
    print(f"Cold-Start Unvisited (N=51)| {m0_row['cold_start_median_error_m']:6.2f}m / {m0_row['cold_start_p90_error_m']:6.2f}m   | {m6_row['cold_start_median_error_m']:6.2f}m / {m6_row['cold_start_p90_error_m']:6.2f}m  | +{m0_row['cold_start_median_error_m'] - m6_row['cold_start_median_error_m']:6.2f}m")
    print(f"Train Split (N=66)        | Median: {m0_row['train_median_error_m']:6.2f}m           | Median: {m6_row['train_median_error_m']:6.2f}m          | +{m0_row['train_median_error_m'] - m6_row['train_median_error_m']:6.2f}m")
    print(f"Validation Split (N=19)   | Median: {m0_row['val_median_error_m']:6.2f}m           | Median: {m6_row['val_median_error_m']:6.2f}m          | +{m0_row['val_median_error_m'] - m6_row['val_median_error_m']:6.2f}m")
    print(f"Untouched Test Split (N=15)| Median: {m0_row['test_median_error_m']:6.2f}m           | Median: {m6_row['test_median_error_m']:6.2f}m          | +{m0_row['test_median_error_m'] - m6_row['test_median_error_m']:6.2f}m")

    # Empirical coverage verification
    overall_cov = df_preds["is_covered"].mean()
    vis_cov = df_preds[df_preds["is_visited"] == True]["is_covered"].mean()
    cold_cov = df_preds[df_preds["is_visited"] == False]["is_covered"].mean()
    val_cov = df_preds[df_preds["split"] == "validation"]["is_covered"].mean()
    test_cov = df_preds[df_preds["split"] == "test"]["is_covered"].mean()

    print("\n" + "=" * 75)
    print(f"CALIBRATION COVERAGE VERIFICATION (Target ~90%):")
    print(f"  * All Benchmark Coverage: {overall_cov * 100:.1f}%")
    print(f"  * Visited Addresses Coverage: {vis_cov * 100:.1f}%")
    print(f"  * Cold-Start Coverage: {cold_cov * 100:.1f}%")
    print(f"  * Validation Split Coverage: {val_cov * 100:.1f}%")
    print(f"  * Untouched Test Split Coverage: {test_cov * 100:.1f}%")

    # Summary JSON export
    summary = {
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 4,
        "phase_description": "Candidate Generation, Deterministic Linear Reranking & Empirical Quantile Calibration",
        "provenance": {
            "reranker_weights_fit_scope": "benchmark_train",
            "calibration_fit_scope": "calibration_oof",
            "prediction_fit_scope": "benchmark_train",
            "test_split_untouched": True,
            "test_split_size": 15,
            "validation_split_size": 19,
            "train_split_size": 66,
            "total_scv_visited_address_episodes": len(df_scv),
            "retained_calibration_episodes": cal_summary["total_scv_episodes"],
            "disjoint_from_reranker_train": True,
            "disjoint_from_val": True,
            "disjoint_from_test": True,
            "candidate_availability_policy": (
                "Test addresses are excluded from candidate selection tuning, candidate-prior fitting, "
                "reranker fitting, and calibration estimation. Candidate availability is measured during "
                "the final untouched evaluation for reporting, but no test truth is used to alter availability rules."
            ),
        },
        "candidate_availability_on_benchmark": cand_avail_counts,
        "reranker_weights": fitted_weights,
        "calibration_tiers": cal_summary["tiers"],
        "calibration_fallback_counts": cal_summary["fallback_counts"],
        "ablation_ladder": df_ablation.to_dict("records"),
        "metrics_summary": {
            "all_benchmark": {
                "sample_size": 100,
                "baseline_median_m": float(m0_row["overall_median_error_m"]),
                "phase4_median_m": float(m6_row["overall_median_error_m"]),
                "baseline_p90_m": float(m0_row["overall_p90_error_m"]),
                "phase4_p90_m": float(m6_row["overall_p90_error_m"]),
                "improvement_m": float(m0_row["overall_median_error_m"] - m6_row["overall_median_error_m"]),
                "empirical_coverage": round(float(overall_cov), 4),
            },
            "visited_segment": {
                "sample_size": 49,
                "baseline_median_m": float(m0_row["visited_median_error_m"]),
                "phase4_median_m": float(m6_row["visited_median_error_m"]),
                "baseline_p90_m": float(m0_row["visited_p90_error_m"]),
                "phase4_p90_m": float(m6_row["visited_p90_error_m"]),
                "improvement_m": float(m0_row["visited_median_error_m"] - m6_row["visited_median_error_m"]),
                "empirical_coverage": round(float(vis_cov), 4),
            },
            "cold_start_segment": {
                "sample_size": 51,
                "baseline_median_m": float(m0_row["cold_start_median_error_m"]),
                "phase4_median_m": float(m6_row["cold_start_median_error_m"]),
                "baseline_p90_m": float(m0_row["cold_start_p90_error_m"]),
                "phase4_p90_m": float(m6_row["cold_start_p90_error_m"]),
                "improvement_m": float(m0_row["cold_start_median_error_m"] - m6_row["cold_start_median_error_m"]),
                "empirical_coverage": round(float(cold_cov), 4),
            },
            "splits": {
                "train_n66": {
                    "baseline_median_m": float(m0_row["train_median_error_m"]),
                    "phase4_median_m": float(m6_row["train_median_error_m"]),
                },
                "val_n19": {
                    "baseline_median_m": float(m0_row["val_median_error_m"]),
                    "phase4_median_m": float(m6_row["val_median_error_m"]),
                    "empirical_coverage": round(float(val_cov), 4),
                },
                "test_n15": {
                    "baseline_median_m": float(m0_row["test_median_error_m"]),
                    "phase4_median_m": float(m6_row["test_median_error_m"]),
                    "empirical_coverage": round(float(test_cov), 4),
                },
            },
        },
        "artifact_locations": {
            "candidate_scores_parquet": PATHS.candidate_scores_parquet.relative_to(PATHS.repo_root).as_posix(),
            "phase4_predictions_csv": PATHS.phase4_predictions_csv.relative_to(PATHS.repo_root).as_posix(),
            "reranker_weights_json": PATHS.reranker_weights_json.relative_to(PATHS.repo_root).as_posix(),
            "calibration_quantiles_json": PATHS.calibration_quantiles_json.relative_to(PATHS.repo_root).as_posix(),
            "phase4_ablation_csv": PATHS.phase4_ablation_csv.relative_to(PATHS.repo_root).as_posix(),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_summary_json = PATHS.phase4_summary_json
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Exported phase4_summary.json to {out_summary_json.relative_to(PATHS.repo_root)}")
    print(f"Total elapsed time: {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
