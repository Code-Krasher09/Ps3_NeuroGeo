"""Leakage-Safe Validation: Evidence-Tier, Candidate-Wise, Landmark-Geometry, and Combined Hierarchy.

Authority: Prompt 30 Phase D requirements.
Safety Guarantees:
- Excludes all 100 official benchmark addresses from calibration fitting and evaluation.
- Never accesses or evaluates the official 15-address test split.
- Uses 5-fold grouped cross-validation by account_id on clean SCV episodes (1,428 episodes, 1,295 accounts).
- Fits all quantiles strictly within training folds only.
- Outputs exclusively to reports/eval_production_calibration/.
"""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure src is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from neurogeo.calibrator import (
    assign_status_tier,
    calibrate_prediction_production,
    compute_landmark_spatial_scale,
    find_nearest_unmentioned_poi,
    get_calibrated_radius_with_fallback,
)
from neurogeo.candidate_ranker import (
    extract_candidate_features,
    generate_candidates,
    rank_candidates,
)
from neurogeo.config import DEFAULT_CONFIG, PATHS, ModelConfig
from neurogeo.schemas import PredictionStatus


def run_production_calibration_validation(
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NEUROGEO PS3: PRODUCTION CALIBRATION HIERARCHY VALIDATION (5-FOLD GROUPED CV)")
    print("=" * 80)
    print(f"Target Output Directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 1. Ingest input artifacts
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Ingesting source datasets and SCV artifacts...")
    df_scv = pd.read_csv(PATHS.scv_results_csv)
    df_addr = pd.read_csv(PATHS.addresses_csv)
    df_surv = pd.read_csv(PATHS.surveyed_addresses_csv)
    df_parsed = pd.read_parquet(PATHS.parsed_addresses_parquet)
    df_base = pd.read_csv(PATHS.baseline_geocodes_csv)
    df_localities = pd.read_csv(PATHS.localities_csv)
    df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
    weights = json.loads(PATHS.reranker_weights_json.read_text())["weights"]

    # Exclude all 100 official benchmark addresses
    bench_address_ids = set(df_surv["address_id"])
    assert len(bench_address_ids) == 100, f"Expected 100 benchmark addresses, got {len(bench_address_ids)}"

    scv_with_acc = df_scv.merge(df_addr[["address_id", "account_id"]], on="address_id", how="left")
    scv_clean = scv_with_acc[~scv_with_acc["address_id"].isin(bench_address_ids)].copy().reset_index(drop=True)

    print(f"  * Total SCV rows available: {len(df_scv)}")
    print(f"  * Benchmark addresses excluded: {len(df_scv) - len(scv_clean)}")
    print(f"  * Retained clean SCV episodes: {len(scv_clean)}")
    print(f"  * Distinct accounts in clean sample: {scv_clean['account_id'].nunique()}")

    parsed_map = df_parsed.set_index("address_id").to_dict("index")
    base_map = df_base.set_index("address_id").to_dict("index")
    loc_map = df_localities.set_index("locality_id").to_dict("index")
    poi_map = df_pois.set_index("poi_id").to_dict("index")

    # -------------------------------------------------------------------------
    # 2. Simulate production inference under cold start (direct visits masked)
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Simulating production inference path under cold start...")
    from dataclasses import replace

    episodes: List[Dict[str, Any]] = []
    cfg = DEFAULT_CONFIG

    for _, row in scv_clean.iterrows():
        aid = row["address_id"]
        p_ent = parsed_map.get(aid, {})
        b_row = base_map.get(aid)

        c4_avail = bool(row["c4_available"])
        c4_res = {
            "c4_x": row["c4_x"] if c4_avail else None,
            "c4_y": row["c4_y"] if c4_avail else None,
            "neighbor_count": int(row["neighbor_count"]) if c4_avail else 0,
            "total_neighbor_weight": float(row["total_neighbor_weight"]) if c4_avail else 0.0,
            "neighbor_address_ids": str(row["neighbor_address_ids"]).split(",") if pd.notna(row.get("neighbor_address_ids")) else [],
            "town_id": row["town_id"],
            "locality_id": row["locality_id_resolved"],
            "street_normalized": row["street_normalized"],
        }

        lid = row.get("locality_id_resolved")
        l_row = loc_map.get(str(lid)) if pd.notna(lid) else None
        pid = p_ent.get("landmark_poi_id")
        p_row = poi_map.get(pid) if pd.notna(pid) else None

        # Direct doorstep visits are masked out (simulated cold start)
        cands = generate_candidates(
            address_id=aid,
            baseline_row=b_row,
            medoid_row=None,
            poi_row=p_row,
            c4_result=c4_res,
            locality_row=l_row,
            parsed_entity=p_ent,
            fit_scope="scv_leave_one_out",
        )

        b_coord = (float(b_row["geocoder_x"]), float(b_row["geocoder_y"])) if b_row and pd.notna(b_row.get("geocoder_x")) else None
        feated = []
        for c in cands:
            if c.available and c.x is not None and c.y is not None:
                f = extract_candidate_features(
                    candidate=c,
                    parsed_entity=p_ent,
                    direct_visits=None,
                    baseline_coord=b_coord,
                )
                feated.append(replace(c, features=f))

        ranked = rank_candidates(feated, weights=weights, deduplicate=True)
        assert len(ranked) > 0, f"No candidate available for {aid}"
        sel = ranked[0]

        tier = assign_status_tier(
            selected_candidate_type=sel.candidate_type,
            has_credible_doorstep=False,
            neighbor_count=c4_res["neighbor_count"] if sel.candidate_type == "C4" else 0,
            is_poi_resolved=bool(p_row is not None),
            baseline_precision=str(b_row.get("precision", "unknown")) if b_row else "unknown",
        )

        gt_x = float(row["ground_truth_medoid_x"])
        gt_y = float(row["ground_truth_medoid_y"])
        err_m = float(math.hypot(sel.x - gt_x, sel.y - gt_y))

        # Check landmark geometry eligibility
        geom_eligible = False
        g_scale: Optional[float] = None
        d_min_val: Optional[float] = None
        unmentioned_poi_id: Optional[str] = None
        geom_rejection_reason: Optional[str] = None

        if p_ent.get("poi_resolution_status") == "resolved" and pid and pid in poi_map:
            anchor = poi_map[pid]
            ax, ay = float(anchor["x"]), float(anchor["y"])
            town = str(anchor["town_id"])

            d_min_val, unmentioned_poi_id, pool_status = find_nearest_unmentioned_poi(
                primary_poi_id=pid,
                town_id=town,
                locality_id=str(lid) if pd.notna(lid) else None,
                df_pois=df_pois,
                df_localities=df_localities,
                collocation_distance_threshold_m=cfg.collocation_distance_threshold_m,
                town_fallback_radius_m=cfg.town_poi_fallback_radius_m,
            )

            if d_min_val is not None:
                g_scale, scale_status = compute_landmark_spatial_scale(
                    predicted_coord=(sel.x, sel.y),
                    anchor_coord=(ax, ay),
                    nearest_unmentioned_distance_m=d_min_val,
                    max_consistency_distance_m=cfg.landmark_spatial_consistency_radius_m,
                    min_scale_floor_m=cfg.min_calibration_radius_floor_m,
                )
                if g_scale is not None:
                    geom_eligible = True
                else:
                    geom_rejection_reason = scale_status
            else:
                geom_rejection_reason = pool_status
        else:
            geom_rejection_reason = f"landmark_{p_ent.get('poi_resolution_status', 'unresolved')}"

        episodes.append({
            "address_id": aid,
            "account_id": str(row["account_id"]),
            "town_id": str(row["town_id"]),
            "locality_id": str(lid) if pd.notna(lid) else "",
            "street_normalized": str(row.get("street_normalized", "")),
            "selected_candidate_type": sel.candidate_type,
            "status_tier": tier.value,
            "pred_x": round(float(sel.x), 2),
            "pred_y": round(float(sel.y), 2),
            "gt_x": round(gt_x, 2),
            "gt_y": round(gt_y, 2),
            "error_m": round(err_m, 2),
            "poi_resolution_status": p_ent.get("poi_resolution_status", "unresolved"),
            "landmark_poi_id": pid if pd.notna(pid) else "",
            "geom_eligible": geom_eligible,
            "geom_scale_m": g_scale,
            "d_min_m": d_min_val,
            "unmentioned_poi_id": unmentioned_poi_id or "",
            "geom_rejection_reason": geom_rejection_reason or "",
        })

    df_eval = pd.DataFrame(episodes)
    print(f"  * Predictions successfully reconstructed: {len(df_eval)}")
    print(f"  * Candidate selection breakdown: {dict(Counter(df_eval['selected_candidate_type']))}")
    print(f"  * Status tier breakdown: {dict(Counter(df_eval['status_tier']))}")
    print(f"  * Landmark geometry eligible: {int(df_eval['geom_eligible'].sum())} / {len(df_eval)} ({df_eval['geom_eligible'].mean() * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # 3. Partition into 5 grouped folds by account_id
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Partitioning into 5-fold grouped cross-validation by account_id...")
    gkf = GroupKFold(n_splits=5)
    df_eval["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_eval, groups=df_eval["account_id"])):
        df_eval.loc[val_idx, "fold"] = fold_idx

    # Initialize evaluation columns
    df_eval["m1_tier_radius_m"] = np.nan
    df_eval["m1_tier_is_covered"] = False

    df_eval["m2_cand_radius_m"] = np.nan
    df_eval["m2_cand_is_covered"] = False

    df_eval["m3_geom_radius_m"] = np.nan
    df_eval["m3_geom_is_covered"] = False

    df_eval["m4_prod_radius_m"] = np.nan
    df_eval["m4_prod_is_covered"] = False
    df_eval["m4_prod_method"] = ""
    df_eval["m4_prod_fallback_level"] = ""
    df_eval["m4_prod_fallback_reason"] = ""

    fold_metrics = []

    # -------------------------------------------------------------------------
    # 4. Execute fold-by-fold cross-validation
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Executing 5-fold comparison across all 4 calibration methods...")

    for fold_idx in range(5):
        train_mask = df_eval["fold"] != fold_idx
        val_mask = df_eval["fold"] == fold_idx
        train_df = df_eval[train_mask]
        val_df = df_eval[val_mask]

        # ---------------------------------------------------------------------
        # Fitting within training fold only:
        # ---------------------------------------------------------------------

        # A. Evidence-tier calibration
        tier_radii: Dict[str, float] = {}
        global_p90 = float(np.quantile(train_df["error_m"], 0.90))

        for t_name in ["STREET_RESOLVED", "LOCALITY_BOUNDED", "BASELINE_FALLBACK", "DOORSTEP_VERIFIED"]:
            t_errs = train_df[train_df["status_tier"] == t_name]["error_m"]
            if len(t_errs) >= cfg.min_tier_samples:
                r = float(np.quantile(t_errs, 0.90))
            else:
                r = global_p90
            tier_radii[t_name] = min(max(round(r, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)

        # B. Candidate-wise P90 calibration
        cand_radii: Dict[str, float] = {}
        for c_type in ["C1", "C2", "C3", "C4", "C5"]:
            c_errs = train_df[train_df["selected_candidate_type"] == c_type]["error_m"]
            if len(c_errs) >= cfg.min_tier_samples:
                r = float(np.quantile(c_errs, 0.90))
                cand_radii[c_type] = min(max(round(r, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)
            else:
                default_tier = {
                    "C1": "BASELINE_FALLBACK",
                    "C2": "DOORSTEP_VERIFIED",
                    "C3": "STREET_RESOLVED",
                    "C4": "STREET_RESOLVED",
                    "C5": "LOCALITY_BOUNDED",
                }.get(c_type, "LOCALITY_BOUNDED")
                cand_radii[c_type] = tier_radii[default_tier]

        # C. Landmark geometry normalized-residual calibration
        geom_train = train_df[train_df["geom_eligible"]].copy()
        geom_train["q_i"] = geom_train["error_m"] / geom_train["geom_scale_m"]

        pooled_q_factor = float(np.quantile(geom_train["q_i"], 0.90)) if len(geom_train) >= cfg.min_tier_samples else 5.7363
        cand_q_factors: Dict[str, float] = {}
        for c_type in ["C1", "C3", "C4", "C5"]:
            sub_q = geom_train[geom_train["selected_candidate_type"] == c_type]["q_i"]
            if len(sub_q) >= cfg.min_tier_samples:
                cand_q_factors[c_type] = float(np.quantile(sub_q, 0.90))
            else:
                cand_q_factors[c_type] = pooled_q_factor

        # D. Locality empirical P90
        locality_radii: Dict[str, float] = {}
        for lid, group in train_df.groupby("locality_id"):
            if lid and len(group) >= cfg.min_tier_samples:
                r_loc = float(np.quantile(group["error_m"], 0.90))
                locality_radii[str(lid)] = min(max(round(r_loc, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)

        # Build mock calibration_dict representing this fold's training fit
        fold_cal_dict = {
            "tiers": {
                t: {"radius_m": r, "quantile": 0.90, "sample_count": len(train_df[train_df["status_tier"] == t]), "fallback_level": "tier_direct"}
                for t, r in tier_radii.items()
            },
            "candidates": {
                c: {"radius_m": r, "quantile": 0.90, "sample_count": len(train_df[train_df["selected_candidate_type"] == c])}
                for c, r in cand_radii.items()
            },
            "landmark_geometry": {
                "pooled_quantile_factor": round(pooled_q_factor, 4),
                "candidate_quantile_factors": {c: round(f, 4) for c, f in cand_q_factors.items()},
                "candidate_sample_counts": {c: len(geom_train[geom_train["selected_candidate_type"] == c]) for c in ["C1", "C3", "C4", "C5"]},
                "sample_count": len(geom_train),
            },
            "localities": {
                lid: {"radius_m": r, "sample_count": len(train_df[train_df["locality_id"] == lid])}
                for lid, r in locality_radii.items()
            },
        }

        # ---------------------------------------------------------------------
        # Evaluation on validation fold
        # ---------------------------------------------------------------------
        for idx in val_df.index:
            t = df_eval.loc[idx, "status_tier"]
            c = df_eval.loc[idx, "selected_candidate_type"]
            e = df_eval.loc[idx, "error_m"]
            is_geom = bool(df_eval.loc[idx, "geom_eligible"])
            g_scale = df_eval.loc[idx, "geom_scale_m"]
            lid = df_eval.loc[idx, "locality_id"]

            # Method 1: Evidence-Tier
            r_m1 = tier_radii[t]
            df_eval.loc[idx, "m1_tier_radius_m"] = r_m1
            df_eval.loc[idx, "m1_tier_is_covered"] = bool(e <= r_m1)

            # Method 2: Candidate-Wise P90
            r_m2 = cand_radii[c]
            df_eval.loc[idx, "m2_cand_radius_m"] = r_m2
            df_eval.loc[idx, "m2_cand_is_covered"] = bool(e <= r_m2)

            # Method 3: Landmark Geometry (with candidate P90 fallback if ineligible)
            if is_geom and pd.notna(g_scale):
                q_fact = cand_q_factors.get(c, pooled_q_factor)
                r_m3 = min(max(round(q_fact * g_scale, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)
            else:
                r_m3 = cand_radii[c]
            df_eval.loc[idx, "m3_geom_radius_m"] = r_m3
            df_eval.loc[idx, "m3_geom_is_covered"] = bool(e <= r_m3)

            # Method 4: Full Production Hierarchy via calibrator engine
            p_ent = parsed_map.get(df_eval.loc[idx, "address_id"], {})
            pred_coord = (df_eval.loc[idx, "pred_x"], df_eval.loc[idx, "pred_y"])
            prod_res = calibrate_prediction_production(
                selected_candidate_type=c,
                predicted_coord=pred_coord,
                status_tier=t,
                calibration_dict=fold_cal_dict,
                parsed_entity=p_ent,
                df_pois=df_pois,
                df_localities=df_localities,
                has_credible_doorstep=False,
                config=cfg,
            )
            r_m4 = prod_res.radius_m
            df_eval.loc[idx, "m4_prod_radius_m"] = r_m4
            df_eval.loc[idx, "m4_prod_is_covered"] = bool(e <= r_m4)
            df_eval.loc[idx, "m4_prod_method"] = prod_res.calibration_method
            df_eval.loc[idx, "m4_prod_fallback_level"] = prod_res.fallback_level
            df_eval.loc[idx, "m4_prod_fallback_reason"] = prod_res.fallback_reason or ""

        val_evaluated = df_eval.loc[val_mask]
        fold_metrics.append({
            "fold_idx": fold_idx,
            "train_sample_size": len(train_df),
            "val_sample_size": len(val_df),
            "val_accounts": val_df["account_id"].nunique(),
            "m1_tier_coverage_pct": round(float(val_evaluated["m1_tier_is_covered"].mean() * 100.0), 2),
            "m2_cand_coverage_pct": round(float(val_evaluated["m2_cand_is_covered"].mean() * 100.0), 2),
            "m3_geom_coverage_pct": round(float(val_evaluated["m3_geom_is_covered"].mean() * 100.0), 2),
            "m4_prod_coverage_pct": round(float(val_evaluated["m4_prod_is_covered"].mean() * 100.0), 2),
            "m1_tier_median_radius_m": round(float(val_evaluated["m1_tier_radius_m"].median()), 2),
            "m2_cand_median_radius_m": round(float(val_evaluated["m2_cand_radius_m"].median()), 2),
            "m3_geom_median_radius_m": round(float(val_evaluated["m3_geom_radius_m"].median()), 2),
            "m4_prod_median_radius_m": round(float(val_evaluated["m4_prod_radius_m"].median()), 2),
            "m1_tier_p90_radius_m": round(float(val_evaluated["m1_tier_radius_m"].quantile(0.90)), 2),
            "m2_cand_p90_radius_m": round(float(val_evaluated["m2_cand_radius_m"].quantile(0.90)), 2),
            "m3_geom_p90_radius_m": round(float(val_evaluated["m3_geom_radius_m"].quantile(0.90)), 2),
            "m4_prod_p90_radius_m": round(float(val_evaluated["m4_prod_radius_m"].quantile(0.90)), 2),
        })

        f_res = fold_metrics[-1]
        print(
            f"  * Fold {fold_idx}: N={len(val_df):3d} | Cov: "
            f"M1={f_res['m1_tier_coverage_pct']:5.2f}%, M2={f_res['m2_cand_coverage_pct']:5.2f}%, "
            f"M3={f_res['m3_geom_coverage_pct']:5.2f}%, M4={f_res['m4_prod_coverage_pct']:5.2f}% | "
            f"Med Rad: M1={f_res['m1_tier_median_radius_m']:6.2f}m, M4={f_res['m4_prod_median_radius_m']:6.2f}m"
        )

    # -------------------------------------------------------------------------
    # 5. Compile aggregate and comparative analysis
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Compiling aggregate metrics, paired comparisons, and export artifacts...")

    # Aggregate coverages and radii
    agg_cov = {
        "m1_tier_coverage_pct": round(float(df_eval["m1_tier_is_covered"].mean() * 100.0), 2),
        "m2_cand_coverage_pct": round(float(df_eval["m2_cand_is_covered"].mean() * 100.0), 2),
        "m3_geom_coverage_pct": round(float(df_eval["m3_geom_is_covered"].mean() * 100.0), 2),
        "m4_prod_coverage_pct": round(float(df_eval["m4_prod_is_covered"].mean() * 100.0), 2),
    }

    agg_med_r = {
        "m1_tier_median_radius_m": round(float(df_eval["m1_tier_radius_m"].median()), 2),
        "m2_cand_median_radius_m": round(float(df_eval["m2_cand_radius_m"].median()), 2),
        "m3_geom_median_radius_m": round(float(df_eval["m3_geom_radius_m"].median()), 2),
        "m4_prod_median_radius_m": round(float(df_eval["m4_prod_radius_m"].median()), 2),
    }

    agg_p90_r = {
        "m1_tier_p90_radius_m": round(float(df_eval["m1_tier_radius_m"].quantile(0.90)), 2),
        "m2_cand_p90_radius_m": round(float(df_eval["m2_cand_radius_m"].quantile(0.90)), 2),
        "m3_geom_p90_radius_m": round(float(df_eval["m3_geom_radius_m"].quantile(0.90)), 2),
        "m4_prod_p90_radius_m": round(float(df_eval["m4_prod_radius_m"].quantile(0.90)), 2),
    }

    # Production method activation breakdown
    method_counts = dict(Counter(df_eval["m4_prod_method"]))
    method_rates = {k: round(float(v / len(df_eval) * 100.0), 2) for k, v in method_counts.items()}

    # Rejection and fallback breakdown
    fb_reason_counts = dict(Counter(df_eval["m4_prod_fallback_reason"].replace("", "no_fallback")))

    # Paired subset: eligible geometry cases (N=586)
    geom_sub = df_eval[df_eval["geom_eligible"]].copy()
    geom_paired_stats = {
        "sample_size": len(geom_sub),
        "distinct_accounts": geom_sub["account_id"].nunique(),
        "m1_tier_coverage_pct": round(float(geom_sub["m1_tier_is_covered"].mean() * 100.0), 2),
        "m2_cand_coverage_pct": round(float(geom_sub["m2_cand_is_covered"].mean() * 100.0), 2),
        "m3_geom_coverage_pct": round(float(geom_sub["m3_geom_is_covered"].mean() * 100.0), 2),
        "m4_prod_coverage_pct": round(float(geom_sub["m4_prod_is_covered"].mean() * 100.0), 2),
        "m1_tier_median_radius_m": round(float(geom_sub["m1_tier_radius_m"].median()), 2),
        "m2_cand_median_radius_m": round(float(geom_sub["m2_cand_radius_m"].median()), 2),
        "m3_geom_median_radius_m": round(float(geom_sub["m3_geom_radius_m"].median()), 2),
        "m4_prod_median_radius_m": round(float(geom_sub["m4_prod_radius_m"].median()), 2),
        "m1_tier_p90_radius_m": round(float(geom_sub["m1_tier_radius_m"].quantile(0.90)), 2),
        "m2_cand_p90_radius_m": round(float(geom_sub["m2_cand_radius_m"].quantile(0.90)), 2),
        "m3_geom_p90_radius_m": round(float(geom_sub["m3_geom_radius_m"].quantile(0.90)), 2),
        "m4_prod_p90_radius_m": round(float(geom_sub["m4_prod_radius_m"].quantile(0.90)), 2),
    }

    # Per-candidate statistics under Production Hierarchy (Method 4)
    per_candidate_stats = {}
    for c_type in ["C1", "C2", "C3", "C4", "C5"]:
        sub = df_eval[df_eval["selected_candidate_type"] == c_type]
        if len(sub) > 0:
            errs = sub["error_m"]
            per_candidate_stats[c_type] = {
                "candidate_type": c_type,
                "is_estimable": True,
                "sample_size": len(sub),
                "distinct_addresses": sub["address_id"].nunique(),
                "distinct_accounts": sub["account_id"].nunique(),
                "error_median_m": round(float(errs.median()), 2),
                "error_p90_m": round(float(errs.quantile(0.90)), 2),
                "m1_tier_coverage_pct": round(float(sub["m1_tier_is_covered"].mean() * 100.0), 2),
                "m2_cand_coverage_pct": round(float(sub["m2_cand_is_covered"].mean() * 100.0), 2),
                "m4_prod_coverage_pct": round(float(sub["m4_prod_is_covered"].mean() * 100.0), 2),
                "m1_tier_median_radius_m": round(float(sub["m1_tier_radius_m"].median()), 2),
                "m4_prod_median_radius_m": round(float(sub["m4_prod_radius_m"].median()), 2),
                "m1_tier_p90_radius_m": round(float(sub["m1_tier_radius_m"].quantile(0.90)), 2),
                "m4_prod_p90_radius_m": round(float(sub["m4_prod_radius_m"].quantile(0.90)), 2),
                "method_breakdown": dict(Counter(sub["m4_prod_method"])),
            }
        else:
            per_candidate_stats[c_type] = {
                "candidate_type": c_type,
                "is_estimable": False,
                "sample_size": 0,
                "explanation": "Not estimable under SCV protocol: target doorstep visits are masked out by design to simulate cold start.",
            }

    # Fold variability statistics
    fold_cov_m4 = [f["m4_prod_coverage_pct"] for f in fold_metrics]
    fold_med_m4 = [f["m4_prod_median_radius_m"] for f in fold_metrics]

    fold_variability = {
        "m4_coverage_mean_pct": round(float(np.mean(fold_cov_m4)), 2),
        "m4_coverage_std_pct": round(float(np.std(fold_cov_m4)), 2),
        "m4_coverage_min_pct": round(float(np.min(fold_cov_m4)), 2),
        "m4_coverage_max_pct": round(float(np.max(fold_cov_m4)), 2),
        "m4_median_radius_mean_m": round(float(np.mean(fold_med_m4)), 2),
        "m4_median_radius_std_m": round(float(np.std(fold_med_m4)), 2),
    }

    # Complete summary dictionary
    summary = {
        "evaluation_name": "NeuroGeo PS3 Production Calibration Validation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_quantile": 0.90,
        "sample_size": len(df_eval),
        "account_count": df_eval["account_id"].nunique(),
        "folds_count": 5,
        "validation_strategy": "5-Fold Grouped Cross-Validation by account_id",
        "benchmark_addresses_excluded_count": len(bench_address_ids),
        "aggregate_coverage": agg_cov,
        "aggregate_median_radius_m": agg_med_r,
        "aggregate_p90_radius_m": agg_p90_r,
        "production_method_activation_counts": method_counts,
        "production_method_activation_rates_pct": method_rates,
        "production_fallback_reasons": fb_reason_counts,
        "eligible_geometry_paired_comparison": geom_paired_stats,
        "per_candidate_statistics": per_candidate_stats,
        "fold_level_metrics": fold_metrics,
        "fold_variability": fold_variability,
        "c2_direct_doorstep_status": {
            "is_estimable_in_scv": False,
            "doorstep_verified_operational_radius_m": 35.0,
            "explanation": "Target doorstep visits are masked in SCV by design. In production, C2 is strictly DOORSTEP_VERIFIED (35.0m).",
        },
    }

    # Export summary JSON
    summary_path = output_dir / "production_calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  * Exported summary: {summary_path.relative_to(_repo_root)}")

    # Export folds JSON
    folds_path = output_dir / "production_calibration_folds.json"
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"  * Exported fold metrics: {folds_path.relative_to(_repo_root)}")

    # Export full predictions CSV
    preds_path = output_dir / "production_calibration_predictions.csv"
    export_cols = [
        "address_id", "account_id", "fold", "town_id", "locality_id",
        "selected_candidate_type", "status_tier",
        "pred_x", "pred_y", "gt_x", "gt_y", "error_m",
        "m1_tier_radius_m", "m1_tier_is_covered",
        "m2_cand_radius_m", "m2_cand_is_covered",
        "m3_geom_radius_m", "m3_geom_is_covered",
        "m4_prod_radius_m", "m4_prod_is_covered",
        "m4_prod_method", "m4_prod_fallback_level", "m4_prod_fallback_reason",
        "geom_eligible", "geom_scale_m", "d_min_m", "unmentioned_poi_id", "geom_rejection_reason",
    ]
    df_eval[export_cols].to_csv(preds_path, index=False)
    print(f"  * Exported predictions: {preds_path.relative_to(_repo_root)}")

    # Export paired comparison CSV (Method 1 vs Method 4)
    df_eval["m4_minus_m1_radius_m"] = np.round(df_eval["m4_prod_radius_m"] - df_eval["m1_tier_radius_m"], 2)
    paired_cols = [
        "address_id", "account_id", "selected_candidate_type", "status_tier", "error_m",
        "m1_tier_radius_m", "m1_tier_is_covered",
        "m4_prod_radius_m", "m4_prod_is_covered", "m4_minus_m1_radius_m",
        "m4_prod_method", "m4_prod_fallback_level",
    ]
    paired_path = output_dir / "production_calibration_paired_comparison.csv"
    df_eval[paired_cols].to_csv(paired_path, index=False)
    print(f"  * Exported paired comparison: {paired_path.relative_to(_repo_root)}")

    return summary


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    run_production_calibration_validation(out_dir)
