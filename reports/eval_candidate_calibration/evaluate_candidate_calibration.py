"""Isolated Diagnostic Evaluator: Candidate-wise P90 vs Evidence-Tier Calibration.

Authority: Prompt 28 requirements.
Safety Constraints:
- Purely diagnostic; does not modify production code or authoritative documentation.
- Excludes all 100 official benchmark addresses from calibration fitting.
- Never accesses or evaluates the official 15-address test split.
- Uses 5-fold grouped cross-validation by account_id on clean SCV episodes.
- Outputs exclusively to reports/eval_candidate_calibration/.
"""

import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure src is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from neurogeo.calibrator import assign_status_tier
from neurogeo.candidate_ranker import (
    extract_candidate_features,
    generate_candidates,
    rank_candidates,
)
from neurogeo.config import PATHS


def run_candidate_calibration_diagnostic(
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("NEUROGEO PS3: CANDIDATE-WISE P90 VS EVIDENCE-TIER CALIBRATION DIAGNOSTIC")
    print("=" * 75)
    print(f"Target Output Directory: {output_dir}")

    # 1. Ingest input artifacts
    print("\n[Step 1/5] Ingesting source datasets and SCV artifacts...")
    df_scv = pd.read_csv(PATHS.scv_results_csv)
    df_addr = pd.read_csv(PATHS.addresses_csv)
    df_surv = pd.read_csv(PATHS.surveyed_addresses_csv)
    df_parsed = pd.read_parquet(PATHS.parsed_addresses_parquet)
    df_base = pd.read_csv(PATHS.baseline_geocodes_csv)
    df_localities = pd.read_csv(PATHS.localities_csv)
    df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
    weights = json.loads(PATHS.reranker_weights_json.read_text())["weights"]

    # Exclude 100 benchmark addresses
    bench_address_ids = set(df_surv["address_id"])
    assert len(bench_address_ids) == 100, f"Expected 100 benchmark addresses, got {len(bench_address_ids)}"

    # Map account_id
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

    # 2. Reconstruct exact production inference predictions for SCV episodes
    print("\n[Step 2/5] Simulating production inference path under cold start (medoid hidden)...")
    episodes: List[Dict[str, Any]] = []

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

        episodes.append({
            "address_id": aid,
            "account_id": str(row["account_id"]),
            "town_id": str(row["town_id"]),
            "locality_id": str(row["locality_id_resolved"]) if pd.notna(row.get("locality_id_resolved")) else "",
            "street_normalized": str(row.get("street_normalized", "")),
            "selected_candidate_type": sel.candidate_type,
            "status_tier": tier.value,
            "pred_x": round(float(sel.x), 2),
            "pred_y": round(float(sel.y), 2),
            "gt_x": round(gt_x, 2),
            "gt_y": round(gt_y, 2),
            "error_m": round(err_m, 2),
        })

    df_eval = pd.DataFrame(episodes)
    print(f"  * Predictions successfully reconstructed: {len(df_eval)}")
    print(f"  * Candidate selection breakdown: {dict(Counter(df_eval['selected_candidate_type']))}")
    print(f"  * Status tier breakdown: {dict(Counter(df_eval['status_tier']))}")

    # 3. Partition into 5 grouped folds by account_id
    print("\n[Step 3/5] Partitioning into 5-fold grouped cross-validation by account_id...")
    gkf = GroupKFold(n_splits=5)
    df_eval["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_eval, groups=df_eval["account_id"])):
        df_eval.loc[val_idx, "fold"] = fold_idx

    # Initialize evaluation columns
    df_eval["tier_radius_m"] = np.nan
    df_eval["cand_radius_m"] = np.nan
    df_eval["tier_is_covered"] = False
    df_eval["cand_is_covered"] = False
    df_eval["cand_calibration_method"] = ""
    df_eval["cand_fallback_reason"] = ""

    fold_metrics = []

    # 4. Execute fold-by-fold cross-validation
    print("\n[Step 4/5] Executing 5-fold paired calibration comparison...")
    for fold_idx in range(5):
        train_mask = df_eval["fold"] != fold_idx
        val_mask = df_eval["fold"] == fold_idx
        train_df = df_eval[train_mask]
        val_df = df_eval[val_mask]

        # Approach A: Evidence-tier calibration fitted strictly on train_df
        tier_radii = {}
        tier_counts = {}
        tier_fallbacks = {}
        global_p90 = float(np.quantile(train_df["error_m"], 0.90))

        for t_name in ["STREET_RESOLVED", "LOCALITY_BOUNDED", "BASELINE_FALLBACK", "DOORSTEP_VERIFIED"]:
            t_errs = train_df[train_df["status_tier"] == t_name]["error_m"]
            tier_counts[t_name] = len(t_errs)
            if len(t_errs) >= 15:
                r = float(np.quantile(t_errs, 0.90))
                tier_fallbacks[t_name] = "tier_direct"
            else:
                r = global_p90
                tier_fallbacks[t_name] = "global_p90"
            tier_radii[t_name] = min(max(round(r, 2), 15.0), 2500.0)

        # Approach B: Candidate-wise P90 calibration fitted strictly on train_df
        cand_radii = {}
        cand_counts = {}
        cand_method = {}
        cand_reason = {}

        for c_type in ["C1", "C2", "C3", "C4", "C5"]:
            c_errs = train_df[train_df["selected_candidate_type"] == c_type]["error_m"]
            cand_counts[c_type] = len(c_errs)
            if len(c_errs) >= 15:
                r = float(np.quantile(c_errs, 0.90))
                cand_radii[c_type] = min(max(round(r, 2), 15.0), 2500.0)
                cand_method[c_type] = "candidate_p90"
                cand_reason[c_type] = "sufficient_candidate_samples"
            else:
                default_tier = {
                    "C1": "BASELINE_FALLBACK",
                    "C2": "DOORSTEP_VERIFIED",
                    "C3": "STREET_RESOLVED",
                    "C4": "STREET_RESOLVED",
                    "C5": "LOCALITY_BOUNDED",
                }.get(c_type, "LOCALITY_BOUNDED")
                cand_radii[c_type] = tier_radii[default_tier]
                cand_method[c_type] = "tier_fallback"
                cand_reason[c_type] = "insufficient_candidate_samples"

        # Apply to validation fold
        for idx in val_df.index:
            t = df_eval.loc[idx, "status_tier"]
            c = df_eval.loc[idx, "selected_candidate_type"]
            e = df_eval.loc[idx, "error_m"]

            r_tier = tier_radii[t]
            r_cand = cand_radii[c]

            df_eval.loc[idx, "tier_radius_m"] = r_tier
            df_eval.loc[idx, "cand_radius_m"] = r_cand
            df_eval.loc[idx, "tier_is_covered"] = bool(e <= r_tier)
            df_eval.loc[idx, "cand_is_covered"] = bool(e <= r_cand)
            df_eval.loc[idx, "cand_calibration_method"] = cand_method[c]
            df_eval.loc[idx, "cand_fallback_reason"] = cand_reason[c]

        val_evaluated = df_eval.loc[val_mask]
        fold_metrics.append({
            "fold_idx": fold_idx,
            "train_sample_size": len(train_df),
            "val_sample_size": len(val_df),
            "val_accounts": val_df["account_id"].nunique(),
            "tier_coverage_pct": round(float(val_evaluated["tier_is_covered"].mean() * 100.0), 2),
            "cand_coverage_pct": round(float(val_evaluated["cand_is_covered"].mean() * 100.0), 2),
            "tier_median_radius_m": round(float(val_evaluated["tier_radius_m"].median()), 2),
            "cand_median_radius_m": round(float(val_evaluated["cand_radius_m"].median()), 2),
            "tier_p90_radius_m": round(float(val_evaluated["tier_radius_m"].quantile(0.90)), 2),
            "cand_p90_radius_m": round(float(val_evaluated["cand_radius_m"].quantile(0.90)), 2),
            "tier_fitted_radii": tier_radii,
            "cand_fitted_radii": cand_radii,
            "cand_train_counts": cand_counts,
        })
        print(f"  * Fold {fold_idx}: N_val={len(val_df):3d} | Cov: Tier={fold_metrics[-1]['tier_coverage_pct']:5.2f}%, Cand={fold_metrics[-1]['cand_coverage_pct']:5.2f}% | Med Rad: Tier={fold_metrics[-1]['tier_median_radius_m']:6.2f}m, Cand={fold_metrics[-1]['cand_median_radius_m']:6.2f}m")

    # 5. Compile aggregate and per-candidate analysis
    print("\n[Step 5/5] Compiling aggregate and per-candidate diagnostic summaries...")
    df_eval["radius_difference_m"] = np.round(df_eval["cand_radius_m"] - df_eval["tier_radius_m"], 2)

    cov_tier_agg = round(float(df_eval["tier_is_covered"].mean() * 100.0), 2)
    cov_cand_agg = round(float(df_eval["cand_is_covered"].mean() * 100.0), 2)
    med_r_tier_agg = round(float(df_eval["tier_radius_m"].median()), 2)
    med_r_cand_agg = round(float(df_eval["cand_radius_m"].median()), 2)
    p90_r_tier_agg = round(float(df_eval["tier_radius_m"].quantile(0.90)), 2)
    p90_r_cand_agg = round(float(df_eval["cand_radius_m"].quantile(0.90)), 2)

    # Material changes
    diff_abs = df_eval["radius_difference_m"].abs()
    mat_change_15m = int((diff_abs >= 15.0).sum())
    mat_change_50m = int((diff_abs >= 50.0).sum())

    # Per-candidate statistics
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
                "error_p75_m": round(float(errs.quantile(0.75)), 2),
                "error_p90_m": round(float(errs.quantile(0.90)), 2),
                "error_p95_m": round(float(errs.quantile(0.95)), 2),
                "error_max_m": round(float(errs.max()), 2),
                "tier_coverage_pct": round(float(sub["tier_is_covered"].mean() * 100.0), 2),
                "cand_coverage_pct": round(float(sub["cand_is_covered"].mean() * 100.0), 2),
                "tier_median_radius_m": round(float(sub["tier_radius_m"].median()), 2),
                "cand_median_radius_m": round(float(sub["cand_radius_m"].median()), 2),
                "tier_p90_radius_m": round(float(sub["tier_radius_m"].quantile(0.90)), 2),
                "cand_p90_radius_m": round(float(sub["cand_radius_m"].quantile(0.90)), 2),
                "cand_specific_rate_pct": round(float((sub["cand_calibration_method"] == "candidate_p90").mean() * 100.0), 2),
                "tier_fallback_rate_pct": round(float((sub["cand_calibration_method"] == "tier_fallback").mean() * 100.0), 2),
            }
        else:
            per_candidate_stats[c_type] = {
                "candidate_type": c_type,
                "is_estimable": False,
                "sample_size": 0,
                "distinct_addresses": 0,
                "distinct_accounts": 0,
                "explanation": "Not estimable under SCV protocol: target doorstep visits are masked out by design to simulate cold start, so C2 direct medoid is never available during leave-one-out evaluation.",
            }

    # Summary artifact
    summary = {
        "diagnostic_name": "Candidate-wise P90 vs Evidence-Tier Calibration Diagnostic",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_quantile": 0.90,
        "sample_size": len(df_eval),
        "account_count": df_eval["account_id"].nunique(),
        "folds_count": 5,
        "validation_strategy": "5-Fold Grouped Cross-Validation by account_id",
        "benchmark_addresses_excluded_count": len(bench_address_ids),
        "aggregate_comparison": {
            "tier_coverage_pct": cov_tier_agg,
            "cand_coverage_pct": cov_cand_agg,
            "coverage_difference_pct": round(cov_cand_agg - cov_tier_agg, 2),
            "tier_median_radius_m": med_r_tier_agg,
            "cand_median_radius_m": med_r_cand_agg,
            "median_radius_difference_m": round(med_r_cand_agg - med_r_tier_agg, 2),
            "tier_p90_radius_m": p90_r_tier_agg,
            "cand_p90_radius_m": p90_r_cand_agg,
            "p90_radius_difference_m": round(p90_r_cand_agg - p90_r_tier_agg, 2),
            "predictions_with_material_change_15m": mat_change_15m,
            "predictions_with_material_change_50m": mat_change_50m,
            "material_change_rate_pct": round(float(mat_change_15m / len(df_eval) * 100.0), 2),
        },
        "per_candidate_statistics": per_candidate_stats,
        "fold_level_metrics": fold_metrics,
        "c2_protocol_status": {
            "is_estimable_in_scv": False,
            "reason": "SCV hides target direct doorstep visits to simulate cold start. In production, C2 is strictly assigned DOORSTEP_VERIFIED with direct-visit evidence.",
        },
    }

    # Export outputs
    summary_path = output_dir / "candidate_calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  * Exported summary: {summary_path.relative_to(_repo_root)}")

    folds_path = output_dir / "candidate_calibration_folds.json"
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"  * Exported fold metrics: {folds_path.relative_to(_repo_root)}")

    preds_path = output_dir / "candidate_calibration_predictions.csv"
    export_cols = [
        "address_id", "account_id", "fold", "selected_candidate_type", "status_tier",
        "pred_x", "pred_y", "gt_x", "gt_y", "error_m",
        "tier_radius_m", "cand_radius_m", "radius_difference_m",
        "tier_is_covered", "cand_is_covered",
        "cand_calibration_method", "cand_fallback_reason",
    ]
    df_eval[export_cols].to_csv(preds_path, index=False)
    print(f"  * Exported predictions: {preds_path.relative_to(_repo_root)}")

    return summary


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    run_candidate_calibration_diagnostic(out_dir)
