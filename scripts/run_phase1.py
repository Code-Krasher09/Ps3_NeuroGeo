"""Phase 1 Runner: Operational Visit Sanitizer & Trajectory Miner.

Authority: NEUROGEO_IMPLEMENTATION_PLAN.md Phase 1, NEUROGEO_MODEL_SPEC.md, and NEUROGEO_DATA_CONTRACT.md.
Usage:
    python scripts/run_phase1.py
"""

from pathlib import Path
import sys
import json
import time

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from neurogeo.config import PATHS
from neurogeo.sanitizer import sanitize_visits
from neurogeo.trajectory_miner import (
    extract_trajectory_features_batch,
    compute_all_address_doorstep_medoids,
)
from neurogeo.evaluation import load_benchmark_ground_truth, compute_euclidean_error, compute_error_metrics


def run_phase1() -> int:
    """Execute all Phase 1 transformations and generate Phase 1 artifacts."""
    print("=" * 70)
    print("NEUROGEO-FIELD :: PHASE 1 OPERATIONAL SANITIZER & TRAJECTORY MINER")
    print("=" * 70)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Loading field visits and GPS trajectory breadcrumbs...")
    df_visits = pd.read_csv(PATHS.field_visits_csv)
    df_gps = pd.read_csv(PATHS.visit_gps_points_csv)
    df_addresses = pd.read_csv(PATHS.addresses_csv)

    print(f"  + Field Visits Count : {len(df_visits)} visits across {df_visits['address_id'].nunique()} unique addresses")
    print(f"  + GPS Breadcrumbs    : {len(df_gps)} points across {df_gps['visit_id'].nunique()} unique visits")
    print(f"  + Addresses Ingested : {len(df_addresses)} master records")

    # -------------------------------------------------------------------------
    # 2. TRAJECTORY MINING & DWELL MEDOIDS
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Mining GPS trajectories, tortuosity, and dwell medoids...")
    t0 = time.time()
    df_traj = extract_trajectory_features_batch(df_gps, df_visits)
    print(f"  => Extracted trajectory features for {len(df_traj)} visits in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 3. VISIT SANITIZATION & INTEGRITY SCORING
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Sanitizing operational visits & computing I(v), S(v)...")
    df_sanitized = sanitize_visits(df_visits, df_trajectory_metrics=df_traj)
    # Merge dwell medoid coordinates into sanitized visits
    df_sanitized = df_sanitized.merge(
        df_traj[["visit_id", "dwell_medoid_x", "dwell_medoid_y", "tortuosity", "radius_of_gyration_m", "is_trajectory_guarded"]],
        on="visit_id",
        how="left",
    )

    clean_visits_cnt = int(df_sanitized["is_credible_doorstep"].sum())
    shifted_cnt = int(df_sanitized["is_address_state_shifted"].sum())
    search_cnt = int(df_sanitized["is_untraceable_search"].sum())
    high_integrity_cnt = int((df_sanitized["visit_integrity_score"] >= 0.70).sum())
    compromised_cnt = int((df_sanitized["visit_integrity_score"] < 0.10).sum())

    print(f"  + High Integrity Visits (I >= 0.70) : {high_integrity_cnt} ({high_integrity_cnt / len(df_sanitized) * 100:.1f}%)")
    print(f"  + Compromised Visits   (I < 0.10)  : {compromised_cnt} ({compromised_cnt / len(df_sanitized) * 100:.1f}%)")
    print(f"  + Credible Doorstep Observations    : {clean_visits_cnt}")
    print(f"  + Shifted-Borrower Observations     : {shifted_cnt} (address-state evidence)")
    print(f"  + Untraceable Search Loops          : {search_cnt} (negative search evidence)")

    # -------------------------------------------------------------------------
    # 4. ADDRESS-LEVEL DOORSTEP MEDOIDS
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Computing address-level doorstep medoids...")
    df_medoids = compute_all_address_doorstep_medoids(df_sanitized, min_integrity=0.50)
    credible_medoids_cnt = int(df_medoids["has_credible_doorstep"].sum())
    print(f"  + Total Visited Addresses with Medoids : {len(df_medoids)}")
    print(f"  + Addresses with Credible Doorsteps    : {credible_medoids_cnt} ({credible_medoids_cnt / len(df_medoids) * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # 5. MODEL 1 MILESTONE: FIELD CHECK-IN BASELINE ON BENCHMARK
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Evaluating Model 1 Milestone (Doorstep Medoid on Visited Benchmark)...")
    df_bench = load_benchmark_ground_truth(PATHS)
    visited_bench = df_bench[df_bench["is_visited"]].copy()

    # Join with computed doorstep medoids
    visited_eval = visited_bench.merge(
        df_medoids[["address_id", "doorstep_medoid_x", "doorstep_medoid_y", "has_credible_doorstep"]],
        on="address_id",
        how="inner",
    )

    medoid_errors = compute_euclidean_error(
        visited_eval["doorstep_medoid_x"],
        visited_eval["doorstep_medoid_y"],
        visited_eval["surveyed_x"],
        visited_eval["surveyed_y"],
    )
    visited_eval["model1_error_m"] = medoid_errors
    m1_metrics = compute_error_metrics(medoid_errors)

    baseline_visited_errors = visited_bench["baseline_error_m"] if "baseline_error_m" in visited_bench.columns else compute_euclidean_error(
        visited_bench["geocoder_x"], visited_bench["geocoder_y"], visited_bench["surveyed_x"], visited_bench["surveyed_y"]
    )
    m0_visited_metrics = compute_error_metrics(baseline_visited_errors)

    print(f"\n  MODEL 1 MILESTONE RESULTS (N = {len(visited_eval)} visited benchmark addresses):")
    print(f"  -------------------------------------------------------------")
    print(f"  Commercial Geocoder Median Error (M0) : {m0_visited_metrics.median_error_m:6.2f} m")
    print(f"  Model 1 Doorstep Medoid Median Error  : {m1_metrics.median_error_m:6.2f} m  (Target: <= 25.0m)")
    print(f"  Model 1 Doorstep Medoid P90 Error     : {m1_metrics.p90_error_m:6.2f} m")
    print(f"  Error Reduction Relative to Baseline  : {(1.0 - m1_metrics.median_error_m / m0_visited_metrics.median_error_m) * 100:6.1f} %")
    print(f"  -------------------------------------------------------------")

    if m1_metrics.median_error_m > 25.0:
        print(f"[-] ERROR: Model 1 milestone failed: median error {m1_metrics.median_error_m:.2f}m > 25.0m")
        return 1
    print(f"  => Model 1 Milestone PASSED: {m1_metrics.median_error_m:.2f}m <= 25.0m target.")

    # -------------------------------------------------------------------------
    # EXPORT ARTIFACTS
    # -------------------------------------------------------------------------
    # 1. Summary JSON report
    summary_report = {
        "phase": 1,
        "phase_name": "Operational Visit Sanitizer & Trajectory Miner",
        "timestamp_seconds": round(time.time() - t_start, 2),
        "dataset_statistics": {
            "total_visits": int(len(df_sanitized)),
            "unique_addresses_visited": int(df_sanitized["address_id"].nunique()),
            "total_gps_breadcrumbs": int(len(df_gps)),
            "unique_visits_with_gps": int(df_gps["visit_id"].nunique()),
        },
        "integrity_distributions": {
            "mean_integrity_score": round(float(df_sanitized["visit_integrity_score"].mean()), 4),
            "median_integrity_score": round(float(df_sanitized["visit_integrity_score"].median()), 4),
            "high_integrity_visits_count": high_integrity_cnt,
            "compromised_visits_count": compromised_cnt,
            "duplicate_photo_groups_count": int((df_sanitized["photo_collision_count"] > 1).sum()),
        },
        "evidence_distributions": {
            "credible_doorstep_count": clean_visits_cnt,
            "shifted_borrower_count": shifted_cnt,
            "untraceable_search_count": search_cnt,
            "outcome_counts": df_sanitized["outcome"].value_counts().to_dict(),
        },
        "trajectory_distributions": {
            "mean_tortuosity": round(float(df_traj["tortuosity"].mean()), 3),
            "median_tortuosity": round(float(df_traj["tortuosity"].median()), 3),
            "guarded_trajectories_count": int(df_traj["is_trajectory_guarded"].sum()),
            "mean_radius_of_gyration_m": round(float(df_traj["radius_of_gyration_m"].mean()), 2),
        },
        "model_1_benchmark_milestone": {
            "visited_benchmark_count": int(len(visited_eval)),
            "commercial_baseline_median_m": round(m0_visited_metrics.median_error_m, 2),
            "model_1_medoid_median_m": round(m1_metrics.median_error_m, 2),
            "model_1_medoid_p90_m": round(m1_metrics.p90_error_m, 2),
            "median_error_reduction_pct": round(float((1.0 - m1_metrics.median_error_m / m0_visited_metrics.median_error_m) * 100), 2),
            "milestone_passed": bool(m1_metrics.median_error_m <= 25.0),
        },
        "artifacts_generated": {
            "summary_json": "reports/metrics/phase1_visit_features_summary.json",
            "visit_features_parquet": "reports/metrics/visit_features.parquet",
            "address_doorstep_medoids_parquet": "reports/metrics/address_doorstep_medoids.parquet",
        },
    }

    out_json = PATHS.metrics_dir / "phase1_visit_features_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # 2. Tabular Parquet artifacts with fit_scope tracking
    out_visits_parquet = PATHS.metrics_dir / "visit_features.parquet"
    df_sanitized.to_parquet(out_visits_parquet, index=False)

    out_medoids_parquet = PATHS.metrics_dir / "address_doorstep_medoids.parquet"
    df_medoids.to_parquet(out_medoids_parquet, index=False)

    print(f"\n  => Saved summary report: {out_json.relative_to(REPO_ROOT)}")
    print(f"  => Saved visit features: {out_visits_parquet.relative_to(REPO_ROOT)} ({out_visits_parquet.stat().st_size / 1024:.1f} KB)")
    print(f"  => Saved address medoids: {out_medoids_parquet.relative_to(REPO_ROOT)} ({out_medoids_parquet.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 70)
    print("PHASE 1 GATE COMPLETED SUCCESSFULLY: Visit sanitization & miner ready.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_phase1())
