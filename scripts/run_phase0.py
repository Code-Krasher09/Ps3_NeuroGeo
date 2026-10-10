"""Phase 0 Runner: Official Baseline Reproduction & Zero-Leakage Evaluation Harness.

Authority: NEUROGEO_IMPLEMENTATION_PLAN.md Phase 0 and NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md.
Usage:
    python scripts/run_phase0.py
"""

from pathlib import Path
import sys

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from neurogeo.config import PATHS
from neurogeo.evaluation import evaluate_commercial_baseline, load_benchmark_ground_truth
from neurogeo.leakage import (
    compute_isolation_closure,
    validate_fit_scope,
    assert_no_leakage,
)


def run_phase0() -> int:
    """Execute all Phase 0 verifications and generate baseline_metrics.json."""
    print("=" * 70)
    print("NEUROGEO-FIELD :: PHASE 0 VERIFICATION & BASELINE REPRODUCTION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. DATA CONTRACT VALIDATION
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Validating source data contracts and table integrity...")
    required_tables = [
        ("addresses.csv", PATHS.addresses_csv),
        ("baseline_geocodes.csv", PATHS.baseline_geocodes_csv),
        ("field_visits.csv", PATHS.field_visits_csv),
        ("visit_gps_points.csv", PATHS.visit_gps_points_csv),
        ("landmarks_poi.csv", PATHS.landmarks_poi_csv),
        ("localities.csv", PATHS.localities_csv),
        ("towns.csv", PATHS.towns_csv),
        ("surveyed_addresses.csv", PATHS.surveyed_addresses_csv),
        ("splits.csv", PATHS.splits_csv),
    ]

    for name, path in required_tables:
        if not path.exists():
            print(f"[-] ERROR: Required table missing: {name} at {path}")
            return 1
        size_kb = path.stat().st_size / 1024
        print(f"  + {name:25s} : OK ({size_kb:.1f} KB)")

    # Validate row counts and keys
    df_addr = pd.read_csv(PATHS.addresses_csv)
    df_base = pd.read_csv(PATHS.baseline_geocodes_csv)
    df_surv = pd.read_csv(PATHS.surveyed_addresses_csv)
    df_visits = pd.read_csv(PATHS.field_visits_csv)

    if len(df_addr) != 3117 or not df_addr["address_id"].is_unique:
        print(f"[-] ERROR: addresses.csv contract violation: len={len(df_addr)}")
        return 1
    if len(df_base) != 2880 or not df_base["address_id"].is_unique:
        print(f"[-] ERROR: baseline_geocodes.csv contract violation: len={len(df_base)}")
        return 1
    if len(df_surv) != 100 or not df_surv["address_id"].is_unique:
        print(f"[-] ERROR: surveyed_addresses.csv contract violation: len={len(df_surv)}")
        return 1
    if len(df_visits) != 5578 or not df_visits["visit_id"].is_unique:
        print(f"[-] ERROR: field_visits.csv contract violation: len={len(df_visits)}")
        return 1

    print("  => All primary keys, row counts, and schema constraints verified.")

    # -------------------------------------------------------------------------
    # 2. BENCHMARK PARTITION INTEGRITY
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Validating official 100-address benchmark partition...")
    df_bench = load_benchmark_ground_truth(PATHS)
    
    train_count = int((df_bench["split"] == "train").sum())
    val_count = int((df_bench["split"].isin(["validation", "val"])).sum())
    test_count = int((df_bench["split"] == "test").sum())
    visited_count = int(df_bench["is_visited"].sum())
    cold_start_count = int((~df_bench["is_visited"]).sum())

    print(f"  + Benchmark Count  : {len(df_bench)} addresses")
    print(f"  + Train Split      : {train_count} (expected 66)")
    print(f"  + Validation Split : {val_count} (expected 19)")
    print(f"  + Test Split       : {test_count} (expected 15)")
    print(f"  + Visited Subgroup : {visited_count} (expected 49)")
    print(f"  + Cold-Start Subgrp: {cold_start_count} (expected 51)")

    if (train_count, val_count, test_count) != (66, 19, 15):
        print(f"[-] ERROR: Benchmark partition mismatch: ({train_count}, {val_count}, {test_count})")
        return 1

    train_ids = set(df_bench[df_bench["split"] == "train"]["address_id"])
    val_ids = set(df_bench[df_bench["split"].isin(["validation", "val"])]["address_id"])
    test_ids = set(df_bench[df_bench["split"] == "test"]["address_id"])
    if train_ids.intersection(val_ids) or train_ids.intersection(test_ids) or val_ids.intersection(test_ids):
        print("[-] ERROR: Split overlap detected across benchmark partitions!")
        return 1
    print("  => Zero overlap verified across train, validation, and test splits.")

    # -------------------------------------------------------------------------
    # 3. LEAKAGE ISOLATION HARNESS CHECKS
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Testing zero-leakage isolation boundary and fit_scope...")
    sample_addr = df_bench["address_id"].iloc[0]
    closure = compute_isolation_closure(sample_addr, df_visits, df_addr)
    filtered_visits = closure.filter_visits(df_visits)
    filtered_addr = closure.filter_addresses(df_addr)

    assert sample_addr not in set(filtered_visits["address_id"])
    assert sample_addr not in set(filtered_addr["address_id"])

    # Fit-scope validation check
    assert validate_fit_scope("visit_features", "unsupervised_all")
    assert validate_fit_scope("street_centroids", "train_fold_0")
    assert validate_fit_scope("calibration_quantiles", "calibration_oof")
    assert not validate_fit_scope("calibration_quantiles", "val_set_direct")

    # Leakage assertion check
    assert_no_leakage(sample_addr, filtered_visits)
    print(f"  + Closure({sample_addr}) purged {len(closure.purged_visit_ids)} visits cleanly.")
    print("  + Fit-scope provenance validators verified.")
    print("  => Zero-leakage harness functional and contract-compliant.")

    # -------------------------------------------------------------------------
    # 4. OFFICIAL COMMERCIAL BASELINE REPRODUCTION
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Reproducing official commercial baseline metrics...")
    report_file = PATHS.metrics_dir / "baseline_metrics.json"
    report = evaluate_commercial_baseline(PATHS, write_report=True, output_path=report_file)

    overall = report["overall"]
    med = overall["median_error_m"]
    p90 = overall["p90_error_m"]
    max_e = overall["max_error_m"]
    mean_e = overall["mean_error_m"]

    print(f"\n  COMMERCIAL BASELINE PERFORMANCE (N = {report['benchmark_count']}):")
    print(f"  -------------------------------------------------------------")
    print(f"  Median Error (P50) : {med:8.2f} m  (Documented target: 376.4m)")
    print(f"  90th Pct Error (P90): {p90:8.2f} m  (Documented target: 839.2m)")
    print(f"  Mean Error         : {mean_e:8.2f} m")
    print(f"  Max Error          : {max_e:8.2f} m  (Documented target: ~4.8km)")
    print(f"  -------------------------------------------------------------")

    print("\n  BREAKDOWN BY PRECISION TIER:")
    for prec, stats in report["by_precision"].items():
        print(f"    - {prec:12s} (N={stats['count']:2d}): Median = {stats['median_error_m']:6.1f}m, P90 = {stats['p90_error_m']:6.1f}m")

    print("\n  BREAKDOWN BY OPERATIONAL HISTORY:")
    for status, stats in report["by_visited_status"].items():
        print(f"    - {status:20s} (N={stats['count']:2d}): Median = {stats['median_error_m']:6.1f}m, P90 = {stats['p90_error_m']:6.1f}m")

    # Tolerance verification
    tol_check = report["reproducibility"]["tolerance_check"]
    if not tol_check["reproduction_verified"]:
        print(f"\n[-] ERROR: Baseline reproduction failed tolerance check: {tol_check}")
        return 1

    print(f"\n  => Tolerance check passed: actual median {med:.2f}m is within ±0.5m of 376.4m.")
    print(f"  => Metrics report saved to: {report_file.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    print("PHASE 0 GATE COMPLETED SUCCESSFULLY: Evaluation harness ready.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_phase0())
