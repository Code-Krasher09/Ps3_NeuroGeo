"""Phase 2 Runner: Multilingual Address Parser & Locality-Bounded POI Resolver.

Authority: PS3_FINAL_ARCHITECTURE.md Section 2, PS3_DATA_CONTRACT.md Section 2 & 3,
and PS3_IMPLEMENTATION_PLAN.md Phase 2.
"""

from typing import Any, Dict
import json
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from neurogeo.address_parser import parse_addresses_batch
from neurogeo.config import DEFAULT_CONFIG, PATHS
from neurogeo.poi_resolver import _compute_poi_locality_assignments


def main() -> None:
    print("=" * 70)
    print("NeuroGeo-Field: Phase 2 Execution")
    print("Multilingual Script-Aware Address Parser & Locality-Bounded POI Resolver")
    print("=" * 70)
    print(f"Repository Root: {PATHS.repo_root.name}")
    print(f"Outputs Directory: {PATHS.metrics_dir.relative_to(PATHS.repo_root)}")

    # 1. Load Input Tables
    t0 = time.time()
    print("\n[Step 1/5] Ingesting address and geospatial reference datasets...")
    df_addr = pd.read_csv(PATHS.addresses_csv)
    df_loc = pd.read_csv(PATHS.localities_csv)
    df_poi = pd.read_csv(PATHS.landmarks_poi_csv)
    df_towns = pd.read_csv(PATHS.towns_csv)
    print(f"  * Addresses loaded: {len(df_addr):,} rows")
    print(f"  * Localities loaded: {len(df_loc):,} rows (36 administrative neighborhoods)")
    print(f"  * Landmarks (POIs) loaded: {len(df_poi):,} rows")
    print(f"  * Towns loaded: {len(df_towns):,} rows ({', '.join(df_towns['town_id'].tolist())})")

    # 2. Verify Data Contract & Structural Invariants
    print("\n[Step 2/5] Verifying Phase 2 structural invariants...")
    # Zero-Duplicate Locality Invariant verification
    poi_assignments = _compute_poi_locality_assignments(df_poi, df_loc)
    poi_loc_types = []
    for _, poi in df_poi.iterrows():
        loc_id, dist_m = poi_assignments[str(poi["poi_id"])]
        poi_loc_types.append((loc_id, poi["landmark_type"]))
    
    dup_series = pd.Series(poi_loc_types).value_counts()
    max_dups = dup_series.max() if not dup_series.empty else 0
    print(f"  * Max duplicate POI types per locality disc: {max_dups} (Required: <= 1)")
    assert max_dups <= 1, f"Invariant violation: duplicate POI type within locality found ({max_dups})"
    print("  [PASS] Zero-Duplicate Locality Invariant verified (0 duplicate POIs per type per locality).")

    # 3. Execute Batch Address Parsing & Entity Resolution
    print("\n[Step 3/5] Parsing addresses with multilingual normalizer and POI resolver...")
    t_parse_start = time.time()
    parsed_entities = parse_addresses_batch(
        df_addresses=df_addr,
        df_localities=df_loc,
        df_pois=df_poi,
        config=DEFAULT_CONFIG,
    )
    t_parse_end = time.time()
    print(f"  * Parsed {len(parsed_entities):,} addresses in {t_parse_end - t_parse_start:.2f}s "
          f"({len(parsed_entities)/(t_parse_end - t_parse_start):.1f} addr/s)")

    # 4. Compute Quality & Resolution Metrics
    print("\n[Step 4/5] Computing entity resolution diagnostics & quality scoreboard...")
    df_parsed = pd.DataFrame([e.__dict__ for e in parsed_entities])
    
    # In-scope filtering (towns T1, T2, T3)
    in_scope_mask = df_parsed["town_id"].isin(["T1", "T2", "T3"])
    df_scope = df_parsed[in_scope_mask]
    n_total = len(df_parsed)
    n_scope = len(df_scope)
    n_out = n_total - n_scope

    # Locality resolution counts
    loc_resolved = int((df_scope["locality_resolution_status"] == "resolved").sum())
    loc_ambiguous = int((df_scope["locality_resolution_status"] == "ambiguous").sum())
    loc_unresolved = int((df_scope["locality_resolution_status"] == "unresolved").sum())
    loc_rate = loc_resolved / n_scope if n_scope > 0 else 0.0

    # Street extraction counts
    street_extracted = int(df_scope["street_normalized"].notna().sum())
    street_rate = street_extracted / n_scope if n_scope > 0 else 0.0

    # Landmark mention and POI resolution counts
    lm_detected = int(df_scope["landmark_raw"].notna().sum())
    poi_resolved = int((df_scope["poi_resolution_status"] == "resolved").sum())
    poi_ambiguous = int((df_scope["poi_resolution_status"] == "ambiguous").sum())
    poi_unresolved = int((df_scope["poi_resolution_status"] == "unresolved").sum())
    
    # Relation detection counts
    rel_detected = int(df_scope["spatial_relation"].notna().sum())
    rel_breakdown = df_scope["spatial_relation"].value_counts().to_dict()

    # Script distribution
    script_counts = df_parsed["script_detected"].value_counts().to_dict()
    multi_count = int(df_parsed["is_multilingual"].sum())

    print("-" * 70)
    print("PHASE 2 ENTITY RESOLUTION SCOREBOARD:")
    print("-" * 70)
    print(f"Total Addresses Ingested:        {n_total:,} (In-Scope: {n_scope:,}, OUT: {n_out:,})")
    print(f"Script Classification:           Latin: {script_counts.get('latin', 0):,}, "
          f"Mixed Native: {script_counts.get('mixed', 0):,}, "
          f"Total Multilingual: {multi_count:,}")
    print(f"Locality Resolution (In-Scope):  Resolved: {loc_resolved:,} ({loc_rate:.1%}) [Target: >= 65.0%]")
    print(f"                                 Ambiguous: {loc_ambiguous:,} ({loc_ambiguous/n_scope:.1%})")
    print(f"                                 Unresolved: {loc_unresolved:,} ({loc_unresolved/n_scope:.1%})")
    print(f"Street Token Extraction:         Extracted: {street_extracted:,} ({street_rate:.1%}) [Target: >= 55.0%]")
    print(f"Landmark Mentions Detected:      {lm_detected:,} ({lm_detected/n_scope:.1%})")
    print(f"POI Resolution Status:           Resolved: {poi_resolved:,} ({poi_resolved/n_scope:.1%})")
    print(f"                                 Ambiguous: {poi_ambiguous:,} ({poi_ambiguous/n_scope:.1%})")
    print(f"                                 Unresolved: {poi_unresolved:,} ({poi_unresolved/n_scope:.1%})")
    print(f"Spatial Relational Cues:         Detected: {rel_detected:,} ({rel_detected/n_scope:.1%})")
    for r_k, r_v in rel_breakdown.items():
        print(f"                                 * '{r_k}': {r_v:,}")
    print("-" * 70)

    # Validate Acceptance Criteria
    assert loc_rate >= 0.65, f"Locality recognition failed: {loc_rate:.1%} < 65.0%"
    assert street_rate >= 0.55, f"Street token extraction failed: {street_rate:.1%} < 55.0%"
    print("[PASS] Target criteria achieved: Locality Rate >= 65% and Street Rate >= 55%.")

    # 5. Export Auditable Artifacts
    print("\n[Step 5/5] Exporting auditable machine-readable artifacts...")
    PATHS.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Parquet tables
    df_parsed.to_parquet(PATHS.parsed_addresses_parquet, index=False)
    df_parsed.to_parquet(PATHS.address_entities_parquet, index=False)
    print(f"  * Exported: {PATHS.parsed_addresses_parquet.name} ({len(df_parsed)} rows)")
    print(f"  * Exported: {PATHS.address_entities_parquet.name} ({len(df_parsed)} rows)")


    # JSON Summaries (zero absolute paths, repository-relative only)
    parser_summary: Dict[str, Any] = {
        "phase": 2,
        "description": "Multilingual Script-Aware Address Parser & Locality-Bounded POI Resolver",
        "total_addresses": n_total,
        "in_scope_addresses": n_scope,
        "out_of_territory_addresses": n_out,
        "multilingual_addresses_count": multi_count,
        "script_distribution": script_counts,
        "locality_resolution": {
            "resolved_count": loc_resolved,
            "resolved_percentage": round(loc_rate * 100.0, 2),
            "target_percentage": 65.0,
            "ambiguous_count": loc_ambiguous,
            "unresolved_count": loc_unresolved,
            "status": "PASS" if loc_rate >= 0.65 else "FAIL",
        },
        "street_extraction": {
            "extracted_count": street_extracted,
            "extracted_percentage": round(street_rate * 100.0, 2),
            "target_percentage": 55.0,
            "status": "PASS" if street_rate >= 0.55 else "FAIL",
        },
        "landmark_poi_resolution": {
            "landmarks_detected_count": lm_detected,
            "landmarks_detected_percentage": round((lm_detected / n_scope) * 100.0, 2),
            "poi_resolved_count": poi_resolved,
            "poi_resolved_percentage": round((poi_resolved / n_scope) * 100.0, 2),
            "poi_ambiguous_count": poi_ambiguous,
            "poi_ambiguous_percentage": round((poi_ambiguous / n_scope) * 100.0, 2),
            "poi_unresolved_count": poi_unresolved,
            "poi_unresolved_percentage": round((poi_unresolved / n_scope) * 100.0, 2),
            "zero_duplicate_invariant": "VERIFIED (max duplicates per locality = 0)",
        },
        "relational_phrases": {
            "total_detected": rel_detected,
            "percentage": round((rel_detected / n_scope) * 100.0, 2),
            "breakdown": rel_breakdown,
        },
        "execution_time_seconds": round(time.time() - t0, 2),
    }

    poi_summary: Dict[str, Any] = {
        "phase": 2,
        "artifact": "poi_resolution_summary",
        "total_pois_catalog": len(df_poi),
        "total_localities_catalog": len(df_loc),
        "zero_duplicate_locality_invariant": True,
        "max_poi_per_type_within_locality": int(max_dups),
        "resolution_statistics": {
            "resolved": poi_resolved,
            "ambiguous": poi_ambiguous,
            "unresolved": poi_unresolved,
        },
        "spatial_offset_policy": "NO_DISPLACEMENT_APPLIED (C3 = physical POI catalog coordinate)",
    }

    with open(PATHS.phase2_parser_summary_json, "w", encoding="utf-8") as f:
        json.dump(parser_summary, f, indent=2)
    with open(PATHS.poi_resolution_summary_json, "w", encoding="utf-8") as f:
        json.dump(poi_summary, f, indent=2)

    print(f"  * Exported: {PATHS.phase2_parser_summary_json.name}")
    print(f"  * Exported: {PATHS.poi_resolution_summary_json.name}")

    print("\n" + "=" * 70)
    print("PHASE 2 EXECUTION COMPLETE -- ALL TESTS AND CONSTRAINTS SATISFIED")
    print("=" * 70)



if __name__ == "__main__":
    main()
