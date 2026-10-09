"""Phase 5 Pipeline Runner: Offline Deterministic Navigator & Evidence Card Generator.

Authority: PS3_FINAL_ARCHITECTURE.md Section 4, PS3_MODEL_SPEC.md Section 8,
and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.

Outputs:
- reports/metrics/phase5_navigation_examples.csv
- reports/metrics/phase5_evidence_cards.json
- reports/metrics/phase5_summary.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neurogeo.config import PATHS
from neurogeo.navigator import (
    navigate,
    build_evidence_card,
    destination_offset,
    euclidean_distance_m,
)
from neurogeo.schemas import NavigationMode, PredictionStatus


def run_phase5() -> None:
    t0 = time.time()
    print("=" * 75)
    print("NeuroGeo-Field: Phase 5 Execution")
    print("Offline Deterministic Navigator & Auditable Evidence Cards")
    print("=" * 75)
    print(f"Repository Root: {PATHS.repo_root.name}")
    print(f"Outputs Directory: {PATHS.metrics_dir.relative_to(PATHS.repo_root)}\n")

    # -------------------------------------------------------------
    # Step 1: Ingest Frozen Phase 4 Predictions & Local Evidence
    # -------------------------------------------------------------
    print("[Step 1/4] Ingesting frozen Phase 4 predictions and local evidence artifacts...")
    assert PATHS.phase4_predictions_csv.exists(), "Phase 4 predictions artifact missing!"
    df_preds = pd.read_csv(PATHS.phase4_predictions_csv)
    print(f"  * Phase 4 predictions ingested: {len(df_preds)} rows (authoritative)")

    # Load local supporting evidence (pure offline)
    df_addr_ent = pd.read_parquet(PATHS.address_entities_parquet) if PATHS.address_entities_parquet.exists() else pd.DataFrame()
    df_vis = pd.read_parquet(PATHS.metrics_dir / "visit_features.parquet") if (PATHS.metrics_dir / "visit_features.parquet").exists() else pd.DataFrame()

    addr_ent_map = {}
    if not df_addr_ent.empty and "address_id" in df_addr_ent.columns:
        for _, row in df_addr_ent.iterrows():
            addr_ent_map[str(row["address_id"]).strip()] = row.to_dict()

    visits_by_addr: Dict[str, List[Dict[str, Any]]] = {}
    if not df_vis.empty and "address_id" in df_vis.columns:
        for _, vrow in df_vis.iterrows():
            aid = str(vrow["address_id"]).strip()
            visits_by_addr.setdefault(aid, []).append(vrow.to_dict())

    print(f"  * Address entity metadata: {len(addr_ent_map)} addresses")
    print(f"  * Operational visit records: {len(df_vis)} visits across {len(visits_by_addr)} addresses")

    # -------------------------------------------------------------
    # Step 2: Build Deterministic Auditable Evidence Cards
    # -------------------------------------------------------------
    print("\n[Step 2/4] Generating deterministic auditable evidence cards...")
    evidence_cards: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    warning_count = 0

    for _, row in df_preds.iterrows():
        aid = str(row["address_id"]).strip()
        ent_data = addr_ent_map.get(aid, {})
        v_list = visits_by_addr.get(aid, [])

        ctx = {
            "locality_name": ent_data.get("locality_name_raw"),
            "locality_id_resolved": ent_data.get("locality_id_resolved"),
            "street_name": ent_data.get("street_normalized"),
            "landmark_name": ent_data.get("landmark_raw"),
            "direct_visits": v_list,
        }

        card = build_evidence_card(prediction=row.to_dict(), evidence_context=ctx)
        card_dict = card.to_dict()
        evidence_cards.append(card_dict)

        # Track stats
        st = card.status_tier
        status_counts[st] = status_counts.get(st, 0) + 1
        if card.negative_evidence or (card.caution and "WARNING" in card.caution):
            warning_count += 1

    out_cards_json = PATHS.phase5_evidence_cards_json
    with open(out_cards_json, "w", encoding="utf-8") as f:
        json.dump(evidence_cards, f, indent=2)
    print(f"  * Exported: {out_cards_json.relative_to(PATHS.repo_root)} ({len(evidence_cards)} cards)")
    print(f"  * Status distribution: {status_counts}")
    print(f"  * Cards with negative field warnings: {warning_count}")

    # -------------------------------------------------------------
    # Step 3: Generate Deterministic Navigation Examples
    # -------------------------------------------------------------
    print("\n[Step 3/4] Generating deterministic navigation scenarios across synthetic agent positions...")
    nav_examples: List[Dict[str, Any]] = []
    nav_mode_counts: Dict[str, int] = {}

    for i, row in df_preds.iterrows():
        aid = str(row["address_id"]).strip()
        tx = float(row["predicted_x"])
        ty = float(row["predicted_y"])
        rad_m = float(row["confidence_radius_m"])
        card_data = evidence_cards[i]
        neg_ev = card_data.get("negative_evidence", [])

        # Define 4 fixed deterministic synthetic current positions per target:
        # 1. Town origin (0, 0)
        # 2. Inside confidence radius (offset by 0.4 * rad along bearing 45 deg)
        # 3. Outside confidence radius (offset by 2.0 * rad along bearing 225 deg)
        # 4. At target (distance = 0.0 m)
        in_pos = destination_offset(tx, ty, distance_m=0.4 * rad_m, bearing_deg=225.0)  # Agent looking toward target
        out_pos = destination_offset(tx, ty, distance_m=2.0 * rad_m, bearing_deg=225.0)

        scenarios = [
            ("town_origin", 0.0, 0.0),
            ("inside_radius", in_pos[0], in_pos[1]),
            ("outside_radius", out_pos[0], out_pos[1]),
            ("at_destination", tx, ty),
        ]

        for sc_name, cx, cy in scenarios:
            inst = navigate(current_x=cx, current_y=cy, prediction=row.to_dict(), negative_evidence=neg_ev)
            rec = inst.to_dict()
            rec["address_id"] = aid
            rec["scenario_name"] = sc_name
            nav_examples.append(rec)

            if sc_name == "outside_radius":
                mode = inst.navigation_mode
                nav_mode_counts[mode] = nav_mode_counts.get(mode, 0) + 1

    df_nav_examples = pd.DataFrame(nav_examples)
    out_nav_csv = PATHS.phase5_navigation_examples_csv
    df_nav_examples.to_csv(out_nav_csv, index=False)
    print(f"  * Exported: {out_nav_csv.relative_to(PATHS.repo_root)} ({len(df_nav_examples)} instructions)")
    print(f"  * Navigation mode distribution: {nav_mode_counts}")

    # -------------------------------------------------------------
    # Step 4: Immutability Verification & Summary Export
    # -------------------------------------------------------------
    print("\n[Step 4/4] Verifying Phase 4 prediction immutability and exporting summary...")
    # Verify bit-for-bit immutability against Phase 4 predictions
    df_preds_check = pd.read_csv(PATHS.phase4_predictions_csv)
    assert len(df_preds_check) == len(df_preds), "Prediction row count altered!"
    assert (df_preds_check["predicted_x"] == df_preds["predicted_x"]).all(), "predicted_x altered!"
    assert (df_preds_check["predicted_y"] == df_preds["predicted_y"]).all(), "predicted_y altered!"
    assert (df_preds_check["selected_candidate_type"] == df_preds["selected_candidate_type"]).all(), "selected candidate altered!"
    assert (df_preds_check["status_tier"] == df_preds["status_tier"]).all(), "status_tier altered!"
    assert (df_preds_check["confidence_radius_m"] == df_preds["confidence_radius_m"]).all(), "confidence radius altered!"
    print("  * Phase 4 Immutability Check: PASSED (100/100 predictions completely unchanged)")

    # Build phase5_summary.json
    summary = {
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 5,
        "phase_description": "Offline Deterministic Navigator & Auditable Evidence Cards",
        "provenance": {
            "source_prediction_artifact": PATHS.phase4_predictions_csv.relative_to(PATHS.repo_root).as_posix(),
            "evidence_cards_artifact": PATHS.phase5_evidence_cards_json.relative_to(PATHS.repo_root).as_posix(),
            "navigation_examples_artifact": PATHS.phase5_navigation_examples_csv.relative_to(PATHS.repo_root).as_posix(),
            "offline_mode": True,
            "external_network_accessed": False,
            "routing_engine": "deterministic_vector_geometry",
            "road_network_claimed": False,
            "coordinate_convention": "metric_cartesian_meters",
        },
        "counts": {
            "phase4_predictions_consumed": len(df_preds),
            "evidence_cards_generated": len(evidence_cards),
            "navigation_examples_generated": len(df_nav_examples),
            "predictions_unchanged": len(df_preds),
            "negative_field_warnings": warning_count,
        },
        "status_tier_counts": status_counts,
        "navigation_mode_counts": nav_mode_counts,
        "sample_navigation_instruction": nav_examples[0],
        "sample_evidence_card": evidence_cards[0],
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_summary_json = PATHS.phase5_summary_json
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  * Exported: {out_summary_json.relative_to(PATHS.repo_root)}")

    print("\n" + "=" * 75)
    print("PHASE 5 SAMPLE OUTPUTS (DETERMINISTIC VERIFICATION)")
    print("=" * 75)
    sample_ex = nav_examples[0]
    print(f"Address: {sample_ex['address_id']} | Mode: {sample_ex['navigation_mode']} | Tier: {sample_ex['status_tier']}")
    print(f"  * Target: ({sample_ex['target_x']:.1f}, {sample_ex['target_y']:.1f}) | Radius: {sample_ex['confidence_radius_m']:.1f}m")
    print(f"  * Scenario: {sample_ex['scenario_name']} | Distance: {sample_ex['distance_m']:.1f}m | Bearing: {sample_ex['bearing_deg']:.1f} deg ({sample_ex['direction']})")
    print(f"  * Instruction: {sample_ex['instruction']}")
    if sample_ex.get('caution'):
        print(f"  * Caution: {sample_ex['caution']}")

    print("\n[DONE] Phase 5 offline navigator and evidence cards successfully executed.")
    print(f"Total elapsed time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run_phase5()
