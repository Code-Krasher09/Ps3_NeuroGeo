"""Evidence-Tier, Candidate-Wise, and Close-Landmark Geometry Calibration.

Authority: NEUROGEO_FINAL_ARCHITECTURE.md, NEUROGEO_MODEL_SPEC.md, NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md,
and Prompt 30 (Candidate-wise P90 & Close-Landmark Spatial-Scale Geometry Integration).
"""

import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, ModelConfig, PATHS
from neurogeo.leakage import assert_fit_scope_valid
from neurogeo.poi_resolver import _compute_poi_locality_assignments
from neurogeo.schemas import CalibrationResult, PredictionStatus


# Evidence family mapping for Level 2 sample fallback
EVIDENCE_FAMILIES = {
    PredictionStatus.DOORSTEP_VERIFIED.value: "visited",
    PredictionStatus.STREET_RESOLVED.value: "cold_start",
    PredictionStatus.LOCALITY_BOUNDED.value: "cold_start",
    PredictionStatus.BASELINE_FALLBACK.value: "cold_start",
}


def assign_status_tier(
    selected_candidate_type: str,
    has_credible_doorstep: bool = False,
    neighbor_count: int = 0,
    is_poi_resolved: bool = False,
    baseline_precision: str = "unknown",
) -> PredictionStatus:
    """Deterministically assign prediction to one of 4 discrete prediction status tiers.
    
    Authority: NEUROGEO_FINAL_ARCHITECTURE.md Section 4 and NEUROGEO_MODEL_SPEC.md Section 7:
    - DOORSTEP_VERIFIED: Confirmed by direct high-integrity doorstep visits (C2 selected or credible doorstep).
    - STREET_RESOLVED: Cold-start address, Candidate C4 selected with >= 2 supporting neighbor accounts, or C3 anchored to resolved landmark.
    - LOCALITY_BOUNDED: Unvisited address, Candidate C5 (Locality Centroid) selected, or C4 with insufficient neighbors.
    - BASELINE_FALLBACK: Commercial baseline geocoder selected (C1).
    """
    c_type = str(selected_candidate_type).upper()

    if c_type == "C2" or (c_type in {"C1", "C3", "C4"} and has_credible_doorstep):
        return PredictionStatus.DOORSTEP_VERIFIED

    if c_type == "C4":
        if neighbor_count >= 2:
            return PredictionStatus.STREET_RESOLVED
        return PredictionStatus.LOCALITY_BOUNDED

    if c_type == "C3":
        if is_poi_resolved:
            return PredictionStatus.STREET_RESOLVED
        return PredictionStatus.LOCALITY_BOUNDED

    if c_type == "C5":
        return PredictionStatus.LOCALITY_BOUNDED

    # C1 (Baseline)
    prec = str(baseline_precision).lower()
    if prec in {"rooftop", "street"}:
        return PredictionStatus.STREET_RESOLVED
    elif prec == "locality":
        return PredictionStatus.LOCALITY_BOUNDED
    else:
        return PredictionStatus.BASELINE_FALLBACK


def get_calibrated_radius_with_fallback(
    tier: str,
    residuals_by_tier: Dict[str, List[float]],
    min_samples: int = 15,
    target_quantile: float = 0.90,
    config: Optional[ModelConfig] = None,
) -> Tuple[float, int, str]:
    """Compute calibrated empirical radius using frozen 4-level fallback hierarchy.
    
    Authority: NEUROGEO_MODEL_SPEC.md Section 7 lines 158-176.
    
    Returns:
        (radius_m, sample_count, fallback_level)
        where fallback_level is one of:
        - 'tier_direct' (Level 1: >= min_samples in specific tier)
        - 'evidence_family' (Level 2: >= min_samples in evidence family)
        - 'global_p90' (Level 3: pooled across all SCV residuals)
        - 'operational_prior' (Level 4: initial architectural design target if no data)
    """
    cfg = config or DEFAULT_CONFIG
    t_key = str(tier)

    # Level 1: Tier-specific empirical P90
    tier_res = residuals_by_tier.get(t_key, [])
    if len(tier_res) >= min_samples:
        r = float(np.quantile(tier_res, target_quantile))
        return round(r, 2), len(tier_res), "tier_direct"

    # Level 2: Evidence-Family pooled P90 (Visited Family: T1 vs Cold-Start Family: T2, T3, T4)
    family = EVIDENCE_FAMILIES.get(t_key, "cold_start")
    family_res: List[float] = []
    for t_name, res_list in residuals_by_tier.items():
        if EVIDENCE_FAMILIES.get(t_name, "cold_start") == family:
            family_res.extend(res_list)

    if len(family_res) >= min_samples:
        r = float(np.quantile(family_res, target_quantile))
        return round(r, 2), len(family_res), "evidence_family"

    # Level 3: Global pooled empirical P90
    global_res = [r for res in residuals_by_tier.values() for r in res]
    if len(global_res) >= min_samples:
        r = float(np.quantile(global_res, target_quantile))
        return round(r, 2), len(global_res), "global_p90"

    # Level 4: Operational engineering design target (fallback prior)
    fallback_defaults = {
        PredictionStatus.DOORSTEP_VERIFIED.value: cfg.status_threshold_doorstep_r_m,
        PredictionStatus.STREET_RESOLVED.value: cfg.status_threshold_street_r_m,
        PredictionStatus.LOCALITY_BOUNDED.value: cfg.status_threshold_locality_r_m,
        PredictionStatus.BASELINE_FALLBACK.value: cfg.status_threshold_baseline_r_m,
    }
    r_default = fallback_defaults.get(t_key, 250.0)
    return round(float(r_default), 2), len(global_res), "operational_prior"


_UNMENTIONED_POI_CACHE: Dict[Tuple[int, int, str, str, Optional[str], float, float], Tuple[Optional[float], Optional[str], str]] = {}


def find_nearest_unmentioned_poi(
    primary_poi_id: str,
    town_id: str,
    locality_id: Optional[str],
    df_pois: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
    collocation_distance_threshold_m: float = 5.0,
    town_fallback_radius_m: float = 1500.0,
) -> Tuple[Optional[float], Optional[str], str]:
    """Find distance to nearest eligible unmentioned POI for a resolved landmark anchor.
    
    Authority: Prompt 30 Phase C.2.
    
    Criteria:
    - Excludes the anchor POI itself (poi_id != primary_poi_id).
    - Excludes collocated POIs representing the same physical location
      (distance < collocation_distance_threshold_m, provisional default 5.0 m).
    - Scoped to same town (town_id).
    - Preferred pool: same town and same locality.
    - Fallback pool: same town within town_fallback_radius_m (provisional default 1500.0 m)
      only if local pool is empty.
    - If neither pool contains eligible POIs, marks geometry path ineligible.
    
    Returns:
        (nearest_distance_m, nearest_poi_id, pool_status)
    """
    cache_key = (
        id(df_pois),
        len(df_pois),
        str(primary_poi_id),
        str(town_id),
        str(locality_id) if locality_id is not None else None,
        float(collocation_distance_threshold_m),
        float(town_fallback_radius_m),
    )
    if cache_key in _UNMENTIONED_POI_CACHE:
        return _UNMENTIONED_POI_CACHE[cache_key]

    if primary_poi_id not in df_pois["poi_id"].values:
        _UNMENTIONED_POI_CACHE[cache_key] = (None, None, "anchor_not_found")
        return None, None, "anchor_not_found"

    anchor_row = df_pois[df_pois["poi_id"] == primary_poi_id].iloc[0]
    ax, ay = float(anchor_row["x"]), float(anchor_row["y"])

    if df_localities is None and PATHS.localities_csv.exists():
        try:
            df_localities = pd.read_csv(PATHS.localities_csv)
        except Exception:
            df_localities = None

    poi_loc_map: Dict[str, Tuple[str, float]] = {}
    if df_localities is not None:
        poi_loc_map = _compute_poi_locality_assignments(df_pois, df_localities)

    anc_loc, _ = poi_loc_map.get(primary_poi_id, (locality_id, 0.0))
    if not anc_loc and locality_id:
        anc_loc = locality_id

    same_loc_pois: List[Tuple[float, str]] = []
    same_town_pois: List[Tuple[float, str]] = []

    for _, poi in df_pois.iterrows():
        p_id = str(poi["poi_id"])
        if p_id == primary_poi_id:
            continue
        if str(poi["town_id"]) != str(town_id):
            continue

        px, py = float(poi["x"]), float(poi["y"])
        d = math.hypot(px - ax, py - ay)

        # Collocated tolerance: exclude POIs representing the same physical location
        if d < collocation_distance_threshold_m:
            continue

        p_loc, _ = poi_loc_map.get(p_id, (None, 9999.0))
        if anc_loc and p_loc == anc_loc:
            same_loc_pois.append((d, p_id))

        if d <= town_fallback_radius_m:
            same_town_pois.append((d, p_id))

    if same_loc_pois:
        same_loc_pois.sort(key=lambda item: (item[0], item[1]))
        res = (round(same_loc_pois[0][0], 2), same_loc_pois[0][1], "locality_pool")
        _UNMENTIONED_POI_CACHE[cache_key] = res
        return res

    if same_town_pois:
        same_town_pois.sort(key=lambda item: (item[0], item[1]))
        res = (round(same_town_pois[0][0], 2), same_town_pois[0][1], "town_fallback_pool")
        _UNMENTIONED_POI_CACHE[cache_key] = res
        return res

    res = (None, None, "no_eligible_unmentioned_pois")
    _UNMENTIONED_POI_CACHE[cache_key] = res
    return res


def compute_landmark_spatial_scale(
    predicted_coord: Tuple[float, float],
    anchor_coord: Tuple[float, float],
    nearest_unmentioned_distance_m: float,
    max_consistency_distance_m: float = 1000.0,
    min_scale_floor_m: float = 15.0,
) -> Tuple[Optional[float], str]:
    """Compute characteristic spatial scale G_i for a prediction and landmark anchor.
    
    Authority: Prompt 30 Phase C.3.
    Formula:
        G_i = max(||P_i - L_0||_2, d_min, 15.0 m)
        
    Consistency check:
        ||P_i - L_0||_2 <= max_consistency_distance_m (1000.0 m).
    """
    px, py = predicted_coord
    ax, ay = anchor_coord
    d_pred_anchor = math.hypot(px - ax, py - ay)

    if d_pred_anchor > max_consistency_distance_m:
        return None, "prediction_anchor_spatial_inconsistency"

    scale_m = max(d_pred_anchor, float(nearest_unmentioned_distance_m), float(min_scale_floor_m))
    return round(scale_m, 2), "scale_computed"


_SCV_EPISODES_CACHE: Dict[str, Dict[str, Any]] = {}


def extract_scv_candidate_episodes(
    df_clean: pd.DataFrame,
    df_parsed: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_localities: pd.DataFrame,
    df_pois: pd.DataFrame,
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Simulate production cold-start candidate generation & ranking for clean SCV episodes.
    
    Zero-leakage: target doorstep visits are masked out by design.
    Caches computed predictions across folds and test invocations.
    """
    global _SCV_EPISODES_CACHE
    from dataclasses import replace
    from neurogeo.candidate_ranker import (
        generate_candidates,
        extract_candidate_features,
        rank_candidates,
    )

    needed_aids = [str(r).strip() for r in df_clean["address_id"]]
    missing_aids = [aid for aid in needed_aids if aid not in _SCV_EPISODES_CACHE]

    if missing_aids:
        parsed_map = df_parsed.set_index("address_id").to_dict("index")
        base_map = df_baseline.set_index("address_id").to_dict("index")
        loc_map = df_localities.set_index("locality_id").to_dict("index")
        poi_map = df_pois.set_index("poi_id").to_dict("index")

        df_missing = df_clean[df_clean["address_id"].astype(str).str.strip().isin(set(missing_aids))]
        for _, row in df_missing.iterrows():
            aid = str(row["address_id"]).strip()
            p_ent = parsed_map.get(aid, {})
            b_row = base_map.get(aid)

            c4_avail = bool(row.get("c4_available", False))
            c4_res = {
                "c4_x": row.get("c4_x") if c4_avail else None,
                "c4_y": row.get("c4_y") if c4_avail else None,
                "neighbor_count": int(row.get("neighbor_count", 0)) if c4_avail else 0,
                "total_neighbor_weight": float(row.get("total_neighbor_weight", 0.0)) if c4_avail else 0.0,
                "neighbor_address_ids": str(row.get("neighbor_address_ids", "")).split(",") if pd.notna(row.get("neighbor_address_ids")) else [],
                "town_id": row.get("town_id"),
                "locality_id": row.get("locality_id_resolved"),
                "street_normalized": row.get("street_normalized"),
            }

            lid = row.get("locality_id_resolved")
            l_row = loc_map.get(str(lid)) if pd.notna(lid) else None
            pid = p_ent.get("landmark_poi_id")
            p_row = poi_map.get(pid) if pd.notna(pid) else None

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
            if not ranked:
                continue
            sel = ranked[0]

            gt_x = float(row["ground_truth_medoid_x"])
            gt_y = float(row["ground_truth_medoid_y"])
            err_m = float(math.hypot(sel.x - gt_x, sel.y - gt_y))

            _SCV_EPISODES_CACHE[aid] = {
                "address_id": aid,
                "selected_candidate_type": sel.candidate_type,
                "pred_x": round(float(sel.x), 2),
                "pred_y": round(float(sel.y), 2),
                "gt_x": round(gt_x, 2),
                "gt_y": round(gt_y, 2),
                "error_m": round(err_m, 2),
                "town_id": str(row.get("town_id", "")),
                "locality_id": str(lid) if pd.notna(lid) else None,
                "poi_resolution_status": p_ent.get("poi_resolution_status"),
                "landmark_poi_id": pid,
            }

    return [_SCV_EPISODES_CACHE[aid] for aid in needed_aids if aid in _SCV_EPISODES_CACHE]


def fit_empirical_calibration_quantiles(
    df_scv: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
    test_address_ids: Optional[Set[str]] = None,
    validation_address_ids: Optional[Set[str]] = None,
    train_address_ids: Optional[Set[str]] = None,
    min_samples: int = 15,
    target_quantile: float = 0.90,
    fit_scope: str = "calibration_oof",
    df_parsed: Optional[pd.DataFrame] = None,
    df_baseline: Optional[pd.DataFrame] = None,
    df_pois: Optional[pd.DataFrame] = None,
    reranker_weights: Optional[Dict[str, float]] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Fit complete production calibration tables from leakage-safe held-out SCV residuals.
    
    Zero-leakage guarantees:
    - Target test, validation, and train benchmark addresses are strictly excluded from fitting calibration quantiles.
    - Explicit assertion: calibration_address_ids ∩ benchmark_address_ids == EMPTY.
    - Explicit assertion: calibration_address_ids ∩ reranker_train_address_ids == EMPTY.
    - Uses SCV residuals (reserved_for_calibration = True, target_history_masked = True).
    - Preserves provenance fit_scope.
    """
    assert_fit_scope_valid("calibration_quantiles", fit_scope)
    cfg = config or DEFAULT_CONFIG

    # 1. Filter out all benchmark addresses (train, validation, test)
    excluded_ids: Set[str] = set()
    if train_address_ids:
        excluded_ids.update(train_address_ids)
    if validation_address_ids:
        excluded_ids.update(validation_address_ids)
    if test_address_ids:
        excluded_ids.update(test_address_ids)

    df_clean = df_scv[~df_scv["address_id"].isin(excluded_ids)].copy()

    # Programmatic assertion of disjointness
    cal_addrs = set(df_clean["address_id"])
    if train_address_ids:
        assert len(cal_addrs & train_address_ids) == 0, (
            f"Leakage detected: {len(cal_addrs & train_address_ids)} train addresses in calibration data"
        )
    if validation_address_ids:
        assert len(cal_addrs & validation_address_ids) == 0, (
            f"Leakage detected: {len(cal_addrs & validation_address_ids)} validation addresses in calibration data"
        )
    if test_address_ids:
        assert len(cal_addrs & test_address_ids) == 0, (
            f"Leakage detected: {len(cal_addrs & test_address_ids)} test addresses in calibration data"
        )

    # Load localities if provided or from PATHS
    if df_localities is None and PATHS.localities_csv.exists():
        try:
            df_localities = pd.read_csv(PATHS.localities_csv)
        except Exception:
            df_localities = None

    loc_map: Dict[str, Tuple[float, float]] = {}
    if df_localities is not None and "locality_id" in df_localities.columns:
        for _, lrow in df_localities.iterrows():
            lid = str(lrow["locality_id"]).strip()
            cx, cy = float(lrow.get("centroid_x", 0.0)), float(lrow.get("centroid_y", 0.0))
            loc_map[lid] = (cx, cy)

    # 2. Extract residuals and unique episodes by evidence tier from leakage-safe held-out SCV residuals
    residuals_by_tier: Dict[str, List[float]] = {
        PredictionStatus.DOORSTEP_VERIFIED.value: [],
        PredictionStatus.STREET_RESOLVED.value: [],
        PredictionStatus.LOCALITY_BOUNDED.value: [],
        PredictionStatus.BASELINE_FALLBACK.value: [],
    }
    episodes_by_tier: Dict[str, Set[str]] = {
        PredictionStatus.DOORSTEP_VERIFIED.value: set(),
        PredictionStatus.STREET_RESOLVED.value: set(),
        PredictionStatus.LOCALITY_BOUNDED.value: set(),
        PredictionStatus.BASELINE_FALLBACK.value: set(),
    }

    for _, row in df_clean.iterrows():
        aid = str(row["address_id"]).strip()
        gt_x = row.get("ground_truth_medoid_x")
        gt_y = row.get("ground_truth_medoid_y")
        if pd.isna(gt_x) or pd.isna(gt_y):
            continue
        gt_x_f, gt_y_f = float(gt_x), float(gt_y)

        # C4 -> STREET_RESOLVED (when c4_available and neighbor_count >= 2)
        c4_avail = bool(row.get("c4_available", False))
        n_cnt = int(row.get("neighbor_count", 0))
        err_c4 = row.get("error_vs_medoid_m")
        if c4_avail and n_cnt >= 2 and pd.notna(err_c4):
            residuals_by_tier[PredictionStatus.STREET_RESOLVED.value].append(float(err_c4))
            episodes_by_tier[PredictionStatus.STREET_RESOLVED.value].add(aid)

        # C5 -> LOCALITY_BOUNDED
        loc_id = row.get("locality_id_resolved")
        if pd.notna(loc_id) and str(loc_id).strip() in loc_map:
            cx, cy = loc_map[str(loc_id).strip()]
            err_c5 = math.sqrt((cx - gt_x_f) ** 2 + (cy - gt_y_f) ** 2)
            residuals_by_tier[PredictionStatus.LOCALITY_BOUNDED.value].append(float(err_c5))
            episodes_by_tier[PredictionStatus.LOCALITY_BOUNDED.value].add(aid)

        # C1 -> BASELINE_FALLBACK
        b_err = row.get("baseline_error_vs_medoid_m")
        if pd.notna(b_err):
            residuals_by_tier[PredictionStatus.BASELINE_FALLBACK.value].append(float(b_err))
            episodes_by_tier[PredictionStatus.BASELINE_FALLBACK.value].add(aid)

    # 3. Compute calibrated quantiles using fallback hierarchy for evidence tiers
    tier_records: Dict[str, Dict[str, Any]] = {}
    fallback_counts: Dict[str, int] = {
        "tier_direct": 0,
        "evidence_family": 0,
        "global_p90": 0,
        "operational_prior": 0,
    }

    for tier_status in PredictionStatus:
        t_name = tier_status.value
        radius_m, sample_cnt, fb_level = get_calibrated_radius_with_fallback(
            tier=t_name,
            residuals_by_tier=residuals_by_tier,
            min_samples=min_samples,
            target_quantile=target_quantile,
            config=cfg,
        )
        fallback_counts[fb_level] = fallback_counts.get(fb_level, 0) + 1

        if fb_level == "tier_direct":
            n_episodes = len(episodes_by_tier[t_name])
            n_residuals = len(residuals_by_tier[t_name])
            family = EVIDENCE_FAMILIES.get(t_name, "cold_start")
        elif fb_level == "evidence_family":
            fam = EVIDENCE_FAMILIES.get(t_name, "cold_start")
            fam_episodes = set().union(*(episodes_by_tier[k] for k, f in EVIDENCE_FAMILIES.items() if f == fam))
            n_episodes = len(fam_episodes)
            n_residuals = sample_cnt
            family = fam
        elif fb_level == "global_p90":
            all_episodes = set().union(*episodes_by_tier.values())
            n_episodes = len(all_episodes)
            n_residuals = sample_cnt
            family = "all"
        else:  # operational_prior
            all_episodes = set().union(*episodes_by_tier.values())
            n_episodes = len(all_episodes)
            n_residuals = sample_cnt
            family = None

        # Physical floor & cap
        radius_m = min(max(radius_m, cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)

        tier_records[t_name] = {
            "tier": t_name,
            "unique_episode_count": n_episodes,
            "residual_count": n_residuals,
            "sample_count": sample_cnt,
            "quantile": target_quantile,
            "radius_m": radius_m,
            "source_scope": "scv_leave_one_out",
            "fallback_level": fb_level,
            "evidence_family": family,
        }

    # 4. Fit Candidate-Wise, Close-Landmark Geometry, and Locality Calibration
    candidates_dict: Dict[str, Any] = {}
    landmark_geom_dict: Dict[str, Any] = {}
    localities_dict: Dict[str, Any] = {}

    # Attempt to load supporting datasets if not explicitly provided
    if df_parsed is None and PATHS.parsed_addresses_parquet.exists():
        try:
            df_parsed = pd.read_parquet(PATHS.parsed_addresses_parquet)
        except Exception:
            df_parsed = None

    if df_baseline is None and PATHS.baseline_geocodes_csv.exists():
        try:
            df_baseline = pd.read_csv(PATHS.baseline_geocodes_csv)
        except Exception:
            df_baseline = None

    if df_pois is None and PATHS.landmarks_poi_csv.exists():
        try:
            df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
        except Exception:
            df_pois = None

    if reranker_weights is None and PATHS.reranker_weights_json.exists():
        try:
            reranker_weights = json.loads(PATHS.reranker_weights_json.read_text())["weights"]
        except Exception:
            reranker_weights = None

    # Check if we have complete artifacts to reconstruct candidate predictions
    can_reconstruct = (
        df_parsed is not None
        and df_baseline is not None
        and df_localities is not None
        and df_pois is not None
        and reranker_weights is not None
        and len(df_clean) > 0
        and df_clean["address_id"].isin(df_parsed["address_id"]).any()
    )

    if can_reconstruct:
        scv_episodes = extract_scv_candidate_episodes(
            df_clean=df_clean,
            df_parsed=df_parsed,
            df_baseline=df_baseline,
            df_localities=df_localities,
            df_pois=df_pois,
            weights=reranker_weights,
        )

        # 4.1 Candidate-wise P90 calibration
        default_tier_radii = {
            "C1": tier_records["BASELINE_FALLBACK"]["radius_m"],
            "C2": cfg.status_threshold_doorstep_r_m,
            "C3": tier_records["STREET_RESOLVED"]["radius_m"],
            "C4": tier_records["STREET_RESOLVED"]["radius_m"],
            "C5": tier_records["LOCALITY_BOUNDED"]["radius_m"],
        }

        for c_type in ["C1", "C2", "C3", "C4", "C5"]:
            if c_type == "C2":
                # C2 target visits masked in SCV by design: operational doorstep radius 35.0 m
                candidates_dict["C2"] = {
                    "candidate_type": "C2",
                    "sample_count": 0,
                    "radius_m": cfg.status_threshold_doorstep_r_m,
                    "quantile": target_quantile,
                    "fallback_level": "direct_doorstep",
                    "explanation": "Target doorstep visits masked in SCV; direct doorstep operational evidence.",
                }
                continue

            c_errs = [ep["error_m"] for ep in scv_episodes if ep["selected_candidate_type"] == c_type]
            if len(c_errs) >= min_samples:
                r_cand = float(np.quantile(c_errs, target_quantile))
                r_cand = min(max(round(r_cand, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)
                candidates_dict[c_type] = {
                    "candidate_type": c_type,
                    "sample_count": len(c_errs),
                    "radius_m": r_cand,
                    "quantile": target_quantile,
                    "fallback_level": "candidate_p90",
                }
            else:
                candidates_dict[c_type] = {
                    "candidate_type": c_type,
                    "sample_count": len(c_errs),
                    "radius_m": default_tier_radii.get(c_type, 250.0),
                    "quantile": target_quantile,
                    "fallback_level": "tier_fallback",
                }

        # 4.2 Close-landmark normalized-residual calibration
        poi_map = df_pois.set_index("poi_id").to_dict("index")
        geom_q_records: List[Dict[str, Any]] = []

        for ep in scv_episodes:
            if ep.get("poi_resolution_status") != "resolved":
                continue
            pid = ep.get("landmark_poi_id")
            if not pid or pid not in poi_map:
                continue

            anchor = poi_map[pid]
            ax, ay = float(anchor["x"]), float(anchor["y"])
            town = str(anchor["town_id"])

            d_min, unmentioned_id, pool_status = find_nearest_unmentioned_poi(
                primary_poi_id=pid,
                town_id=town,
                locality_id=ep.get("locality_id"),
                df_pois=df_pois,
                df_localities=df_localities,
                collocation_distance_threshold_m=cfg.collocation_distance_threshold_m,
                town_fallback_radius_m=cfg.town_poi_fallback_radius_m,
            )
            if d_min is None:
                continue

            g_scale, scale_status = compute_landmark_spatial_scale(
                predicted_coord=(ep["pred_x"], ep["pred_y"]),
                anchor_coord=(ax, ay),
                nearest_unmentioned_distance_m=d_min,
                max_consistency_distance_m=cfg.landmark_spatial_consistency_radius_m,
                min_scale_floor_m=cfg.min_calibration_radius_floor_m,
            )
            if g_scale is None:
                continue

            q_i = ep["error_m"] / g_scale
            geom_q_records.append({
                "address_id": ep["address_id"],
                "candidate_type": ep["selected_candidate_type"],
                "q_i": q_i,
                "g_scale": g_scale,
                "error_m": ep["error_m"],
            })

        if geom_q_records:
            all_q = [r["q_i"] for r in geom_q_records]
            pooled_q_factor = float(np.quantile(all_q, target_quantile))

            cand_q_factors: Dict[str, float] = {}
            cand_q_counts: Dict[str, int] = {}
            for c_type in ["C1", "C3", "C4", "C5"]:
                sub_q = [r["q_i"] for r in geom_q_records if r["candidate_type"] == c_type]
                cand_q_counts[c_type] = len(sub_q)
                if len(sub_q) >= min_samples:
                    cand_q_factors[c_type] = round(float(np.quantile(sub_q, target_quantile)), 4)
                else:
                    cand_q_factors[c_type] = round(pooled_q_factor, 4)

            landmark_geom_dict = {
                "target_quantile": target_quantile,
                "sample_count": len(geom_q_records),
                "pooled_quantile_factor": round(pooled_q_factor, 4),
                "candidate_quantile_factors": cand_q_factors,
                "candidate_sample_counts": cand_q_counts,
                "collocation_distance_threshold_m": cfg.collocation_distance_threshold_m,
                "landmark_spatial_consistency_radius_m": cfg.landmark_spatial_consistency_radius_m,
                "town_poi_fallback_radius_m": cfg.town_poi_fallback_radius_m,
                "min_scale_floor_m": cfg.min_calibration_radius_floor_m,
            }

        # 4.3 Locality empirical calibration
        locality_errors: Dict[str, List[float]] = {}
        for ep in scv_episodes:
            lid = ep.get("locality_id")
            if lid:
                locality_errors.setdefault(str(lid), []).append(ep["error_m"])

        for lid, err_list in locality_errors.items():
            if len(err_list) >= min_samples:
                r_loc = float(np.quantile(err_list, target_quantile))
                r_loc = min(max(round(r_loc, 2), cfg.min_calibration_radius_floor_m), cfg.max_calibration_radius_cap_m)
                localities_dict[lid] = {
                    "locality_id": lid,
                    "sample_count": len(err_list),
                    "radius_m": r_loc,
                    "quantile": target_quantile,
                }

    return {
        "fit_scope": fit_scope,
        "evaluation_scope": "scv_leave_one_out",
        "source_data_scope": "reports/metrics/scv_results.csv (1,477 SCV visited-address episodes)",
        "target_quantile": target_quantile,
        "min_tier_samples": min_samples,
        "total_scv_calibration_records": len(df_clean),
        "total_scv_episodes": len(df_clean),
        "excluded_benchmark_address_count": len(excluded_ids),
        "excluded_test_account_count": len(excluded_ids),
        "tiers": tier_records,
        "candidates": candidates_dict,
        "landmark_geometry": landmark_geom_dict,
        "localities": localities_dict,
        "fallback_counts": fallback_counts,
    }


def calibrate_prediction_production(
    selected_candidate_type: str,
    predicted_coord: Tuple[float, float],
    status_tier: Union[PredictionStatus, str],
    calibration_dict: Dict[str, Any],
    parsed_entity: Optional[Dict[str, Any]] = None,
    df_pois: Optional[pd.DataFrame] = None,
    df_localities: Optional[pd.DataFrame] = None,
    has_credible_doorstep: bool = False,
    config: Optional[ModelConfig] = None,
) -> CalibrationResult:
    """Production calibrated confidence-radius finder implementing full 5-level hierarchy.
    
    Authority: Prompt 30 (Integrate Candidate-Wise P90 and Close-Landmark Geometry).
    
    Hierarchy:
    - Level 0 (Direct Doorstep Evidence): C2 selected or verified doorstep visits -> 35.0 m.
    - Level 1 (Close-Landmark Spatial-Scale Geometry): Resolved anchor POI + unmentioned nearby POI
      -> Characteristic spatial scale G_i = max(||P_i - L_0||, d_min, 15m), radius = Q_0.90(q) * G_i.
    - Level 2 (Candidate-Wise Empirical P90): Candidate-specific empirical error P90 (>=15 samples).
    - Level 3 (Locality-Level Empirical P90): Locality-specific empirical error P90 (>=15 samples).
    - Level 4 (Evidence-Tier Empirical Quantile Fallback): STREET_RESOLVED, LOCALITY_BOUNDED, BASELINE_FALLBACK.
    
    Physical bounds [15.0 m, 2500.0 m] are enforced strictly AFTER final radius calculation across all levels.
    """
    cfg = config or DEFAULT_CONFIG
    c_type = str(selected_candidate_type).upper()
    t_name = status_tier.value if isinstance(status_tier, PredictionStatus) else str(status_tier)

    r_floor = cfg.min_calibration_radius_floor_m
    r_cap = cfg.max_calibration_radius_cap_m

    # -------------------------------------------------------------------------
    # Level 0: Direct Doorstep Evidence (C2 or credible doorstep visit)
    # -------------------------------------------------------------------------
    if c_type == "C2" or has_credible_doorstep:
        doorstep_r = min(max(cfg.status_threshold_doorstep_r_m, r_floor), r_cap)
        sample_count = calibration_dict.get("tiers", {}).get(PredictionStatus.DOORSTEP_VERIFIED.value, {}).get("sample_count", 0)
        return CalibrationResult(
            radius_m=round(doorstep_r, 2),
            confidence_level=0.90,
            calibration_method="direct_doorstep",
            calibration_tier=PredictionStatus.DOORSTEP_VERIFIED.value,
            fallback_level="direct_doorstep",
            sample_count=sample_count,
            geometry_scale_m=None,
            primary_landmark_poi_id=None,
            nearest_unmentioned_poi_id=None,
            nearest_unmentioned_distance_m=None,
            fallback_reason=None,
        )

    fallback_reason: Optional[str] = None

    # -------------------------------------------------------------------------
    # Level 1: Close-Landmark Spatial-Scale Geometry
    # -------------------------------------------------------------------------
    if cfg.enable_landmark_geometry_calibration and parsed_entity is not None:
        poi_status = str(parsed_entity.get("poi_resolution_status", "")).lower()
        pid = parsed_entity.get("landmark_poi_id")

        if poi_status == "resolved" and pid and pd.notna(pid):
            pid_str = str(pid).strip()
            if df_pois is None and PATHS.landmarks_poi_csv.exists():
                try:
                    df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
                except Exception:
                    df_pois = None

            if df_pois is not None and pid_str in df_pois["poi_id"].values:
                anchor_row = df_pois[df_pois["poi_id"] == pid_str].iloc[0]
                ax, ay = float(anchor_row["x"]), float(anchor_row["y"])
                town_id = str(anchor_row["town_id"])
                loc_id = parsed_entity.get("locality_id_resolved")

                d_min, unmentioned_id, pool_status = find_nearest_unmentioned_poi(
                    primary_poi_id=pid_str,
                    town_id=town_id,
                    locality_id=str(loc_id) if pd.notna(loc_id) else None,
                    df_pois=df_pois,
                    df_localities=df_localities,
                    collocation_distance_threshold_m=cfg.collocation_distance_threshold_m,
                    town_fallback_radius_m=cfg.town_poi_fallback_radius_m,
                )

                if d_min is not None:
                    g_scale, scale_status = compute_landmark_spatial_scale(
                        predicted_coord=predicted_coord,
                        anchor_coord=(ax, ay),
                        nearest_unmentioned_distance_m=d_min,
                        max_consistency_distance_m=cfg.landmark_spatial_consistency_radius_m,
                        min_scale_floor_m=cfg.min_calibration_radius_floor_m,
                    )

                    if g_scale is not None:
                        geom_dict = calibration_dict.get("landmark_geometry", {})
                        cand_factors = geom_dict.get("candidate_quantile_factors", {})
                        cand_counts = geom_dict.get("candidate_sample_counts", {})
                        pooled_factor = geom_dict.get("pooled_quantile_factor")
                        pooled_count = geom_dict.get("sample_count", 0)

                        q_factor: Optional[float] = None
                        fb_level = "pooled_geom"
                        s_count = pooled_count

                        if c_type in cand_factors and cand_counts.get(c_type, 0) >= cfg.min_calibration_samples:
                            q_factor = float(cand_factors[c_type])
                            fb_level = "candidate_specific_geom"
                            s_count = cand_counts[c_type]
                        elif pooled_factor is not None and pooled_count >= cfg.min_calibration_samples:
                            q_factor = float(pooled_factor)
                            fb_level = "pooled_geom"
                            s_count = pooled_count

                        if q_factor is not None:
                            calc_r = round(q_factor * g_scale, 2)
                            final_r = min(max(calc_r, r_floor), r_cap)
                            return CalibrationResult(
                                radius_m=final_r,
                                confidence_level=0.90,
                                calibration_method="landmark_geometry",
                                calibration_tier=t_name,
                                fallback_level=fb_level,
                                sample_count=s_count,
                                geometry_scale_m=g_scale,
                                primary_landmark_poi_id=pid_str,
                                nearest_unmentioned_poi_id=unmentioned_id,
                                nearest_unmentioned_distance_m=d_min,
                                fallback_reason=None,
                            )
                        else:
                            fallback_reason = "insufficient_geometry_calibration_samples"
                    else:
                        fallback_reason = scale_status  # e.g. prediction_anchor_spatial_inconsistency
                else:
                    fallback_reason = pool_status  # e.g. no_eligible_unmentioned_pois
            else:
                fallback_reason = "anchor_not_found_in_catalog"
        else:
            fallback_reason = f"landmark_{poi_status if poi_status else 'unresolved'}"
    else:
        fallback_reason = "landmark_geometry_disabled_or_no_parsed_entity"

    # -------------------------------------------------------------------------
    # Level 2: Candidate-Wise Empirical P90
    # -------------------------------------------------------------------------
    if cfg.enable_candidate_wise_calibration:
        cand_dict = calibration_dict.get("candidates", {}).get(c_type)
        if cand_dict and cand_dict.get("sample_count", 0) >= cfg.min_calibration_samples:
            r_cand = float(cand_dict["radius_m"])
            final_r = min(max(r_cand, r_floor), r_cap)
            return CalibrationResult(
                radius_m=round(final_r, 2),
                confidence_level=0.90,
                calibration_method="candidate_p90",
                calibration_tier=t_name,
                fallback_level="candidate_p90",
                sample_count=int(cand_dict["sample_count"]),
                geometry_scale_m=None,
                primary_landmark_poi_id=None,
                nearest_unmentioned_poi_id=None,
                nearest_unmentioned_distance_m=None,
                fallback_reason=fallback_reason,
            )

    # -------------------------------------------------------------------------
    # Level 3: Locality-Level Empirical P90
    # -------------------------------------------------------------------------
    if parsed_entity is not None:
        loc_id = parsed_entity.get("locality_id_resolved")
        if loc_id and pd.notna(loc_id):
            loc_dict = calibration_dict.get("localities", {}).get(str(loc_id).strip())
            if loc_dict and loc_dict.get("sample_count", 0) >= cfg.min_calibration_samples:
                r_loc = float(loc_dict["radius_m"])
                final_r = min(max(r_loc, r_floor), r_cap)
                return CalibrationResult(
                    radius_m=round(final_r, 2),
                    confidence_level=0.90,
                    calibration_method="locality_empirical",
                    calibration_tier=t_name,
                    fallback_level="locality_specific",
                    sample_count=int(loc_dict["sample_count"]),
                    geometry_scale_m=None,
                    primary_landmark_poi_id=None,
                    nearest_unmentioned_poi_id=None,
                    nearest_unmentioned_distance_m=None,
                    fallback_reason=fallback_reason,
                )

    # -------------------------------------------------------------------------
    # Level 4: Evidence-Tier Empirical Quantile Fallback
    # -------------------------------------------------------------------------
    tiers = calibration_dict.get("tiers", {})
    if t_name in tiers:
        t_data = tiers[t_name]
        r_tier = float(t_data.get("radius_m", 250.0))
        final_r = min(max(r_tier, r_floor), r_cap)
        return CalibrationResult(
            radius_m=round(final_r, 2),
            confidence_level=float(t_data.get("quantile", 0.90)),
            calibration_method="evidence_tier",
            calibration_tier=t_name,
            fallback_level=str(t_data.get("fallback_level", "tier_direct")),
            sample_count=int(t_data.get("residual_count", t_data.get("sample_count", 0))),
            geometry_scale_m=None,
            primary_landmark_poi_id=None,
            nearest_unmentioned_poi_id=None,
            nearest_unmentioned_distance_m=None,
            fallback_reason=fallback_reason,
        )

    # Operational Prior Fallback
    fallback_defaults = {
        PredictionStatus.DOORSTEP_VERIFIED.value: cfg.status_threshold_doorstep_r_m,
        PredictionStatus.STREET_RESOLVED.value: cfg.status_threshold_street_r_m,
        PredictionStatus.LOCALITY_BOUNDED.value: cfg.status_threshold_locality_r_m,
        PredictionStatus.BASELINE_FALLBACK.value: cfg.status_threshold_baseline_r_m,
    }
    r_default = fallback_defaults.get(t_name, 250.0)
    final_r = min(max(float(r_default), r_floor), r_cap)
    return CalibrationResult(
        radius_m=round(final_r, 2),
        confidence_level=0.90,
        calibration_method="evidence_tier",
        calibration_tier=t_name,
        fallback_level="operational_prior",
        sample_count=0,
        geometry_scale_m=None,
        primary_landmark_poi_id=None,
        nearest_unmentioned_poi_id=None,
        nearest_unmentioned_distance_m=None,
        fallback_reason=fallback_reason,
    )


def calibrate_prediction(
    status_tier: Union[PredictionStatus, str],
    calibration_dict: Dict[str, Any],
    selected_candidate_type: Optional[str] = None,
    predicted_coord: Optional[Tuple[float, float]] = None,
    parsed_entity: Optional[Dict[str, Any]] = None,
    df_pois: Optional[pd.DataFrame] = None,
    df_localities: Optional[pd.DataFrame] = None,
    has_credible_doorstep: bool = False,
    config: Optional[ModelConfig] = None,
    return_result_object: bool = False,
) -> Union[Tuple[float, float, str, int], CalibrationResult]:
    """Retrieve calibrated uncertainty radius and metadata.
    
    If selected_candidate_type is provided:
        Executes hierarchical production calibration:
        Level 0 (Direct Doorstep) -> Level 1 (Close-Landmark Geometry) ->
        Level 2 (Candidate P90) -> Level 3 (Locality P90) -> Level 4 (Evidence Tier).
        
    If selected_candidate_type is None:
        Preserves legacy Level 4 evidence-tier behavior for complete backward compatibility.
    """
    if selected_candidate_type is not None:
        p_coord = predicted_coord or (0.0, 0.0)
        res = calibrate_prediction_production(
            selected_candidate_type=selected_candidate_type,
            predicted_coord=p_coord,
            status_tier=status_tier,
            calibration_dict=calibration_dict,
            parsed_entity=parsed_entity,
            df_pois=df_pois,
            df_localities=df_localities,
            has_credible_doorstep=has_credible_doorstep,
            config=config,
        )
        if return_result_object:
            return res
        return (res.radius_m, res.confidence_level, res.fallback_level, res.sample_count)

    # Legacy evidence-tier fallback
    t_name = status_tier.value if isinstance(status_tier, PredictionStatus) else str(status_tier)
    tiers = calibration_dict.get("tiers", {})

    if t_name in tiers:
        t_data = tiers[t_name]
        return (
            float(t_data.get("radius_m", 250.0)),
            float(t_data.get("quantile", 0.90)),
            str(t_data.get("fallback_level", "tier_direct")),
            int(t_data.get("residual_count", t_data.get("sample_count", 0))),
        )

    # Fallback if tier not in calibration dictionary
    default_radii = {
        PredictionStatus.DOORSTEP_VERIFIED.value: 35.0,
        PredictionStatus.STREET_RESOLVED.value: 85.0,
        PredictionStatus.LOCALITY_BOUNDED.value: 250.0,
        PredictionStatus.BASELINE_FALLBACK.value: 400.0,
    }
    return default_radii.get(t_name, 250.0), 0.90, "operational_prior", 0
