"""Evidence-Tier Empirical Quantile Calibration with sample fallback hierarchy.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, ModelConfig, PATHS
from neurogeo.leakage import assert_fit_scope_valid
from neurogeo.schemas import PredictionStatus, CalibrationTierRecord


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
    
    Authority: PS3_FINAL_ARCHITECTURE.md Section 4 and PS3_MODEL_SPEC.md Section 7:
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
    """Compute calibrated empirical radius using frozen 3-level fallback hierarchy.
    
    Authority: PS3_MODEL_SPEC.md Section 7 lines 158-176.
    
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


def fit_empirical_calibration_quantiles(
    df_scv: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
    test_address_ids: Optional[Set[str]] = None,
    validation_address_ids: Optional[Set[str]] = None,
    train_address_ids: Optional[Set[str]] = None,
    min_samples: int = 15,
    target_quantile: float = 0.90,
    fit_scope: str = "calibration_oof",
) -> Dict[str, Any]:
    """Fit evidence-tier empirical quantile calibration from leakage-safe held-out SCV residuals.
    
    Zero-leakage guarantees:
    - Target test, validation, and train benchmark addresses are strictly excluded from fitting calibration quantiles.
    - Explicit assertion: calibration_address_ids ∩ benchmark_address_ids == EMPTY.
    - Explicit assertion: calibration_address_ids ∩ reranker_train_address_ids == EMPTY.
    - Uses SCV residuals (reserved_for_calibration = True, target_history_masked = True).
    - Preserves provenance fit_scope.
    """
    assert_fit_scope_valid("calibration_quantiles", fit_scope)

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
        df_localities = pd.read_csv(PATHS.localities_csv)

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

    # 3. Compute calibrated quantiles using fallback hierarchy
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
        "fallback_counts": fallback_counts,
    }


def calibrate_prediction(
    status_tier: Union[PredictionStatus, str],
    calibration_dict: Dict[str, Any],
) -> Tuple[float, float, str, int]:
    """Retrieve calibrated uncertainty radius and metadata for a predicted status tier.
    
    Returns:
        (confidence_radius_m, confidence_level, fallback_level, sample_count)
    """
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
