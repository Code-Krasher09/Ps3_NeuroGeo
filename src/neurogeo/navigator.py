"""Offline Deterministic Navigator and Auditable Evidence Card Generator.

Authority: PS3_FINAL_ARCHITECTURE.md Section 4, PS3_MODEL_SPEC.md Section 8,
and PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md.

Design Principles:
- Purely offline operation: zero network calls, zero external routing APIs, zero LLMs.
- Deterministic geometry: uses repository's metric Cartesian coordinate convention
  while providing standard geodetic haversine for general reference.
- Uncertainty-aware and status-aware guidance: respects empirical P90 radii
  without overclaiming confidence.
- Negative evidence safety: field warnings (e.g. shifted borrower, untraceable address)
  are strictly flagged and never converted into positive doorstep pins.
- Immutability: consumes frozen Phase 4 predictions without altering scores,
  tiers, or coordinates.
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from neurogeo.schemas import (
    EvidenceCard,
    EvidenceCardData,
    NavigationInstruction,
    NavigationMode,
    ParsedAddressEntity,
    PredictionStatus,
    Phase4Prediction,
)


# Standard 8-sector compass mapping (45 degrees per sector)
SECTORS_8 = [
    "North",
    "North-East",
    "East",
    "South-East",
    "South",
    "South-West",
    "West",
    "North-West",
]

# Mapping candidate code to frozen standardized provenance source name
CANDIDATE_SOURCE_MAP = {
    "C1": "commercial_baseline",
    "C2": "direct_visit_medoid",
    "C3": "physical_poi",
    "C4": "spatial_neighbor_graph",
    "C5": "locality_centroid",
}

# Mapping prediction status tier to navigation mode
TIER_TO_MODE = {
    PredictionStatus.DOORSTEP_VERIFIED.value: NavigationMode.DOORSTEP,
    PredictionStatus.STREET_RESOLVED.value: NavigationMode.STREET,
    PredictionStatus.LOCALITY_BOUNDED.value: NavigationMode.LOCALITY,
    PredictionStatus.BASELINE_FALLBACK.value: NavigationMode.BASELINE,
}


# =============================================================================
# Core Geometric Functions
# =============================================================================

def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute great-circle distance between two WGS-84 points in meters.
    
    Formula:
        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c  (where R = 6,371,000 m)
        
    Note on Coordinate Convention:
        In the NeuroGeo PS3 domain, address coordinates are metric Cartesian
        coordinates in meters (x, y) centered at (0, 0) per town.
        For internal repo navigation, use euclidean_distance_m.
    """
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(earth_radius_m * c, 2)


def euclidean_distance_m(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute 2D Cartesian distance in meters for projected coordinates."""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    return round(math.sqrt(dx * dx + dy * dy), 2)


def bearing_degrees(
    from_x: float,
    from_y: float,
    to_x: float,
    to_y: float,
) -> float:
    """Compute deterministic navigation bearing in degrees in [0.0, 360.0).
    
    Coordinate convention:
        +Y represents North (0 deg)
        +X represents East (90 deg)
        -Y represents South (180 deg)
        -X represents West (270 deg)
        
    Formula: atan2(dx, dy) normalized to [0, 360).
    """
    dx = float(to_x) - float(from_x)
    dy = float(to_y) - float(from_y)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    deg = math.degrees(math.atan2(dx, dy))
    return round((deg % 360.0 + 360.0) % 360.0, 1)


def destination_offset(
    start_x: float,
    start_y: float,
    distance_m: float,
    bearing_deg: float,
) -> Tuple[float, float]:
    """Compute destination Cartesian coordinates from start point, distance, and bearing."""
    rad = math.radians(bearing_deg)
    dx = float(distance_m) * math.sin(rad)
    dy = float(distance_m) * math.cos(rad)
    return (round(float(start_x) + dx, 2), round(float(start_y) + dy, 2))


def point_within_radius(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    radius_m: float,
) -> bool:
    """Check if current position is within the specified confidence radius of target."""
    dist = euclidean_distance_m(current_x, current_y, target_x, target_y)
    return bool(dist <= float(radius_m))


def coarse_direction(bearing_deg: float) -> str:
    """Map bearing in degrees deterministically to one of 8 compass sectors."""
    norm_deg = (float(bearing_deg) % 360.0 + 360.0) % 360.0
    sector_idx = int((norm_deg + 22.5) // 45.0) % 8
    return SECTORS_8[sector_idx]


# =============================================================================
# Helper Utilities
# =============================================================================

def _extract_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute from dataclass, namedtuple, or dictionary safely."""
    if hasattr(obj, key):
        val = getattr(obj, key)
        return default if val is None else val
    if isinstance(obj, dict):
        val = obj.get(key, default)
        return default if val is None else val
    return default


def _standardize_evidence_sources(raw_sources: Any) -> List[str]:
    """Convert raw candidate codes (e.g. 'C4,C1' or ['C4']) to standardized names."""
    if isinstance(raw_sources, str):
        tokens = [s.strip().upper() for s in raw_sources.split(",") if s.strip()]
    elif isinstance(raw_sources, (list, tuple)):
        tokens = [str(s).strip().upper() for s in raw_sources if str(s).strip()]
    else:
        tokens = []

    res: List[str] = []
    for tok in tokens:
        mapped = CANDIDATE_SOURCE_MAP.get(tok, tok.lower())
        if mapped not in res:
            res.append(mapped)
    return res if res else ["unknown_source"]


# =============================================================================
# Navigation Engine
# =============================================================================

def navigate(
    current_x: float,
    current_y: float,
    prediction: Union[Phase4Prediction, Dict[str, Any]],
    negative_evidence: Optional[List[str]] = None,
) -> NavigationInstruction:
    """Generate deterministic, uncertainty-aware offline navigation instructions.
    
    Authority: PS3_FINAL_ARCHITECTURE.md Section 4 and PS3_MODEL_SPEC.md Section 8.
    
    Guarantees:
    - Pure offline computation.
    - Status-aware phrasing: never claims non-doorstep predictions are verified doorsteps.
    - Uncertainty-aware: uses empirical P90 radius without asserting guaranteed boundaries.
    - Field warnings: surfaces negative field evidence if present.
    """
    target_x = float(_extract_attr(prediction, "predicted_x", 0.0))
    target_y = float(_extract_attr(prediction, "predicted_y", 0.0))
    status_tier = str(_extract_attr(prediction, "status_tier", PredictionStatus.BASELINE_FALLBACK.value))
    radius_m = float(_extract_attr(prediction, "confidence_radius_m", 250.0))

    dist_m = euclidean_distance_m(current_x, current_y, target_x, target_y)
    bearing_deg = bearing_degrees(current_x, current_y, target_x, target_y)
    direction = coarse_direction(bearing_deg)
    within_radius = point_within_radius(current_x, current_y, target_x, target_y, radius_m)

    # Map status tier to navigation mode
    nav_mode = TIER_TO_MODE.get(status_tier, NavigationMode.BASELINE)

    # 1. Synthesize status-aware, uncertainty-aware directional instruction
    if dist_m == 0.0:
        instruction = "You have arrived at the predicted target coordinate. Verify premises locally."
    elif within_radius:
        if nav_mode == NavigationMode.DOORSTEP:
            instruction = (
                f"You are within the estimated doorstep uncertainty region (~{radius_m:.0f} m). "
                f"Verify the premises at this doorstep pin."
            )
        elif nav_mode == NavigationMode.STREET:
            instruction = (
                f"You are within the estimated street uncertainty region (~{radius_m:.0f} m). "
                f"The target premises are on this street; verify house numbers or recipient name locally."
            )
        elif nav_mode == NavigationMode.LOCALITY:
            instruction = (
                f"You are within the estimated locality boundary (~{radius_m:.0f} m). "
                f"Exact premises are uncertain within this neighborhood; seek local landmark guidance."
            )
        else:  # BASELINE
            instruction = (
                f"You are within the commercial baseline estimate radius (~{radius_m:.0f} m). "
                f"Confidence is low; verify address text and physical premises carefully."
            )
    else:
        dir_lower = direction.lower()
        if nav_mode == NavigationMode.DOORSTEP:
            instruction = (
                f"Proceed {dir_lower} for approximately {dist_m:.0f} m toward the verified doorstep location. "
                f"Estimated empirical P90 uncertainty radius: ~{radius_m:.0f} m."
            )
        elif nav_mode == NavigationMode.STREET:
            instruction = (
                f"Proceed {dir_lower} for approximately {dist_m:.0f} m toward the resolved street location. "
                f"Final premises should be verified locally (estimated empirical P90 radius: ~{radius_m:.0f} m)."
            )
        elif nav_mode == NavigationMode.LOCALITY:
            instruction = (
                f"Proceed {dir_lower} for approximately {dist_m:.0f} m toward the locality estimate. "
                f"Exact premises are uncertain within the estimated empirical P90 radius (~{radius_m:.0f} m)."
            )
        else:  # BASELINE
            instruction = (
                f"Proceed {dir_lower} for approximately {dist_m:.0f} m toward the baseline location estimate. "
                f"Confidence is low; verify the address carefully (estimated empirical P90 radius: ~{radius_m:.0f} m)."
            )

    # 2. Synthesize status-aware caution and field warnings
    cautions: List[str] = []

    # Negative field evidence warnings (prioritized first)
    if negative_evidence:
        for ev in negative_evidence:
            ev_str = str(ev).strip()
            ev_lower = ev_str.lower()
            if "shifted" in ev_lower:
                cautions.append("WARNING: Field evidence indicates the borrower may have shifted from this address.")
            elif "not_traceable" in ev_lower or "not traceable" in ev_lower:
                cautions.append("WARNING: Previous field search reported the address as not traceable.")
            elif ev_str:
                cautions.append(f"WARNING: {ev_str}")

    # Status-tier caveats
    if nav_mode == NavigationMode.STREET:
        cautions.append("Final premises must be verified locally; prediction is street-level centroid, not confirmed doorstep.")
    elif nav_mode == NavigationMode.LOCALITY:
        cautions.append(f"Coarse locality centroid; premises are unverified and may lie anywhere within the ~{radius_m:.0f} m neighborhood.")
    elif nav_mode == NavigationMode.BASELINE:
        cautions.append("Low-confidence commercial baseline fallback; coordinate may have substantial error and is unverified by field visits.")

    caution_text = " ".join(cautions) if cautions else None

    return NavigationInstruction(
        current_x=round(float(current_x), 2),
        current_y=round(float(current_y), 2),
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
        distance_m=dist_m,
        bearing_deg=bearing_deg,
        direction=direction,
        within_confidence_radius=within_radius,
        confidence_radius_m=radius_m,
        status_tier=status_tier,
        navigation_mode=nav_mode.value,
        instruction=instruction,
        caution=caution_text,
    )


# =============================================================================
# Evidence Card & Human Explanation
# =============================================================================

def explain_prediction(
    prediction: Union[Phase4Prediction, Dict[str, Any]],
    evidence_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate concise, deterministic, human-readable explanation for field agent."""
    ctx = evidence_context or {}
    status_tier = str(_extract_attr(prediction, "status_tier", PredictionStatus.BASELINE_FALLBACK.value))
    selected_cand = str(_extract_attr(prediction, "selected_candidate_type", _extract_attr(prediction, "selected_candidate", "C1"))).upper()
    radius_m = float(_extract_attr(prediction, "confidence_radius_m", 250.0))
    fb_level = str(_extract_attr(prediction, "calibration_fallback_level", "tier_direct"))

    # Status intro
    parts = [
        f"Prediction is {status_tier} with an estimated {radius_m:.0f} m empirical P90 uncertainty radius "
        f"(calibration fallback: {fb_level})."
    ]

    # Primary candidate evidence attribution
    if selected_cand == "C2" or status_tier == PredictionStatus.DOORSTEP_VERIFIED.value:
        parts.append(
            "The location is anchored to a verified doorstep medoid derived from direct high-integrity field visits."
        )
    elif selected_cand == "C4" or status_tier == PredictionStatus.STREET_RESOLVED.value:
        parts.append(
            "The location is anchored to a hierarchical spatial graph neighbor consensus from verified visits "
            "on the same street segment."
        )
    elif selected_cand == "C3":
        landmark_name = ctx.get("landmark_name") or ctx.get("landmark_raw")
        if landmark_name:
            parts.append(f"The location is anchored to physical landmark POI '{landmark_name}' resolved from address text.")
        else:
            parts.append("The location is anchored to a physical landmark POI resolved from address text.")
    elif selected_cand == "C5" or status_tier == PredictionStatus.LOCALITY_BOUNDED.value:
        loc_name = ctx.get("locality_name") or ctx.get("locality_id_resolved")
        if loc_name:
            parts.append(f"The location represents the locality centroid for neighborhood '{loc_name}' as direct visits were unavailable.")
        else:
            parts.append("The location represents the locality neighborhood centroid as direct doorstep and street evidence were unavailable.")
    else:  # C1 / BASELINE_FALLBACK
        parts.append(
            "The location falls back to commercial baseline geocoding due to absence of verified field check-ins or street clusters."
        )

    # Multi-source agreement
    sources_str = str(_extract_attr(prediction, "evidence_sources", ""))
    std_sources = _standardize_evidence_sources(sources_str)
    if len(std_sources) > 1:
        sources_fmt = ", ".join(std_sources)
        parts.append(f"Multi-source agreement was confirmed across: {sources_fmt}.")

    # Negative field evidence
    neg_ev = ctx.get("negative_evidence", [])
    if neg_ev:
        neg_text = "; ".join(str(e) for e in neg_ev)
        parts.append(f"CAUTION: Negative field signals recorded ({neg_text}). Verify occupancy before proceeding.")

    return " ".join(parts)


def build_evidence_card(
    prediction: Union[Phase4Prediction, Dict[str, Any]],
    evidence_context: Optional[Dict[str, Any]] = None,
) -> EvidenceCardData:
    """Build a deterministic, auditable evidence card detailing feature contributions."""
    ctx = evidence_context or {}

    address_id = str(_extract_attr(prediction, "address_id", "UNKNOWN"))
    pred_x = float(_extract_attr(prediction, "predicted_x", 0.0))
    pred_y = float(_extract_attr(prediction, "predicted_y", 0.0))
    status_tier = str(_extract_attr(prediction, "status_tier", PredictionStatus.BASELINE_FALLBACK.value))
    selected_candidate = str(_extract_attr(prediction, "selected_candidate_type", _extract_attr(prediction, "selected_candidate", "C1"))).upper()
    radius_m = float(_extract_attr(prediction, "confidence_radius_m", 250.0))
    cal_tier = str(_extract_attr(prediction, "calibration_tier", status_tier))
    cal_sample_count = int(_extract_attr(prediction, "calibration_sample_count", 0))
    cal_fallback_level = str(_extract_attr(prediction, "calibration_fallback_level", "tier_direct"))
    score = float(_extract_attr(prediction, "selected_score", _extract_attr(prediction, "score", 0.0)))
    score_margin = float(_extract_attr(prediction, "score_margin", 0.0))
    fit_scope = str(_extract_attr(prediction, "fit_scope", "benchmark_train"))
    eval_scope = str(_extract_attr(prediction, "evaluation_scope", "eval_test"))

    # Evidence sources mapping
    raw_sources = _extract_attr(prediction, "evidence_sources", selected_candidate)
    evidence_sources = _standardize_evidence_sources(raw_sources)

    # Positive evidence extraction
    positive_evidence: List[str] = []
    if selected_candidate == "C2" or status_tier == PredictionStatus.DOORSTEP_VERIFIED.value:
        positive_evidence.append("Direct high-integrity field visits confirmed at doorstep")
        positive_evidence.append("Sanitized doorstep medoid kernel")
    elif selected_candidate == "C4" or status_tier == PredictionStatus.STREET_RESOLVED.value:
        positive_evidence.append("Hierarchical street graph neighbor consensus")
        n_cnt = ctx.get("neighbor_count", 0)
        if n_cnt > 0:
            positive_evidence.append(f"Supported by {n_cnt} verified neighbor check-in accounts")
        else:
            positive_evidence.append("Supported by verified street neighbor accounts")
    elif selected_candidate == "C3":
        positive_evidence.append("Anchored to resolved physical POI landmark")
    elif selected_candidate == "C5" or status_tier == PredictionStatus.LOCALITY_BOUNDED.value:
        positive_evidence.append("Locality polygon centroid from parsed address entity")
    else:
        positive_evidence.append("Commercial baseline geocoder coordinate prior")

    # Negative evidence extraction
    negative_evidence: List[str] = []
    # Check context direct visits or explicit negative evidence list
    direct_visits = ctx.get("direct_visits", [])
    if isinstance(direct_visits, list):
        for v in direct_visits:
            outcome = str(v.get("outcome", "") if isinstance(v, dict) else getattr(v, "outcome", "")).strip().lower()
            if outcome == "neighbour_says_shifted":
                if "Neighbor reported borrower shifted from address" not in negative_evidence:
                    negative_evidence.append("Neighbor reported borrower shifted from address")
            elif outcome == "address_not_traceable":
                if "Previous field visit reported address as not traceable" not in negative_evidence:
                    negative_evidence.append("Previous field visit reported address as not traceable")

    # Context override if negative_evidence explicitly provided
    if "negative_evidence" in ctx and isinstance(ctx["negative_evidence"], list):
        for ne in ctx["negative_evidence"]:
            if str(ne) not in negative_evidence:
                negative_evidence.append(str(ne))

    # Caution synthesis
    cautions: List[str] = []
    if negative_evidence:
        for ne in negative_evidence:
            if "shifted" in ne.lower():
                cautions.append("Field evidence indicates borrower may have shifted from address.")
            elif "not traceable" in ne.lower():
                cautions.append("Previous field search reported address as not traceable.")
            else:
                cautions.append(ne)

    if status_tier == PredictionStatus.STREET_RESOLVED.value:
        cautions.append("Street-level estimate; final doorstep must be verified locally.")
    elif status_tier == PredictionStatus.LOCALITY_BOUNDED.value:
        cautions.append(f"Locality-level centroid; uncertainty radius ~{radius_m:.0f} m.")
    elif status_tier == PredictionStatus.BASELINE_FALLBACK.value:
        cautions.append("Unverified commercial baseline fallback with low confidence.")

    caution = " ".join(cautions) if cautions else None

    # Calibration provenance and evidence
    cal_method = _extract_attr(prediction, "calibration_method", None)
    geom_scale = _extract_attr(prediction, "geometry_scale_m", None)
    pri_poi = _extract_attr(prediction, "primary_landmark_poi_id", None)
    near_poi = _extract_attr(prediction, "nearest_unmentioned_poi_id", None)
    near_dist = _extract_attr(prediction, "nearest_unmentioned_distance_m", None)
    cal_fb_reason = _extract_attr(prediction, "calibration_fallback_reason", None)

    if cal_method == "landmark_geometry":
        scale_txt = f" ({float(geom_scale):.1f} m)" if geom_scale is not None and pd.notna(geom_scale) else ""
        positive_evidence.append(f"Confidence radius calibrated via close-landmark spatial scale{scale_txt}")
    elif cal_method == "candidate_p90":
        positive_evidence.append(f"Confidence radius calibrated via candidate-wise empirical P90 ({selected_candidate})")
    elif cal_method == "locality_empirical":
        positive_evidence.append("Confidence radius calibrated via locality-level empirical P90")

    # Provenance block
    provenance = {
        "fit_scope": fit_scope,
        "evaluation_scope": eval_scope,
        "calibration_source": "reports/metrics/calibration_quantiles.json",
        "calibration_scope": "scv_leave_one_out",
        "offline_mode": True,
        "external_network_accessed": False,
    }
    if cal_method is not None:
        provenance["calibration_method"] = str(cal_method)
    if geom_scale is not None and pd.notna(geom_scale):
        provenance["geometry_scale_m"] = round(float(geom_scale), 2)
    if pri_poi is not None and pd.notna(pri_poi):
        provenance["primary_landmark_poi_id"] = str(pri_poi)
    if near_poi is not None and pd.notna(near_poi):
        provenance["nearest_unmentioned_poi_id"] = str(near_poi)
    if near_dist is not None and pd.notna(near_dist):
        provenance["nearest_unmentioned_distance_m"] = round(float(near_dist), 2)
    if cal_fb_reason is not None and pd.notna(cal_fb_reason):
        provenance["calibration_fallback_reason"] = str(cal_fb_reason)

    explanation = explain_prediction(prediction, evidence_context=ctx)

    return EvidenceCardData(
        address_id=address_id,
        status_tier=status_tier,
        selected_candidate=selected_candidate,
        predicted_coordinates=(round(pred_x, 2), round(pred_y, 2)),
        confidence_radius_m=radius_m,
        calibration_tier=cal_tier,
        calibration_sample_count=cal_sample_count,
        calibration_fallback_level=cal_fallback_level,
        score=round(score, 4),
        score_margin=round(score_margin, 4),
        evidence_sources=evidence_sources,
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
        caution=caution,
        provenance=provenance,
        explanation=explanation,
    )


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================

def generate_navigation_guidance(
    entity: ParsedAddressEntity,
    resolved_poi_name: Optional[str] = None,
) -> str:
    """Generate deterministic, human-readable offline turn guidance from parsed entities."""
    parts = []
    if entity.street_normalized:
        parts.append(f"Navigate to {entity.street_normalized}")
    elif entity.locality_name_raw:
        parts.append(f"Navigate to {entity.locality_name_raw}")
    else:
        parts.append("Navigate to target locality")

    if resolved_poi_name:
        parts.append(f"near landmark '{resolved_poi_name}'")
    return " ".join(parts) + "."


def format_evidence_card(card: Union[EvidenceCard, EvidenceCardData]) -> dict:
    """Format structured evidence card for field audit and explainability."""
    if hasattr(card, "to_dict"):
        return card.to_dict()
    if isinstance(card, dict):
        return card
    return {k: getattr(card, k) for k in dir(card) if not k.startswith("_")}
