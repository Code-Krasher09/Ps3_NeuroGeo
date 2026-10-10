"""Spatial candidate generation (C1-C5), coordinate deduplication, and linear utility reranker.

Authority: NEUROGEO_FINAL_ARCHITECTURE.md, NEUROGEO_MODEL_SPEC.md, and NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

from dataclasses import replace
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, ModelConfig
from neurogeo.schemas import ParsedAddressEntity, SpatialCandidate


# Candidate source priority for deterministic tie-breaking and deduplication
CANDIDATE_PRIORITY = {
    "C2": 1,  # Direct visit medoid (highest prior evidence)
    "C4": 2,  # Hierarchical neighbor street centroid
    "C3": 3,  # Physical POI anchor
    "C1": 4,  # Commercial baseline prior
    "C5": 5,  # Locality centroid (broadest fallback)
}


def compute_euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute 2D Euclidean distance in local Cartesian meters."""
    return float(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def _val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to extract attribute or key from dict, Series, or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if isinstance(obj, pd.Series):
        val = obj.get(key, default)
        return default if pd.isna(val) else val
    if hasattr(obj, key):
        return getattr(obj, key, default)
    return default


def generate_candidates(
    address_id: str,
    baseline_row: Optional[Union[Dict[str, Any], pd.Series]] = None,
    medoid_row: Optional[Union[Dict[str, Any], pd.Series]] = None,
    poi_row: Optional[Union[Dict[str, Any], pd.Series]] = None,
    c4_result: Optional[Dict[str, Any]] = None,
    locality_row: Optional[Union[Dict[str, Any], pd.Series]] = None,
    parsed_entity: Optional[Union[ParsedAddressEntity, Dict[str, Any]]] = None,
    fit_scope: str = "production_all",
    is_held_out: bool = False,
) -> List[SpatialCandidate]:
    """Generate spatial candidates (C1-C5) according to NEUROGEO_MODEL_SPEC.md.
    
    Zero coordinate fabrication: If evidence is missing, available = False and coordinates = None.
    Zero leakage: If is_held_out is True, C2 is strictly unavailable.
    """
    candidates: List[SpatialCandidate] = []

    # ==========================================
    # C1: Commercial Baseline Prior
    # ==========================================
    c1_x = _val(baseline_row, "geocoder_x")
    c1_y = _val(baseline_row, "geocoder_y")
    c1_avail = c1_x is not None and c1_y is not None and not (pd.isna(c1_x) or pd.isna(c1_y))
    
    c1_x_f = float(c1_x) if c1_avail else None
    c1_y_f = float(c1_y) if c1_avail else None
    c1_prec = str(_val(baseline_row, "precision", "unknown")).lower()
    c1_prov = str(_val(baseline_row, "provider", "commercial_geocoder"))

    candidates.append(
        SpatialCandidate(
            candidate_id=f"{address_id}::C1",
            candidate_type="C1",
            x=c1_x_f,
            y=c1_y_f,
            available=bool(c1_avail),
            source_type="commercial_baseline",
            prior_confidence=0.50 if c1_prec in {"rooftop", "street"} else 0.20,
            evidence_metadata={
                "precision": c1_prec,
                "provider": c1_prov,
                "source": "baseline_geocodes.csv",
            },
            fit_scope=fit_scope,
        )
    )

    # ==========================================
    # C2: Direct Visit Medoid
    # ==========================================
    # Strictly unavailable if evaluating held-out cold start!
    if is_held_out:
        c2_avail = False
        c2_x_f, c2_y_f = None, None
        c2_meta = {"held_out_suppressed": True}
    else:
        c2_x = _val(medoid_row, "doorstep_medoid_x")
        c2_y = _val(medoid_row, "doorstep_medoid_y")
        has_cred = bool(_val(medoid_row, "has_credible_doorstep", False))
        c2_avail = has_cred and c2_x is not None and c2_y is not None and not (pd.isna(c2_x) or pd.isna(c2_y))
        c2_x_f = float(c2_x) if c2_avail else None
        c2_y_f = float(c2_y) if c2_avail else None
        c2_meta = {
            "has_credible_doorstep": has_cred,
            "credible_visit_count": int(_val(medoid_row, "credible_visit_count", 0)),
            "source": "address_doorstep_medoids.parquet",
        }

    candidates.append(
        SpatialCandidate(
            candidate_id=f"{address_id}::C2",
            candidate_type="C2",
            x=c2_x_f,
            y=c2_y_f,
            available=bool(c2_avail),
            source_type="direct_visit_medoid",
            prior_confidence=0.95 if c2_avail else 0.0,
            evidence_metadata=c2_meta,
            fit_scope=fit_scope,
        )
    )

    # ==========================================
    # C3: Physical POI Anchor
    # ==========================================
    # Actual physical POI coordinates without arbitrary coordinate offsets!
    c3_x = _val(poi_row, "x")
    c3_y = _val(poi_row, "y")
    c3_avail = c3_x is not None and c3_y is not None and not (pd.isna(c3_x) or pd.isna(c3_y))
    c3_x_f = float(c3_x) if c3_avail else None
    c3_y_f = float(c3_y) if c3_avail else None

    relation = _val(parsed_entity, "spatial_relation")
    landmark_raw = _val(parsed_entity, "landmark_raw")
    poi_name = _val(poi_row, "name")
    poi_id = _val(poi_row, "poi_id")
    landmark_type = _val(poi_row, "landmark_type")

    candidates.append(
        SpatialCandidate(
            candidate_id=f"{address_id}::C3",
            candidate_type="C3",
            x=c3_x_f,
            y=c3_y_f,
            available=bool(c3_avail),
            source_type="physical_poi_anchor",
            prior_confidence=0.75 if c3_avail else 0.0,
            evidence_metadata={
                "poi_id": poi_id,
                "poi_name": poi_name,
                "landmark_type": landmark_type,
                "landmark_raw": landmark_raw,
                "spatial_relation": relation,
                "relation_offset_applied": False,  # Explicit contract verification
                "source": "landmarks_poi.csv",
            },
            fit_scope=fit_scope,
        )
    )

    # ==========================================
    # C4: Hierarchical Knowledge-Transfer Centroid
    # ==========================================
    # Must be same town, same resolved locality, same normalized street, with credible neighbors
    c4_x = _val(c4_result, "c4_x")
    c4_y = _val(c4_result, "c4_y")
    n_count = int(_val(c4_result, "neighbor_count", 0))
    c4_avail = c4_x is not None and c4_y is not None and n_count > 0 and not (pd.isna(c4_x) or pd.isna(c4_y))
    c4_x_f = float(c4_x) if c4_avail else None
    c4_y_f = float(c4_y) if c4_avail else None

    candidates.append(
        SpatialCandidate(
            candidate_id=f"{address_id}::C4",
            candidate_type="C4",
            x=c4_x_f,
            y=c4_y_f,
            available=bool(c4_avail),
            source_type="spatial_graph_neighbor_centroid",
            prior_confidence=0.85 if (c4_avail and n_count >= 2) else (0.65 if c4_avail else 0.0),
            evidence_metadata={
                "neighbor_count": n_count,
                "total_neighbor_weight": float(_val(c4_result, "total_neighbor_weight", 0.0)),
                "neighbor_address_ids": _val(c4_result, "neighbor_address_ids", []),
                "town_id": _val(c4_result, "town_id"),
                "locality_id": _val(c4_result, "locality_id"),
                "street_normalized": _val(c4_result, "street_normalized"),
                "source": "spatial_graph.parquet",
            },
            fit_scope=fit_scope,
        )
    )

    # ==========================================
    # C5: Locality Centroid
    # ==========================================
    c5_x = _val(locality_row, "centroid_x")
    c5_y = _val(locality_row, "centroid_y")
    c5_avail = c5_x is not None and c5_y is not None and not (pd.isna(c5_x) or pd.isna(c5_y))
    c5_x_f = float(c5_x) if c5_avail else None
    c5_y_f = float(c5_y) if c5_avail else None

    candidates.append(
        SpatialCandidate(
            candidate_id=f"{address_id}::C5",
            candidate_type="C5",
            x=c5_x_f,
            y=c5_y_f,
            available=bool(c5_avail),
            source_type="locality_centroid",
            prior_confidence=0.35 if c5_avail else 0.0,
            evidence_metadata={
                "locality_id": _val(locality_row, "locality_id"),
                "locality_name": _val(locality_row, "locality_name"),
                "source": "localities.csv",
            },
            fit_scope=fit_scope,
        )
    )

    return candidates


def deduplicate_candidates(
    candidates: List[SpatialCandidate],
    tolerance_m: float = 5.0,
) -> List[SpatialCandidate]:
    """Deterministically deduplicate candidate coordinates within tolerance_m.
    
    In accordance with NEUROGEO_MODEL_SPEC.md Section 5:
    - If two candidates resolve to effectively identical coordinates (dist <= tolerance_m),
      do NOT silently double-count evidence.
    - Keep candidate provenance explicit.
    - Preserve candidate type information in agreed_sources so evidence card reflects agreement.
    - Tie-breaking retains candidate with higher prior priority (C2 > C4 > C3 > C1 > C5).
    - Candidates in the same locality that are > tolerance_m apart are NOT merged.
    """
    avail = [c for c in candidates if c.available and c.x is not None and c.y is not None]
    if not avail:
        return []

    # Sort available candidates by priority (lowest integer priority rank = highest precedence)
    sorted_avail = sorted(avail, key=lambda c: CANDIDATE_PRIORITY.get(c.candidate_type, 99))

    kept: List[SpatialCandidate] = []

    for cand in sorted_avail:
        merged = False
        for i, existing in enumerate(kept):
            dist = compute_euclidean_distance(cand.x, cand.y, existing.x, existing.y)
            if dist <= tolerance_m:
                # Merge into existing: update agreed_sources
                merged_sources = list(existing.agreed_sources)
                if existing.candidate_type not in merged_sources:
                    merged_sources.append(existing.candidate_type)
                if cand.candidate_type not in merged_sources:
                    merged_sources.append(cand.candidate_type)
                
                updated_meta = dict(existing.evidence_metadata)
                updated_meta["agreed_candidate_types"] = tuple(merged_sources)
                updated_meta["dedup_distance_m"] = round(dist, 2)

                kept[i] = replace(
                    existing,
                    agreed_sources=tuple(merged_sources),
                    evidence_metadata=updated_meta,
                )
                merged = True
                break

        if not merged:
            init_sources = (cand.candidate_type,)
            init_meta = dict(cand.evidence_metadata)
            init_meta["agreed_candidate_types"] = init_sources
            kept.append(
                replace(
                    cand,
                    agreed_sources=init_sources,
                    evidence_metadata=init_meta,
                )
            )

    return kept


def extract_candidate_features(
    candidate: SpatialCandidate,
    parsed_entity: Optional[Union[ParsedAddressEntity, Dict[str, Any]]] = None,
    direct_visits: Optional[List[Dict[str, Any]]] = None,
    neighbor_records: Optional[List[Dict[str, Any]]] = None,
    baseline_coord: Optional[Tuple[float, float]] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, float]:
    """Extract deterministic, bounded, leakage-safe features for a single candidate.
    
    Formula from NEUROGEO_MODEL_SPEC.md Section 6:
    1. TextMatch in [0, 1]
    2. DirectEvidence in [0, 1]
    3. NeighborSupport in [0, 1]
    4. BaselineAgreement in [0, 1]
    5. NegativeSearchPenalty in [0, 1]
    """
    cfg = config or DEFAULT_CONFIG
    if not candidate.available or candidate.x is None or candidate.y is None:
        return {
            "TextMatch": 0.0,
            "DirectEvidence": 0.0,
            "NeighborSupport": 0.0,
            "BaselineAgreement": 0.0,
            "NegativeSearchPenalty": 0.0,
        }

    cx, cy = candidate.x, candidate.y
    c_type = candidate.candidate_type
    agreed = set(candidate.agreed_sources) if candidate.agreed_sources else {c_type}

    # ----------------------------------------------------
    # 1. TextMatch in [0, 1]
    # ----------------------------------------------------
    # Measures compatibility between address text parsing and candidate origin
    text_scores: List[float] = []
    
    if "C1" in agreed:
        prec = str(candidate.evidence_metadata.get("precision", "unknown")).lower()
        if prec == "rooftop":
            text_scores.append(1.0)
        elif prec == "street":
            text_scores.append(0.85)
        elif prec == "locality":
            text_scores.append(0.50)
        elif prec == "pincode":
            text_scores.append(0.20)
        else:
            text_scores.append(0.10)

    if "C2" in agreed:
        # Direct visit for the target address itself represents exact address entity match
        text_scores.append(1.0)

    if "C3" in agreed:
        # Match score between parsed landmark and resolved POI
        if parsed_entity is not None:
            status = getattr(parsed_entity, "poi_resolution_status", "unresolved") if hasattr(parsed_entity, "poi_resolution_status") else parsed_entity.get("poi_resolution_status", "unresolved")
            rel = getattr(parsed_entity, "spatial_relation", None) if hasattr(parsed_entity, "spatial_relation") else parsed_entity.get("spatial_relation")
            base_score = 0.90 if status == "resolved" else 0.65
            # Relational terms modulate textual compatibility without coordinate offset
            if rel in {"near", "beside", "opposite", "behind"}:
                base_score = min(1.0, base_score + 0.05)
            text_scores.append(base_score)
        else:
            text_scores.append(0.80)

    if "C4" in agreed:
        # Street extraction match
        text_scores.append(0.90)

    if "C5" in agreed:
        # Locality resolution match
        text_scores.append(0.70)


    text_match = float(max(text_scores)) if text_scores else 0.50
    text_match = max(0.0, min(1.0, text_match))

    # ----------------------------------------------------
    # 2. DirectEvidence in [0, 1]
    # ----------------------------------------------------
    # Sum_{v in V(a)} I(v) * S(v) * exp(-||C_k - X_v||^2 / (2 * sigma^2))
    # Strictly zero if cold-start / unvisited!
    direct_evidence = 0.0
    if direct_visits:
        sigma = cfg.direct_evidence_sigma_m
        sum_ev = 0.0
        for v in direct_visits:
            iv = float(v.get("visit_integrity_score", 0.0))
            sv = float(v.get("spatial_evidence_strength", 1.0))
            vx = float(v.get("checkin_x", 0.0))
            vy = float(v.get("checkin_y", 0.0))
            dist = compute_euclidean_distance(cx, cy, vx, vy)
            weight = iv * sv * math.exp(-(dist ** 2) / (2.0 * (sigma ** 2)))
            sum_ev += weight
        # Bounded into [0, 1] via tanh
        direct_evidence = float(math.tanh(sum_ev / 2.0))

    # ----------------------------------------------------
    # 3. NeighborSupport in [0, 1]
    # ----------------------------------------------------
    # Sum_{j in N(a)} w_j * exp(-||C_k - X_j||^2 / (2 * sigma_neigh^2)) / total_w
    neighbor_support = 0.0
    if neighbor_records:
        sigma_n = cfg.neighbor_support_sigma_m
        weighted_sum = 0.0
        total_w = sum(float(n.get("weight", 1.0)) for n in neighbor_records)
        if total_w > 0:
            for n in neighbor_records:
                w_j = float(n.get("weight", 1.0))
                nx = float(n.get("x", 0.0))
                ny = float(n.get("y", 0.0))
                dist = compute_euclidean_distance(cx, cy, nx, ny)
                weighted_sum += w_j * math.exp(-(dist ** 2) / (2.0 * (sigma_n ** 2)))
            neighbor_support = float(weighted_sum / total_w)
            neighbor_support = max(0.0, min(1.0, neighbor_support))

    # ----------------------------------------------------
    # 4. BaselineAgreement in [0, 1]
    # ----------------------------------------------------
    # exp(-||C_k - X_base||^2 / (2 * sigma_base^2))
    baseline_agreement = 0.0
    if baseline_coord is not None and not (pd.isna(baseline_coord[0]) or pd.isna(baseline_coord[1])):
        sigma_b = cfg.baseline_agreement_sigma_m
        dist_b = compute_euclidean_distance(cx, cy, baseline_coord[0], baseline_coord[1])
        baseline_agreement = float(math.exp(-(dist_b ** 2) / (2.0 * (sigma_b ** 2))))
        baseline_agreement = max(0.0, min(1.0, baseline_agreement))

    # ----------------------------------------------------
    # 5. NegativeSearchPenalty in [0, 1]
    # ----------------------------------------------------
    # Derived from negative field visit outcomes: 'address_not_traceable', 'neighbour_says_shifted'.
    # IMPORTANT: neighbour_says_shifted is address-state evidence indicating the borrower
    # has moved/shifted; it must NOT be treated as positive doorstep evidence.
    negative_penalty = 0.0
    if direct_visits:
        r_neg = cfg.negative_search_radius_m
        pen_sum = 0.0
        has_shifted = False
        for v in direct_visits:
            outcome = str(v.get("outcome", "")).strip().lower()
            if outcome == "neighbour_says_shifted":
                has_shifted = True
            if outcome in {"address_not_traceable", "neighbour_says_shifted"}:
                iv = float(v.get("visit_integrity_score", 0.5))
                vx = float(v.get("checkin_x", 0.0))
                vy = float(v.get("checkin_y", 0.0))
                dist = compute_euclidean_distance(cx, cy, vx, vy)
                if dist <= r_neg:
                    tau = float(v.get("tortuosity", 1.0))
                    pen_sum += iv * min(1.0, tau / 3.0) * math.exp(-(dist ** 2) / (2.0 * (r_neg ** 2)))
        
        # Shifted outcome penalizes overall candidate certainty by at least 0.40
        if has_shifted:
            pen_sum += 0.40
        negative_penalty = float(min(1.0, pen_sum))

    return {
        "TextMatch": round(text_match, 4),
        "DirectEvidence": round(direct_evidence, 4),
        "NeighborSupport": round(neighbor_support, 4),
        "BaselineAgreement": round(baseline_agreement, 4),
        "NegativeSearchPenalty": round(negative_penalty, 4),
    }


def score_candidate(
    candidate: SpatialCandidate,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[ModelConfig] = None,
) -> float:
    """Evaluate candidate using deterministic linear utility function.
    
    Score(C_k) = w_text * TextMatch
               + w_visit * DirectEvidence
               + w_neigh * NeighborSupport
               + w_base * BaselineAgreement
               - w_neg * NegativeSearchPenalty
    """
    if not candidate.available:
        return -999.0

    cfg = config or DEFAULT_CONFIG
    w = weights or {
        "w_text": cfg.w_text,
        "w_visit": cfg.w_visit,
        "w_neigh": cfg.w_neigh,
        "w_base": cfg.w_base,
        "w_neg": cfg.w_neg,
    }

    f = candidate.features
    s = (
        w.get("w_text", 1.0) * f.get("TextMatch", 0.0)
        + w.get("w_visit", 2.5) * f.get("DirectEvidence", 0.0)
        + w.get("w_neigh", 1.5) * f.get("NeighborSupport", 0.0)
        + w.get("w_base", 0.5) * f.get("BaselineAgreement", 0.0)
        - w.get("w_neg", 2.0) * f.get("NegativeSearchPenalty", 0.0)
    )
    return float(round(s, 4))


def rank_candidates(
    candidates: List[SpatialCandidate],
    weights: Optional[Dict[str, float]] = None,
    config: Optional[ModelConfig] = None,
    deduplicate: bool = True,
    tolerance_m: float = 5.0,
) -> List[SpatialCandidate]:
    """Score and rank candidates deterministically in descending order of utility.
    
    Tie-breaking uses candidate priority: C2 > C4 > C3 > C1 > C5.
    """
    cfg = config or DEFAULT_CONFIG
    cands_to_eval = deduplicate_candidates(candidates, tolerance_m=tolerance_m) if deduplicate else [c for c in candidates if c.available]

    if not cands_to_eval:
        return []

    scored: List[SpatialCandidate] = []
    for cand in cands_to_eval:
        score = score_candidate(cand, weights=weights, config=cfg)
        scored.append(replace(cand, total_score=score))

    # Sort descending by score; tie-break by priority (C2 > C4 > C3 > C1 > C5)
    sorted_cands = sorted(
        scored,
        key=lambda c: (-c.total_score, CANDIDATE_PRIORITY.get(c.candidate_type, 99)),
    )

    ranked: List[SpatialCandidate] = []
    for i, c in enumerate(sorted_cands, 1):
        ranked.append(replace(c, rank=i))

    return ranked


def explain_ranking(ranked_candidates: List[SpatialCandidate]) -> Dict[str, Any]:
    """Generate auditable dictionary explaining the reranker's selection."""
    if not ranked_candidates:
        return {"selected": None, "reason": "No candidates available"}

    selected = ranked_candidates[0]
    runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None

    margin = selected.total_score - (runner_up.total_score if runner_up else 0.0)

    return {
        "selected_candidate_type": selected.candidate_type,
        "selected_score": selected.total_score,
        "runner_up_type": runner_up.candidate_type if runner_up else None,
        "runner_up_score": runner_up.total_score if runner_up else 0.0,
        "score_margin": round(margin, 4),
        "agreed_sources": list(selected.agreed_sources),
        "selected_features": dict(selected.features),
        "feature_contributions": {
            k: v for k, v in selected.features.items()
        },
        "all_candidate_ranks": [
            {
                "rank": c.rank,
                "candidate_type": c.candidate_type,
                "total_score": c.total_score,
                "x": c.x,
                "y": c.y,
                "agreed_sources": list(c.agreed_sources),
            }
            for c in ranked_candidates
        ],
    }


def fit_reranker_weights(
    train_records: List[Dict[str, Any]],
    initial_weights: Optional[Dict[str, float]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit interpretable non-negative linear reranker weights on permitted training data.
    
    Zero-leakage guarantees:
    - Never accesses validation or test split benchmark addresses.
    - Uses deterministic coordinate ranking / bounded optimization.
    - Guaranteed reproducible with fixed random_state.
    
    Returns dictionary containing fitted weights and fit metadata.
    """
    cfg = DEFAULT_CONFIG
    w0_dict = initial_weights or {
        "w_text": cfg.w_text,
        "w_visit": cfg.w_visit,
        "w_neigh": cfg.w_neigh,
        "w_base": cfg.w_base,
        "w_neg": cfg.w_neg,
    }

    # If training data is small or empty, return frozen default weights with metadata
    if not train_records or len(train_records) < 5:
        return {
            "fit_scope": "benchmark_train",
            "weights": w0_dict,
            "sample_count": len(train_records),
            "convergence": "default_frozen_spec",
            "method": "frozen_spec_initial_weights",
            "random_state": random_state,
        }

    # Construct feature matrix X and target rank utility y
    # For each address episode with >= 2 candidates, build pairwise ranking preferences
    # We optimize non-negative weights to maximize margin between superior and inferior candidates
    pairs_diff: List[np.ndarray] = []
    
    for rec in train_records:
        candidates = rec.get("candidates", [])
        if len(candidates) < 2:
            continue
        # Ground truth error for each candidate
        # Superior candidate has lower distance to surveyed pin
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c_i, c_j = candidates[i], candidates[j]
                err_i = rec.get("errors", {}).get(c_i.candidate_type, 9999.0)
                err_j = rec.get("errors", {}).get(c_j.candidate_type, 9999.0)
                
                f_i = np.array([
                    c_i.features.get("TextMatch", 0.0),
                    c_i.features.get("DirectEvidence", 0.0),
                    c_i.features.get("NeighborSupport", 0.0),
                    c_i.features.get("BaselineAgreement", 0.0),
                    -c_i.features.get("NegativeSearchPenalty", 0.0),
                ])
                f_j = np.array([
                    c_j.features.get("TextMatch", 0.0),
                    c_j.features.get("DirectEvidence", 0.0),
                    c_j.features.get("NeighborSupport", 0.0),
                    c_j.features.get("BaselineAgreement", 0.0),
                    -c_j.features.get("NegativeSearchPenalty", 0.0),
                ])
                
                if err_i < err_j - 5.0:  # i is clearly superior
                    pairs_diff.append(f_i - f_j)
                elif err_j < err_i - 5.0:  # j is clearly superior
                    pairs_diff.append(f_j - f_i)

    if not pairs_diff:
        return {
            "fit_scope": "benchmark_train",
            "weights": w0_dict,
            "sample_count": len(train_records),
            "convergence": "default_frozen_spec",
            "method": "frozen_spec_initial_weights",
            "random_state": random_state,
        }

    diff_matrix = np.array(pairs_diff)
    w_init = np.array([
        w0_dict["w_text"],
        w0_dict["w_visit"],
        w0_dict["w_neigh"],
        w0_dict["w_base"],
        w0_dict["w_neg"],
    ])

    # Deterministic soft-margin loss: sum log(1 + exp(-w^T diff)) + lambda * ||w - w0||^2
    def loss_fn(w: np.ndarray) -> float:
        margins = diff_matrix @ w
        # numerical stability
        clipped = np.clip(-margins, -30.0, 30.0)
        loss = np.sum(np.log1p(np.exp(clipped)))
        reg = 0.05 * np.sum((w - w_init) ** 2)
        return float(loss + reg)

    from scipy.optimize import minimize
    bounds = [(0.1, 5.0), (0.5, 5.0), (0.2, 5.0), (0.1, 3.0), (0.1, 4.0)]
    res = minimize(loss_fn, w_init, method="L-BFGS-B", bounds=bounds)

    fitted_w = res.x if res.success else w_init
    fitted_dict = {
        "w_text": round(float(fitted_w[0]), 4),
        "w_visit": round(float(fitted_w[1]), 4),
        "w_neigh": round(float(fitted_w[2]), 4),
        "w_base": round(float(fitted_w[3]), 4),
        "w_neg": round(float(fitted_w[4]), 4),
    }

    return {
        "fit_scope": "benchmark_train",
        "weights": fitted_dict,
        "sample_count": len(train_records),
        "pairwise_comparisons": len(pairs_diff),
        "loss": round(float(res.fun), 4),
        "convergence": bool(res.success),
        "method": "deterministic_pairwise_utility_ranking",
        "random_state": random_state,
    }
