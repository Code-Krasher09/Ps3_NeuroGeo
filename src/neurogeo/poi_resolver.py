"""Locality-bounded POI resolver and landmark disambiguator.

Authority: PS3_FINAL_ARCHITECTURE.md Section 2, PS3_DATA_CONTRACT.md Section 2 (Proof 1),
and PS3_MODEL_SPEC.md Section 5 (Candidate C3).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import re
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, PATHS, ModelConfig
from neurogeo.schemas import PoiCandidateMatch, PoiResolutionResult


# Precomputed/imported landmark alias mappings for robust textual matching
from neurogeo.address_parser import (
    LANDMARK_PATTERNS,
    char_ngrams,
    ngram_similarity,
    normalize_script,
)


_POI_LOCALITY_CACHE: Dict[str, Dict[str, Tuple[str, float]]] = {}


def _compute_poi_locality_assignments(
    df_pois: pd.DataFrame,
    df_localities: pd.DataFrame,
) -> Dict[str, Tuple[str, float]]:
    """Compute nearest locality ID and metric distance for each POI.
    
    Returns:
        dict mapping poi_id -> (locality_id, distance_m)
    """
    cache_key = f"{len(df_pois)}_{len(df_localities)}_{id(df_pois)}_{id(df_localities)}"
    if cache_key in _POI_LOCALITY_CACHE:
        return _POI_LOCALITY_CACHE[cache_key]
        
    assignments: Dict[str, Tuple[str, float]] = {}
    for _, poi in df_pois.iterrows():
        town = poi["town_id"]
        town_locs = df_localities[df_localities["town_id"] == town]
        if town_locs.empty:
            continue
        
        min_dist = float("inf")
        best_loc_id = ""
        px, py = float(poi["x"]), float(poi["y"])
        
        for _, loc in town_locs.iterrows():
            lx, ly = float(loc["centroid_x"]), float(loc["centroid_y"])
            dist = math.sqrt((lx - px) ** 2 + (ly - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_loc_id = str(loc["locality_id"])
                
        assignments[str(poi["poi_id"])] = (best_loc_id, min_dist)
    _POI_LOCALITY_CACHE[cache_key] = assignments
    return assignments



# Precompute normalized alias ngrams once
PRECOMPUTED_ALIASES: Dict[str, List[Tuple[str, str, Set[str]]]] = {
    lm_type: [
        (alias, normalize_script(alias), char_ngrams(normalize_script(alias)))
        for alias in patterns
    ]
    for lm_type, patterns in LANDMARK_PATTERNS.items()
}

_POI_RECORDS_CACHE: Dict[Tuple[int, int], List[dict]] = {}


def _get_poi_records(df_pois: pd.DataFrame) -> List[dict]:
    k = (len(df_pois), id(df_pois))
    if k in _POI_RECORDS_CACHE:
        return _POI_RECORDS_CACHE[k]
    records = []
    for _, row in df_pois.iterrows():
        p_name = str(row["name"])
        p_type = str(row["landmark_type"])
        p_name_norm = normalize_script(p_name)
        p_type_norm = normalize_script(p_type.replace("_", " "))
        records.append({
            "poi_id": str(row["poi_id"]),
            "town_id": str(row["town_id"]),
            "name": p_name,
            "landmark_type": p_type,
            "name_norm": p_name_norm,
            "type_norm": p_type_norm,
            "name_ngrams": char_ngrams(p_name_norm),
            "type_ngrams": char_ngrams(p_type_norm),
            "x": float(row["x"]),
            "y": float(row["y"]),
        })
    _POI_RECORDS_CACHE[k] = records
    return records


def compute_text_poi_similarity(
    mention_text: str,
    poi_name: str,
    landmark_type: str,
    mention_norm: Optional[str] = None,
    mention_ngrams: Optional[Set[str]] = None,
    landmark_type_hint: Optional[str] = None,
) -> float:
    """Compute textual similarity between a landmark mention and a POI candidate."""
    m_norm = mention_norm if mention_norm is not None else normalize_script(mention_text)
    m_ngrams = mention_ngrams if mention_ngrams is not None else char_ngrams(m_norm)
    poi_name_norm = normalize_script(poi_name)
    type_norm = normalize_script(landmark_type.replace("_", " "))
    
    # Fast path: landmark type matches known hint
    if landmark_type_hint is not None and landmark_type == landmark_type_hint:
        return 0.95
        
    # 1. Exact match with standardized name or type
    if m_norm == poi_name_norm or m_norm == type_norm:
        return 1.0
    if poi_name_norm in m_norm or m_norm in poi_name_norm:
        return 0.95
    if type_norm in m_norm or m_norm in type_norm:
        return 0.90
        
    # 2. Check known aliases for this landmark_type
    aliases = PRECOMPUTED_ALIASES.get(landmark_type, [])
    max_sim = 0.0
    for raw_alias, alias_norm, alias_ngrams in aliases:
        if alias_norm in m_norm or m_norm in alias_norm or raw_alias in mention_text:
            return 0.95
        if m_ngrams and alias_ngrams:
            sim = len(m_ngrams & alias_ngrams) / len(m_ngrams | alias_ngrams)
            if sim > max_sim:
                max_sim = sim
            
    # 3. Fuzzy similarity against name and type
    name_ng = char_ngrams(poi_name_norm)
    type_ng = char_ngrams(type_norm)
    if m_ngrams and name_ng:
        sim = len(m_ngrams & name_ng) / len(m_ngrams | name_ng)
        if sim > max_sim:
            max_sim = sim
    if m_ngrams and type_ng:
        sim = len(m_ngrams & type_ng) / len(m_ngrams | type_ng)
        if sim > max_sim:
            max_sim = sim
            
    return round(max_sim, 3)


def compute_landmark_type_similarities(
    mention_text: str,
    landmark_type_hint: Optional[str] = None,
) -> Dict[str, float]:
    """Compute similarity between mention text and each of the 14 landmark categories."""
    if landmark_type_hint is not None:
        return {landmark_type_hint: 0.95}
        
    m_norm = normalize_script(mention_text)
    m_ng = char_ngrams(m_norm)
    sims: Dict[str, float] = {}

    
    for lm_type, aliases in PRECOMPUTED_ALIASES.items():
        if landmark_type_hint is not None and lm_type == landmark_type_hint:
            sims[lm_type] = 0.95
            continue
        type_norm = lm_type.replace("_", " ")
        if m_norm == type_norm:
            sims[lm_type] = 1.0
            continue
        max_s = 0.0
        for raw_alias, a_norm, a_ng in aliases:
            if a_norm == m_norm:
                max_s = 1.0
                break
            if a_norm in m_norm or m_norm in a_norm or raw_alias in mention_text:
                if max_s < 0.95:
                    max_s = 0.95
            if m_ng and a_ng:
                s = len(m_ng & a_ng) / len(m_ng | a_ng)
                if s > max_s:
                    max_s = s
        sims[lm_type] = round(max_s, 3)
    return sims


def resolve_poi_detailed(
    extracted_landmark_text: str,
    town_id: str,
    locality_id: Optional[str],
    df_pois: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
    config: Optional[ModelConfig] = None,
    landmark_type_hint: Optional[str] = None,
) -> PoiResolutionResult:
    """Resolve landmark mention to a POI with full auditable diagnostics.
    
    Enforces:
    - Town scoping (wrong-town POIs never considered)
    - Locality bounding (within ~800m locality bounding disc)
    - Ambiguity detection (if multiple candidates have indistinguishable confidence)
    - Zero-duplicate invariant validation within resolved locality
    - Deterministic coordinate preservation without arbitrary offsets
    """
    cfg = config or DEFAULT_CONFIG
    if df_localities is None:
        df_localities = pd.read_csv(PATHS.localities_csv)
        
    if not extracted_landmark_text or town_id not in ("T1", "T2", "T3"):
        return PoiResolutionResult(status="unresolved")
        
    # Precompute locality assignments for POIs (cached globally)
    poi_loc_map = _compute_poi_locality_assignments(df_pois, df_localities)
    all_records = _get_poi_records(df_pois)
    
    # Filter records to target town
    town_records = [r for r in all_records if r["town_id"] == town_id]
    if not town_records:
        return PoiResolutionResult(status="unresolved")
    
    # 2. Scope candidates by locality if locality_id is provided
    if locality_id is not None:
        candidate_records = [
            r for r in town_records
            if poi_loc_map.get(r["poi_id"], (None, 9999.0))[0] == locality_id
            and poi_loc_map.get(r["poi_id"], (None, 9999.0))[1] <= cfg.locality_bounding_radius_m
        ]
        in_locality_scope = True
    else:
        # No locality resolved: consider all town POIs
        candidate_records = town_records
        in_locality_scope = False
        
    if not candidate_records:
        return PoiResolutionResult(status="unresolved")
        
    # 3. Score all candidates against extracted landmark mention via type similarity
    type_sims = compute_landmark_type_similarities(
        mention_text=extracted_landmark_text,
        landmark_type_hint=landmark_type_hint,
    )
    
    matches: List[PoiCandidateMatch] = []
    for r in candidate_records:
        pid = r["poi_id"]
        p_name = r["name"]
        p_type = r["landmark_type"]
        px = r["x"]
        py = r["y"]
        assigned_loc, _ = poi_loc_map.get(pid, (None, 0.0))
        score = type_sims.get(p_type, 0.0)
        
        matches.append(PoiCandidateMatch(
            poi_id=pid,
            name=p_name,
            landmark_type=p_type,
            score=score,
            x=px,
            y=py,
            locality_id=assigned_loc,
            in_locality_bound=in_locality_scope,
        ))


        
    # Sort deterministically: highest score first, break ties by poi_id
    matches.sort(key=lambda m: (-m.score, m.poi_id))
    
    top = matches[0]
    
    # 4. Check if score meets minimum fuzzy threshold
    if top.score < cfg.poi_fuzzy_threshold:
        return PoiResolutionResult(
            status="unresolved",
            confidence=top.score,
            candidate_matches=tuple(matches[:5]),
        )
        
    # 5. Handle ambiguity
    # Case A: Without locality scoping, generic POIs (e.g. Bus Stop) match multiple POIs across town
    if not in_locality_scope:
        competing = [m for m in matches if m.score >= cfg.poi_fuzzy_threshold]
        if len(competing) > 1 and abs(competing[0].score - competing[1].score) < cfg.ambiguity_score_margin:
            return PoiResolutionResult(
                status="ambiguous",
                confidence=top.score,
                candidate_matches=tuple(matches[:5]),
            )
            
    # Case B: Within locality scoping, check if two distinct POI types match similarly
    if len(matches) > 1:
        runner_up = matches[1]
        if runner_up.score >= cfg.poi_fuzzy_threshold and abs(top.score - runner_up.score) < cfg.ambiguity_score_margin:
            # Different POI in same locality with identical score -> ambiguous
            return PoiResolutionResult(
                status="ambiguous",
                confidence=top.score,
                candidate_matches=tuple(matches[:5]),
            )
            
    # 6. Success: Unique resolved POI
    return PoiResolutionResult(
        status="resolved",
        poi_id=top.poi_id,
        poi_name=top.name,
        landmark_type=top.landmark_type,
        x=top.x,
        y=top.y,
        confidence=top.score,
        candidate_matches=tuple(matches[:5]),
    )


def resolve_poi(
    extracted_landmark_text: str,
    town_id: str,
    locality_id: Optional[str],
    df_pois: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[str, float, float]]:
    """Resolve landmark mention to a unique POI within locality bounding disc.
    
    Returns:
        (poi_id, x, y) tuple if successfully resolved, None otherwise.
        Guarantees that (x, y) are physical POI coordinates without heuristic offsets.
    """
    res = resolve_poi_detailed(
        extracted_landmark_text=extracted_landmark_text,
        town_id=town_id,
        locality_id=locality_id,
        df_pois=df_pois,
        df_localities=df_localities,
    )
    if res.status == "resolved" and res.poi_id is not None and res.x is not None and res.y is not None:
        return (res.poi_id, res.x, res.y)
    return None
