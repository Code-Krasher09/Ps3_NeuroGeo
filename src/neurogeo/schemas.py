"""Data contracts, intermediate dataclasses, and output schemas for NeuroGeo-Field.

Authority: PS3_DATA_CONTRACT.md and PS3_FINAL_ARCHITECTURE.md.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PredictionStatus(str, Enum):
    """Architecturally frozen status tiers defined in PS3_FINAL_ARCHITECTURE.md."""
    DOORSTEP_VERIFIED = "DOORSTEP_VERIFIED"
    STREET_RESOLVED = "STREET_RESOLVED"
    LOCALITY_BOUNDED = "LOCALITY_BOUNDED"
    BASELINE_FALLBACK = "BASELINE_FALLBACK"


@dataclass(frozen=True)
class ParsedAddressEntity:
    """Structured entity representation parsed from unstructured address text."""
    address_id: str
    town_id: str
    locality_name_raw: Optional[str] = None
    locality_id_resolved: Optional[str] = None
    street_raw: Optional[str] = None
    street_normalized: Optional[str] = None
    landmark_raw: Optional[str] = None
    landmark_poi_id: Optional[str] = None
    spatial_relation: Optional[str] = None  # e.g., 'behind', 'opposite', 'near', 'beside'
    is_multilingual: bool = False
    script_detected: str = "latin"          # 'latin', 'kannada', 'devanagari', 'mixed'
    parser_confidence: float = 0.0          # [0.0, 1.0]
    house_raw: Optional[str] = None
    locality_resolution_status: str = "unresolved"  # 'resolved', 'unresolved', 'ambiguous'
    poi_resolution_status: str = "unresolved"       # 'resolved', 'unresolved', 'ambiguous'
    normalized_text: str = ""
    residual_text: str = ""


@dataclass(frozen=True)
class PoiCandidateMatch:
    """Auditable record of a single evaluated POI candidate."""
    poi_id: str
    name: str
    landmark_type: str
    score: float
    x: float
    y: float
    locality_id: Optional[str] = None
    in_locality_bound: bool = True


@dataclass(frozen=True)
class PoiResolutionResult:
    """Result of resolving a landmark text mention to a POI."""
    status: str  # 'resolved', 'ambiguous', 'unresolved'
    poi_id: Optional[str] = None
    poi_name: Optional[str] = None
    landmark_type: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: float = 0.0
    candidate_matches: Tuple[PoiCandidateMatch, ...] = ()



@dataclass(frozen=True)
class SanitizedVisitObservation:
    """Operational visit record enhanced with integrity, evidence strength, and trajectory physics."""
    visit_id: str
    address_id: str
    checkin_x: float
    checkin_y: float
    dwell_s: int
    outcome: str
    gps_accuracy_m: float
    photo_collision_count: int
    visit_integrity_score: float     # I(v) in [0.0, 1.0]
    spatial_evidence_strength: float # S(v) in [0.0, 1.0]
    tortuosity: float
    radius_of_gyration_m: float
    doorstep_medoid_x: Optional[float] = None
    doorstep_medoid_y: Optional[float] = None


@dataclass(frozen=True)
class SpatialCandidate:
    """Candidate coordinate hypothesis generated for evidence reranking.
    
    Authority: PS3_FINAL_ARCHITECTURE.md and PS3_MODEL_SPEC.md.
    """
    candidate_id: str                # e.g., 'C1_baseline', 'C2_visit', 'C3_poi', 'C4_street', 'C5_locality'
    x: Optional[float]
    y: Optional[float]
    source_type: str = ""
    prior_confidence: float = 0.0          # [0.0, 1.0]
    candidate_type: str = ""              # 'C1', 'C2', 'C3', 'C4', 'C5'
    available: bool = True
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    fit_scope: str = "production_all"
    feature_vector: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    rank: int = 0
    agreed_sources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase4Prediction:
    """Auditable final prediction record for Phase 4 evaluation and export."""
    address_id: str
    predicted_x: float
    predicted_y: float
    selected_candidate_type: str
    status_tier: str
    confidence_radius_m: float
    confidence_level: float
    candidate_count: int
    selected_score: float
    runner_up_score: float
    score_margin: float
    evidence_sources: str
    calibration_tier: str
    calibration_sample_count: int
    calibration_fallback_level: str
    fit_scope: str
    evaluation_scope: str


@dataclass(frozen=True)
class CalibrationTierRecord:
    """Record of calibration empirical quantile for an evidence tier."""
    tier: str
    sample_count: int
    quantile: float
    radius_m: float
    source_scope: str
    fallback_level: str
    unique_episode_count: int = 0
    residual_count: int = 0
    evidence_family: Optional[str] = None


@dataclass(frozen=True)
class EvidenceCard:
    """Auditable evidence card detailing feature contributions for decision explainability."""
    selected_candidate: str
    locality: Optional[str] = None
    street: Optional[str] = None
    landmark: Optional[str] = None
    relation: Optional[str] = None
    clean_direct_visits: int = 0
    integrity_score: float = 0.0
    baseline_discrepancy_m: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeocodingResult:
    """Atomic final output payload for a single address prediction."""
    address_id: str
    predicted_x: float
    predicted_y: float
    confidence_radius_m: float
    prediction_status: PredictionStatus
    fit_scope: str
    evidence_card: EvidenceCard
    offline_navigation_guidance: str


@dataclass(frozen=True)
class ScopedNeighborCentroidResult:
    """Auditable result of Candidate C4 scoped neighbor centroid computation."""
    address_id: str
    c4_x: float
    c4_y: float
    neighbor_count: int
    neighbor_address_ids: Tuple[str, ...]
    neighbor_weights: Tuple[float, ...]
    total_neighbor_weight: float
    town_id: str
    locality_id: str
    street_normalized: str
    fit_scope: str


class NavigationMode(str, Enum):
    """Deterministic navigation modes corresponding to prediction status tiers."""
    DOORSTEP = "DOORSTEP"
    STREET = "STREET"
    LOCALITY = "LOCALITY"
    BASELINE = "BASELINE"


@dataclass(frozen=True)
class NavigationInstruction:
    """Deterministic offline navigation instruction payload."""
    current_x: float
    current_y: float
    target_x: float
    target_y: float
    distance_m: float
    bearing_deg: float
    direction: str
    within_confidence_radius: bool
    confidence_radius_m: float
    status_tier: str
    navigation_mode: str
    instruction: str
    caution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCardData:
    """Deterministic auditable evidence card detailing feature contributions and decision rationale."""
    address_id: str
    status_tier: str
    selected_candidate: str
    predicted_coordinates: Tuple[float, float]
    confidence_radius_m: float
    calibration_tier: str
    calibration_sample_count: int
    calibration_fallback_level: str
    score: float
    score_margin: float
    evidence_sources: List[str]
    positive_evidence: List[str]
    negative_evidence: List[str]
    caution: Optional[str]
    provenance: Dict[str, Any]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

