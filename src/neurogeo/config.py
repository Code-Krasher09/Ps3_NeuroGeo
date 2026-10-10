"""Configuration system for NeuroGeo-Field.

Provides repository-relative path resolution and configurable hyperparameter specifications
in accordance with NEUROGEO_FINAL_ARCHITECTURE.md and NEUROGEO_MODEL_SPEC.md.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find repository root by looking for NEUROGEO_FINAL_ARCHITECTURE.md or .git."""
    curr = (start_path or Path(__file__)).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "NEUROGEO_FINAL_ARCHITECTURE.md").exists() or (parent / "data").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Repository-relative paths for data and report artifacts."""
    repo_root: Path = field(default_factory=find_repo_root)
    
    # Base Data Paths
    @property
    def data_dir(self) -> Path:
        custom_data_root = os.environ.get("NEUROGEO_DATA_ROOT")
        return Path(custom_data_root) if custom_data_root else self.repo_root / "data"

    @property
    def neurogeo_data_dir(self) -> Path:
        return self.data_dir / "ps3_geocoder"

    @property
    def shared_data_dir(self) -> Path:
        return self.data_dir / "shared"

    # Core PS3 Input Tables
    @property
    def baseline_geocodes_csv(self) -> Path:
        return self.neurogeo_data_dir / "baseline_geocodes.csv"

    @property
    def landmarks_poi_csv(self) -> Path:
        return self.neurogeo_data_dir / "landmarks_poi.csv"

    @property
    def localities_csv(self) -> Path:
        return self.neurogeo_data_dir / "localities.csv"

    @property
    def surveyed_addresses_csv(self) -> Path:
        return self.neurogeo_data_dir / "surveyed_addresses.csv"

    @property
    def towns_csv(self) -> Path:
        return self.neurogeo_data_dir / "towns.csv"

    @property
    def visit_gps_points_csv(self) -> Path:
        return self.neurogeo_data_dir / "visit_gps_points.csv"

    # Core Shared Input Tables
    @property
    def addresses_csv(self) -> Path:
        return self.shared_data_dir / "addresses.csv"

    @property
    def field_visits_csv(self) -> Path:
        return self.shared_data_dir / "field_visits.csv"

    @property
    def splits_csv(self) -> Path:
        return self.shared_data_dir / "splits.csv"

    # External Reference Tables (Shared)
    @property
    def accounts_csv(self) -> Path:
        return self.shared_data_dir / "accounts.csv"

    @property
    def agents_csv(self) -> Path:
        return self.shared_data_dir / "agents.csv"

    # Output & Report Directories
    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def ablations_dir(self) -> Path:
        return self.reports_dir / "ablations"

    # Phase 2 Artifact Paths
    @property
    def parsed_addresses_parquet(self) -> Path:
        return self.metrics_dir / "parsed_addresses.parquet"

    @property
    def address_entities_parquet(self) -> Path:
        return self.metrics_dir / "address_entities.parquet"

    @property
    def poi_resolution_summary_json(self) -> Path:
        return self.metrics_dir / "poi_resolution_summary.json"

    @property
    def phase2_parser_summary_json(self) -> Path:
        return self.metrics_dir / "phase2_parser_summary.json"

    # Phase 3 Artifact Paths
    @property
    def street_centroids_parquet(self) -> Path:
        return self.metrics_dir / "street_centroids.parquet"

    @property
    def neighbor_graph_parquet(self) -> Path:
        return self.metrics_dir / "neighbor_graph.parquet"

    @property
    def scv_results_csv(self) -> Path:
        return self.metrics_dir / "scv_results.csv"

    @property
    def loso_results_csv(self) -> Path:
        return self.metrics_dir / "loso_results.csv"

    @property
    def phase3_spatial_graph_summary_json(self) -> Path:
        return self.metrics_dir / "phase3_spatial_graph_summary.json"

    # Phase 4 Artifact Paths
    @property
    def address_doorstep_medoids_parquet(self) -> Path:
        return self.metrics_dir / "address_doorstep_medoids.parquet"

    @property
    def candidate_scores_parquet(self) -> Path:
        return self.metrics_dir / "candidate_scores.parquet"

    @property
    def phase4_predictions_csv(self) -> Path:
        return self.metrics_dir / "phase4_predictions.csv"

    @property
    def reranker_weights_json(self) -> Path:
        return self.metrics_dir / "reranker_weights.json"

    @property
    def calibration_quantiles_json(self) -> Path:
        return self.metrics_dir / "calibration_quantiles.json"

    @property
    def phase4_ablation_csv(self) -> Path:
        return self.metrics_dir / "phase4_ablation.csv"

    @property
    def phase4_summary_json(self) -> Path:
        return self.metrics_dir / "phase4_summary.json"

    # Phase 5 Artifact Paths
    @property
    def phase5_navigation_examples_csv(self) -> Path:
        return self.metrics_dir / "phase5_navigation_examples.csv"

    @property
    def phase5_evidence_cards_json(self) -> Path:
        return self.metrics_dir / "phase5_evidence_cards.json"

    @property
    def phase5_summary_json(self) -> Path:
        return self.metrics_dir / "phase5_summary.json"


@dataclass
class ModelConfig:
    """Hyperparameter configuration separating frozen architecture from tunable parameters.
    
    Formulas and initial values defined in NEUROGEO_MODEL_SPEC.md.
    """
    # Integrity parameters: I(v)
    lambda_photo: float = 1.0
    dwell_k: float = 0.05
    dwell_t0: float = 45.0  # seconds
    v_max: float = 60.0     # km/h

    # Spatial Evidence Strength parameters: S(v)
    sigma_gps: float = 25.0  # meters

    # Phase 2 Parser & POI Resolver parameters
    poi_fuzzy_threshold: float = 0.65
    locality_fuzzy_threshold: float = 0.65
    ambiguity_score_margin: float = 0.05
    locality_bounding_radius_m: float = 800.0

    # Candidate Scorer linear utility weights (frozen defaults)
    w_text: float = 1.0
    w_visit: float = 2.5
    w_neigh: float = 1.5
    w_base: float = 0.5
    w_neg: float = 2.0

    # Candidate Deduplication and Feature radii
    candidate_dedup_distance_m: float = 5.0
    direct_evidence_sigma_m: float = 25.0
    neighbor_support_sigma_m: float = 50.0
    baseline_agreement_sigma_m: float = 100.0
    negative_search_radius_m: float = 100.0

    # Status Tier Operational Thresholds
    status_threshold_doorstep_r_m: float = 35.0
    status_threshold_street_r_m: float = 85.0
    status_threshold_locality_r_m: float = 250.0
    status_threshold_baseline_r_m: float = 400.0

    # Calibration parameters
    target_coverage: float = 0.90
    min_tier_samples: int = 15
    enable_candidate_wise_calibration: bool = True
    enable_landmark_geometry_calibration: bool = True
    collocation_distance_threshold_m: float = 5.0
    landmark_spatial_consistency_radius_m: float = 1000.0
    town_poi_fallback_radius_m: float = 1500.0
    min_calibration_radius_floor_m: float = 15.0
    max_calibration_radius_cap_m: float = 2500.0
    min_calibration_samples: int = 15

    # Random seed
    random_seed: int = 42


# Default global instances
PATHS = PathConfig()
DEFAULT_CONFIG = ModelConfig()

