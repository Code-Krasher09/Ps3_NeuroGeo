"""End-to-end dual-path inference pipeline for NeuroGeo-Field.

Authority: NEUROGEO_FINAL_ARCHITECTURE.md Section 3 and NEUROGEO_DATA_CONTRACT.md Section 3.
"""

from typing import Dict, List, Optional
import pandas as pd
from neurogeo.schemas import GeocodingResult
from neurogeo.config import ModelConfig


class NeuroGeoPipeline:
    """End-to-end dual-path inference engine (Path A: Visited vs Path B: Cold-Start)."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

    def predict_address(self, address_id: str, address_text: str, town_id: str) -> GeocodingResult:
        """Execute dual-path geocoding for a single address."""
        raise NotImplementedError("Skeleton only. Logic to be implemented in Phase 4.")

    def predict_batch(self, df_addresses: pd.DataFrame) -> List[GeocodingResult]:
        """Execute batch geocoding for multiple addresses."""
        raise NotImplementedError("Skeleton only. Logic to be implemented in Phase 4.")
