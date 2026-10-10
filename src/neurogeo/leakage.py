"""Zero-leakage isolation boundary and provenance verification contracts.

Authority: NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd


# Valid fit_scope provenance patterns defined by the evaluation contract
VALID_FIT_SCOPE_PATTERNS = {
    "visit_features": r"^unsupervised_all$",
    "street_centroids": r"^train_fold_\d+$|^train_all$|^production_all$",
    "neighbor_graph": r"^train_fold_\d+$|^train_all$|^production_all$",
    "calibration_quantiles": r"^calibration_oof(_\d+)?$|^scv_leave_one_out$|^train_all$",
    "reranker_weights": r"^train_fold_\d+$|^train_all$|^production_all$|^benchmark_train$",
    "predictions": r"^eval_val$|^eval_test$|^train_fold_\d+$|^scv_leave_one_out$|^benchmark_train$|^benchmark_val$|^benchmark_test$|^production_all$",
    "candidates": r"^train_fold_\d+$|^train_all$|^production_all$|^benchmark_train$|^benchmark_val$|^benchmark_test$",
}


@dataclass(frozen=True)
class IsolationClosure:
    """Formal definition of Closure(A*): everything that must be purged when predicting A*.
    
    In accordance with NEUROGEO_EVALUATION_AND_LEAKAGE_CONTRACT.md Section 2:
    D_visible(A*) = D_full \\ Closure(A*)
    """
    target_address_id: str
    target_account_id: Optional[str] = None
    purged_visit_ids: Set[str] = field(default_factory=set)

    def filter_addresses(self, df_addresses: pd.DataFrame) -> pd.DataFrame:
        """Purge target address row from addresses table (for neighbor graph construction)."""
        if "address_id" not in df_addresses.columns:
            return df_addresses
        return df_addresses[df_addresses["address_id"] != self.target_address_id].copy()

    def filter_visits(self, df_visits: pd.DataFrame) -> pd.DataFrame:
        """Purge all visits associated with the target address."""
        if "address_id" not in df_visits.columns:
            return df_visits
        return df_visits[df_visits["address_id"] != self.target_address_id].copy()

    def filter_gps_points(self, df_gps: pd.DataFrame) -> pd.DataFrame:
        """Purge all trajectory points belonging to the target address's visits."""
        if "visit_id" not in df_gps.columns:
            return df_gps
        return df_gps[~df_gps["visit_id"].isin(self.purged_visit_ids)].copy()

    def filter_medoids(self, df_medoids: pd.DataFrame) -> pd.DataFrame:
        """Purge target address row from doorstep medoids table."""
        if "address_id" not in df_medoids.columns:
            return df_medoids
        return df_medoids[df_medoids["address_id"] != self.target_address_id].copy()


def compute_isolation_closure(
    address_id: str,
    df_visits: pd.DataFrame,
    df_addresses: Optional[pd.DataFrame] = None,
) -> IsolationClosure:
    """Construct Closure(A*) for a given target address ID."""
    purged_visits: Set[str] = set()
    if "address_id" in df_visits.columns and "visit_id" in df_visits.columns:
        matching_visits = df_visits[df_visits["address_id"] == address_id]["visit_id"]
        purged_visits = set(matching_visits.dropna().unique())

    target_account_id = None
    if df_addresses is not None and "address_id" in df_addresses.columns and "account_id" in df_addresses.columns:
        matching_accounts = df_addresses[df_addresses["address_id"] == address_id]["account_id"]
        if len(matching_accounts) > 0:
            target_account_id = str(matching_accounts.iloc[0])

    return IsolationClosure(
        target_address_id=address_id,
        target_account_id=target_account_id,
        purged_visit_ids=purged_visits,
    )


def validate_fit_scope(artifact_type: str, fit_scope: str) -> bool:
    """Validate declared fit_scope string against the evaluation contract specification."""
    pattern = VALID_FIT_SCOPE_PATTERNS.get(artifact_type)
    if not pattern:
        # Fallback for unrecognized artifact types
        return False
    return bool(re.match(pattern, fit_scope))


def assert_fit_scope_valid(artifact_type: str, fit_scope: str) -> None:
    """Raise ValueError if the fit_scope does not comply with the evaluation contract."""
    if not validate_fit_scope(artifact_type, fit_scope):
        expected_pattern = VALID_FIT_SCOPE_PATTERNS.get(artifact_type, "N/A")
        raise ValueError(
            f"Invalid fit_scope '{fit_scope}' for artifact type '{artifact_type}'. "
            f"Expected pattern matching: {expected_pattern}"
        )


def check_artifact_for_leakage(
    target_address_id: str,
    artifact_data: Any,
    target_visit_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """Inspect an artifact to verify zero contamination from target address A*.
    
    Answers the core question:
    'For target address A*, has any information derived from A* entered this artifact?'
    
    Returns (has_leakage, diagnostic_message).
    """
    # 1. Inspect Pandas DataFrame
    if isinstance(artifact_data, pd.DataFrame):
        if "address_id" in artifact_data.columns:
            if (artifact_data["address_id"] == target_address_id).any():
                return True, f"Found target address_id '{target_address_id}' in DataFrame 'address_id' column."
        if "neighbor_address_id" in artifact_data.columns:
            if (artifact_data["neighbor_address_id"] == target_address_id).any():
                return True, f"Found target address_id '{target_address_id}' in DataFrame 'neighbor_address_id' column."
        if target_visit_ids and "visit_id" in artifact_data.columns:
            found_visits = set(artifact_data["visit_id"]).intersection(target_visit_ids)
            if found_visits:
                return True, f"Found {len(found_visits)} target visit_ids in DataFrame: {found_visits}"

    # 2. Inspect Dictionary or Mapping
    elif isinstance(artifact_data, dict):
        if target_address_id in artifact_data:
            return True, f"Target address_id '{target_address_id}' is a direct key in dictionary."
        # Check dictionary values if they are collections
        for k, v in artifact_data.items():
            if isinstance(v, (set, list, tuple)) and target_address_id in v:
                return True, f"Target address_id '{target_address_id}' found in dictionary values for key '{k}'."

    # 3. Inspect Set or List
    elif isinstance(artifact_data, (set, list, tuple)):
        if target_address_id in artifact_data:
            return True, f"Target address_id '{target_address_id}' found directly in sequence/set."

    return False, "Clean: No target address contamination detected."


def assert_no_leakage(
    target_address_id: str,
    artifact_data: Any,
    target_visit_ids: Optional[Set[str]] = None,
) -> None:
    """Raise AssertionError if target address information has entered the artifact."""
    has_leakage, message = check_artifact_for_leakage(target_address_id, artifact_data, target_visit_ids)
    if has_leakage:
        raise AssertionError(f"LEAKAGE CONTRACT VIOLATION: {message}")
