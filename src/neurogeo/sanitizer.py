"""Operational visit sanitizer, integrity scoring, and spatial evidence extraction.

Authority: NEUROGEO_MODEL_SPEC.md Sections 2 and 3, NEUROGEO_DATA_CONTRACT.md, and NEUROGEO_FINAL_ARCHITECTURE.md.
"""

from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, ModelConfig


# Documented outcome weights from NEUROGEO_MODEL_SPEC.md Section 3
OUTCOME_WEIGHTS: Dict[str, float] = {
    "met_family": 1.00,
    "locked_premises": 0.95,
    "cash_collected": 0.90,
    "met_borrower": 0.85,
    "neighbour_says_shifted": 0.40,  # Address-state evidence only; NOT positive doorstep evidence
    "no_such_person": 0.00,
    "address_not_traceable": 0.00,
}

# Positive doorstep outcomes that qualify for doorstep medoid estimation
POSITIVE_DOORSTEP_OUTCOMES: Set[str] = {
    "met_family",
    "locked_premises",
    "cash_collected",
    "met_borrower",
}


def compute_visit_integrity(
    collision_counts: Union[np.ndarray, pd.Series, List[int], int],
    dwell_s: Union[np.ndarray, pd.Series, List[float], float],
    speed_kmh: Union[np.ndarray, pd.Series, List[float], float] = 0.0,
    lambda_photo: float = DEFAULT_CONFIG.lambda_photo,
    k_dwell: float = DEFAULT_CONFIG.dwell_k,
    t_dwell: float = DEFAULT_CONFIG.dwell_t0,
    v_max: float = DEFAULT_CONFIG.v_max,
) -> Union[np.ndarray, float]:
    """Compute visit operational integrity score I(v) in [0.0, 1.0].
    
    Formula:
        I(v) = exp(-lambda_photo * (CollisionCount_v - 1))
             * (1 / (1 + exp(-k_dwell * (dwell_v - t_dwell))))
             * exp(-speed_v^2 / (2 * v_max^2))
    """
    is_scalar = np.isscalar(collision_counts) and np.isscalar(dwell_s) and np.isscalar(speed_kmh)

    c = np.asarray(collision_counts, dtype=np.float64)
    d = np.asarray(dwell_s, dtype=np.float64)
    s = np.asarray(speed_kmh, dtype=np.float64)

    # Clean inputs: clip collision count to minimum 1
    c_clean = np.maximum(c, 1.0)
    # Dwell clamped non-negative
    d_clean = np.maximum(d, 0.0)
    # Speed clamped non-negative
    s_clean = np.maximum(s, 0.0)

    # Component 1: Photo collision decay
    # Unique photos (c=1) have penalty = exp(0) = 1.0
    photo_decay = np.exp(-lambda_photo * (c_clean - 1.0))

    # Component 2: Continuous dwell sigmoid
    # Midpoint at t_dwell (45s), smoothly transitions from 0 to 1
    dwell_diff = -k_dwell * (d_clean - t_dwell)
    # Numerical stability guard against overflow
    dwell_diff = np.clip(dwell_diff, -60.0, 60.0)
    dwell_sigmoid = 1.0 / (1.0 + np.exp(dwell_diff))

    # Component 3: Velocity attenuation kernel
    speed_decay = np.exp(-(s_clean ** 2) / (2.0 * (v_max ** 2)))

    # Composite integrity
    integrity = photo_decay * dwell_sigmoid * speed_decay
    integrity = np.clip(integrity, 0.0, 1.0)

    if is_scalar:
        return float(integrity.item() if hasattr(integrity, "item") else integrity)
    return integrity


def compute_spatial_evidence_strength(
    gps_accuracy_m: Union[np.ndarray, pd.Series, List[float], float],
    outcomes: Union[np.ndarray, pd.Series, List[str], str],
    sigma_gps: float = DEFAULT_CONFIG.sigma_gps,
) -> Union[np.ndarray, float]:
    """Compute doorstep spatial evidence strength S(v) in [0.0, 1.0].
    
    Formula:
        S(v) = w_outcome(v) * exp(-gps_accuracy_v / sigma_gps)
    """
    is_scalar = np.isscalar(gps_accuracy_m) and isinstance(outcomes, str)

    acc = np.asarray(gps_accuracy_m, dtype=np.float64)
    acc_clean = np.maximum(acc, 0.0)

    if is_scalar:
        w = OUTCOME_WEIGHTS.get(str(outcomes), 0.0)
        decay = float(np.exp(-acc_clean / sigma_gps))
        strength = float(np.clip(w * decay, 0.0, 1.0))
        return strength

    # Vectorized lookup for outcome weights
    out_arr = np.asarray(outcomes)
    weights = np.array([OUTCOME_WEIGHTS.get(str(o), 0.0) for o in out_arr], dtype=np.float64)

    decay = np.exp(-acc_clean / sigma_gps)
    strength = weights * decay
    return np.clip(strength, 0.0, 1.0)


def compute_photo_collision_counts(df_visits: pd.DataFrame) -> pd.Series:
    """Compute photo hash collision counts across visits.
    
    Explicitly handles missing/null photo hashes:
    Missing hashes are NOT treated as a single shared collision group.
    Each missing hash receives count = 1 (no collision penalty).
    """
    if "photo_hash" not in df_visits.columns:
        return pd.Series(1, index=df_visits.index)

    # Valid non-null, non-empty hashes
    valid_mask = df_visits["photo_hash"].notna() & (df_visits["photo_hash"].astype(str).str.strip() != "")
    valid_hashes = df_visits.loc[valid_mask, "photo_hash"].astype(str)

    counts_map = valid_hashes.value_counts()

    # Map counts to valid hashes; missing hashes default to 1
    collision_counts = df_visits["photo_hash"].map(counts_map).fillna(1).astype(int)
    # Ensure missing hashes remain 1
    collision_counts.loc[~valid_mask] = 1

    return collision_counts


def sanitize_visits(
    df_visits: pd.DataFrame,
    df_trajectory_metrics: Optional[pd.DataFrame] = None,
    config: Optional[ModelConfig] = None,
) -> pd.DataFrame:
    """Sanitize all field visits and compute operational integrity, evidence strength, and semantic flags.
    
    Returns a copy of df_visits augmented with:
    - photo_collision_count
    - mean_speed_kmh (from trajectory if available, else 0.0)
    - visit_integrity_score (I_v)
    - spatial_evidence_strength (S_v)
    - is_doorstep_confirmed (bool)
    - is_address_state_shifted (bool)
    - is_untraceable_search (bool)
    - is_disconfirmed (bool)
    - is_credible_doorstep (bool: confirmed and I_v >= 0.50)
    """
    cfg = config or DEFAULT_CONFIG
    df = df_visits.copy()

    # 1. Photo collision counts
    df["photo_collision_count"] = compute_photo_collision_counts(df)

    # 2. Attach trajectory speed if provided
    if df_trajectory_metrics is not None and "visit_id" in df_trajectory_metrics.columns:
        speed_col = "mean_speed_kmh" if "mean_speed_kmh" in df_trajectory_metrics.columns else None
        if speed_col:
            speed_map = df_trajectory_metrics.set_index("visit_id")[speed_col]
            df["mean_speed_kmh"] = df["visit_id"].map(speed_map).fillna(0.0)
        else:
            df["mean_speed_kmh"] = 0.0
    else:
        df["mean_speed_kmh"] = 0.0

    # 3. Dwell duration
    dwell_series = df["dwell_s"].fillna(0.0).astype(float) if "dwell_s" in df.columns else pd.Series(0.0, index=df.index)

    # 4. Visit integrity I(v)
    df["visit_integrity_score"] = compute_visit_integrity(
        collision_counts=df["photo_collision_count"],
        dwell_s=dwell_series,
        speed_kmh=df["mean_speed_kmh"],
        lambda_photo=cfg.lambda_photo,
        k_dwell=cfg.dwell_k,
        t_dwell=cfg.dwell_t0,
        v_max=cfg.v_max,
    )

    # 5. Spatial evidence strength S(v)
    accuracy_series = df["gps_accuracy_m"].fillna(50.0).astype(float) if "gps_accuracy_m" in df.columns else pd.Series(50.0, index=df.index)
    outcome_series = df["outcome"].fillna("unknown").astype(str) if "outcome" in df.columns else pd.Series("unknown", index=df.index)

    df["spatial_evidence_strength"] = compute_spatial_evidence_strength(
        gps_accuracy_m=accuracy_series,
        outcomes=outcome_series,
        sigma_gps=cfg.sigma_gps,
    )

    # 6. Semantic evidence classification
    df["is_doorstep_confirmed"] = df["outcome"].isin(POSITIVE_DOORSTEP_OUTCOMES)
    df["is_address_state_shifted"] = df["outcome"] == "neighbour_says_shifted"
    df["is_untraceable_search"] = df["outcome"] == "address_not_traceable"
    df["is_disconfirmed"] = df["outcome"] == "no_such_person"

    # Credible doorstep evidence: positive outcome and acceptable integrity
    df["is_credible_doorstep"] = df["is_doorstep_confirmed"] & (df["visit_integrity_score"] >= 0.50)

    return df
