
"""Official evaluation metrics, benchmark partitioning, and commercial baseline reproduction.

Authority: PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md and PS3_DATA_CONTRACT.md.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from neurogeo.config import PATHS, PathConfig


@dataclass(frozen=True)
class ErrorMetrics:
    """Summary error metrics in meters."""
    count: int
    median_error_m: float
    p90_error_m: float
    max_error_m: float
    min_error_m: float
    mean_error_m: float
    p25_error_m: float
    p75_error_m: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


def compute_euclidean_error(
    pred_x: Union[np.ndarray, pd.Series],
    pred_y: Union[np.ndarray, pd.Series],
    true_x: Union[np.ndarray, pd.Series],
    true_y: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Compute 2D Euclidean distance error in meters.
    
    Formula: sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)
    """
    px = np.asarray(pred_x, dtype=np.float64)
    py = np.asarray(pred_y, dtype=np.float64)
    tx = np.asarray(true_x, dtype=np.float64)
    ty = np.asarray(true_y, dtype=np.float64)
    return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)


def compute_error_metrics(errors_m: Union[np.ndarray, List[float]]) -> ErrorMetrics:
    """Compute standard competition distance error metrics."""
    arr = np.asarray(errors_m, dtype=np.float64)
    if len(arr) == 0:
        return ErrorMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    return ErrorMetrics(
        count=int(len(arr)),
        median_error_m=float(np.median(arr)),
        p90_error_m=float(np.percentile(arr, 90)),
        max_error_m=float(np.max(arr)),
        min_error_m=float(np.min(arr)),
        mean_error_m=float(np.mean(arr)),
        p25_error_m=float(np.percentile(arr, 25)),
        p75_error_m=float(np.percentile(arr, 75)),
    )


def compute_calibration_coverage(
    errors_m: Union[np.ndarray, List[float]],
    radii_m: Union[np.ndarray, List[float]],
) -> float:
    """Compute empirical coverage percentage of true positions falling within predicted radii.
    
    Coverage = (1/N) * sum(I(Error_i <= R_i))
    """
    err = np.asarray(errors_m, dtype=np.float64)
    rad = np.asarray(radii_m, dtype=np.float64)
    if len(err) == 0:
        return 0.0
    return float(np.mean(err <= rad))


def load_benchmark_ground_truth(paths: Optional[PathConfig] = None) -> pd.DataFrame:
    """Load the 100 surveyed benchmark addresses joined with baseline geocodes and split assignments.
    
    Returns a DataFrame containing:
    - address_id, account_id, town_id, address_text
    - surveyed_x, surveyed_y
    - geocoder_x, geocoder_y, precision
    - split (train: 66, val: 19, test: 15)
    - visit_count (number of field visits logged)
    - is_visited (bool)
    """
    p = paths or PATHS

    df_surveyed = pd.read_csv(p.surveyed_addresses_csv)
    df_addresses = pd.read_csv(p.addresses_csv)
    df_baseline = pd.read_csv(p.baseline_geocodes_csv)
    df_splits = pd.read_csv(p.splits_csv)
    df_visits = pd.read_csv(p.field_visits_csv)

    # 1. Join surveyed with addresses to get account_id and town_id
    merged = df_surveyed.merge(df_addresses, on="address_id", how="inner")
    if len(merged) != len(df_surveyed):
        raise ValueError(f"Foreign key mismatch: {len(df_surveyed)} surveyed vs {len(merged)} joined with addresses.")

    # 2. Join with baseline geocodes
    merged = merged.merge(df_baseline[["address_id", "geocoder_x", "geocoder_y", "precision"]], on="address_id", how="inner")
    if len(merged) != len(df_surveyed):
        raise ValueError(f"Baseline geocode mismatch: missing geocodes for {len(df_surveyed) - len(merged)} surveyed addresses.")

    # 3. Join with splits on account_id
    merged = merged.merge(df_splits[["account_id", "split"]], on="account_id", how="inner")
    if len(merged) != len(df_surveyed):
        raise ValueError(f"Split assignment mismatch: missing split for {len(df_surveyed) - len(merged)} accounts.")

    # 4. Attach visit count and visited status
    visit_counts = df_visits.groupby("address_id").size().rename("visit_count")
    merged = merged.merge(visit_counts, on="address_id", how="left")
    merged["visit_count"] = merged["visit_count"].fillna(0).astype(int)
    merged["is_visited"] = merged["visit_count"] > 0

    return merged


def evaluate_commercial_baseline(
    paths: Optional[PathConfig] = None,
    write_report: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reproduce official Level 0 commercial baseline metrics on the 100 surveyed addresses.
    
    Computes Euclidean distance between (geocoder_x, geocoder_y) and (surveyed_x, surveyed_y).
    """
    p = paths or PATHS
    df_bench = load_benchmark_ground_truth(p)

    errors = compute_euclidean_error(
        df_bench["geocoder_x"],
        df_bench["geocoder_y"],
        df_bench["surveyed_x"],
        df_bench["surveyed_y"],
    )
    df_bench["baseline_error_m"] = errors

    overall_metrics = compute_error_metrics(errors)

    # Metrics by split
    split_metrics = {}
    for split_name in ["train", "validation", "test"]:
        sub = df_bench[df_bench["split"].replace({"val": "validation"}) == split_name]
        split_metrics[split_name] = compute_error_metrics(sub["baseline_error_m"]).to_dict()
    # Provide 'val' alias for consistency
    split_metrics["val"] = split_metrics["validation"]

    # Metrics by visited status
    visited_metrics = {}
    visited_metrics["visited"] = compute_error_metrics(df_bench[df_bench["is_visited"]]["baseline_error_m"]).to_dict()
    visited_metrics["unvisited_cold_start"] = compute_error_metrics(df_bench[~df_bench["is_visited"]]["baseline_error_m"]).to_dict()

    # Metrics by geocoder precision
    precision_metrics = {}
    for prec, sub in df_bench.groupby("precision"):
        precision_metrics[prec] = compute_error_metrics(sub["baseline_error_m"]).to_dict()

    report: Dict[str, Any] = {
        "benchmark_dataset": "surveyed_addresses.csv",
        "benchmark_count": int(len(df_bench)),
        "metric_description": "2D Euclidean distance error in Cartesian meters between commercial geocoder and ground truth",
        "overall": overall_metrics.to_dict(),
        "by_split": split_metrics,
        "by_visited_status": visited_metrics,
        "by_precision": precision_metrics,
        "data_sources": {
            "surveyed_addresses": "data/ps3_geocoder/surveyed_addresses.csv",
            "baseline_geocodes": "data/ps3_geocoder/baseline_geocodes.csv",
            "addresses": "data/shared/addresses.csv",
            "splits": "data/shared/splits.csv",
            "field_visits": "data/shared/field_visits.csv",
        },
        "reproducibility": {
            "split_counts": {
                "train": int((df_bench["split"] == "train").sum()),
                "validation": int(df_bench["split"].isin(["validation", "val"]).sum()),
                "test": int((df_bench["split"] == "test").sum()),
            },
            "tolerance_check": {
                "expected_median_m": 376.4,
                "tolerance_m": 0.5,
                "actual_median_m": round(overall_metrics.median_error_m, 2),
                "reproduction_verified": bool(abs(overall_metrics.median_error_m - 376.4) <= 0.5),
            },
        },
    }

    if write_report:
        out_file = output_path or (p.metrics_dir / "baseline_metrics.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
