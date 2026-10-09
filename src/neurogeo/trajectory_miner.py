"""GPS trajectory physics, numerical guards, and doorstep dwell medoid mining.

Authority: PS3_MODEL_SPEC.md Section 4, PS3_DATA_CONTRACT.md, and PS3_FINAL_ARCHITECTURE.md.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def filter_gps_jumps(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    max_speed_mps: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Filter out instantaneous GPS jumps exceeding physical velocity thresholds.
    
    Returns (filtered_x, filtered_y, filtered_timestamps, num_filtered).
    """
    n = len(x_coords)
    if n <= 1:
        return x_coords, y_coords, timestamps, 0

    valid_indices = [0]
    num_filtered = 0

    for i in range(1, n):
        last_valid = valid_indices[-1]
        dx = x_coords[i] - x_coords[last_valid]
        dy = y_coords[i] - y_coords[last_valid]
        dist = np.hypot(dx, dy)

        if timestamps is not None:
            dt = (timestamps[i] - timestamps[last_valid])
            if isinstance(dt, np.timedelta64):
                dt_s = float(dt / np.timedelta64(1, "s"))
            elif hasattr(dt, "total_seconds"):
                dt_s = float(dt.total_seconds())
            else:
                dt_s = float(dt)

            if dt_s > 0:
                implied_speed = dist / dt_s
                if implied_speed > max_speed_mps:
                    num_filtered += 1
                    continue
            elif dist > (max_speed_mps * 30.0):  # Fallback: jump > 1200m in single step
                num_filtered += 1
                continue
        else:
            # If no timestamps provided, filter single step jumps > 1200m (40m/s * 30s)
            if dist > (max_speed_mps * 30.0):
                num_filtered += 1
                continue

        valid_indices.append(i)

    filtered_x = x_coords[valid_indices]
    filtered_y = y_coords[valid_indices]
    filtered_ts = timestamps[valid_indices] if timestamps is not None else None

    return filtered_x, filtered_y, filtered_ts, num_filtered


def compute_path_tortuosity(
    x_coords: Union[np.ndarray, List[float]],
    y_coords: Union[np.ndarray, List[float]],
    accuracy_m: Optional[Union[np.ndarray, List[float]]] = None,
    min_displacement_m: float = 15.0,
    min_points: int = 3,
    max_accuracy_m: float = 50.0,
) -> float:
    """Compute path tortuosity tau_v with stationary jitter, minimum points, and accuracy guards.
    
    Formula:
        tau_v = path_length / (net_displacement + epsilon)
    
    Guards (return neutral tau = 1.0):
    1. Total points < min_points (e.g. < 3)
    2. Net displacement <= min_displacement_m (e.g. <= 15m; avoids division by zero / jitter explosion)
    3. Mean GPS accuracy > max_accuracy_m (e.g. > 50m; unreliable trajectory)
    """
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    # Guard 1: Too few points
    if len(x) < min_points:
        return 1.0

    # Guard 2: High GPS error
    if accuracy_m is not None:
        acc = np.asarray(accuracy_m, dtype=np.float64)
        if len(acc) > 0 and np.mean(acc) > max_accuracy_m:
            return 1.0

    dx = np.diff(x)
    dy = np.diff(y)
    path_length = float(np.sum(np.hypot(dx, dy)))

    net_displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

    # Guard 3: Stationary jitter
    if net_displacement <= min_displacement_m:
        return 1.0

    tortuosity = path_length / (net_displacement + 1e-6)
    # Tortuosity is geometrically >= 1.0
    return float(max(1.0, tortuosity))


def compute_radius_of_gyration(
    x_coords: Union[np.ndarray, List[float]],
    y_coords: Union[np.ndarray, List[float]],
) -> float:
    """Compute spatial radius of gyration R_g in meters.
    
    Formula:
        R_g = sqrt( (1 / N) * sum( (x_i - mean_x)^2 + (y_i - mean_y)^2 ) )
    """
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    if len(x) <= 1:
        return 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    variance = np.mean((x - mean_x) ** 2 + (y - mean_y) ** 2)
    return float(np.sqrt(max(0.0, variance)))


def compute_spatial_medoid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute the spatial medoid: the point in the sample minimizing weighted Euclidean distance."""
    n = len(x_coords)
    if n == 0:
        raise ValueError("Cannot compute medoid of empty coordinate set.")
    if n == 1:
        return float(x_coords[0]), float(y_coords[0])

    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum <= 0:
        w = np.ones(n, dtype=np.float64)

    # Compute pairwise weighted distances
    total_costs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dists = np.hypot(x_coords - x_coords[i], y_coords - y_coords[i])
        total_costs[i] = np.sum(w * dists)

    best_idx = int(np.argmin(total_costs))
    return float(x_coords[best_idx]), float(y_coords[best_idx])


def build_visit_dwell_medoid(
    df_single_gps: pd.DataFrame,
    checkin_ts: Optional[Union[str, pd.Timestamp]] = None,
    checkin_x: Optional[float] = None,
    checkin_y: Optional[float] = None,
    window_seconds: float = 300.0,  # +/- 5 minutes
    max_stationary_speed_mps: float = 1.0,
    max_checkin_dist_m: float = 50.0,
) -> Tuple[float, float]:
    """Compute spatial median coordinate during the check-in stationary dwell window.
    
    Filters GPS points where instantaneous speed < 1.0 m/s within +/- 5 minutes of check-in
    and within 50m of the check-in coordinate.
    Falls back to (checkin_x, checkin_y) if no valid stationary dwell points exist.
    """
    fb_x = float(checkin_x) if checkin_x is not None else 0.0
    fb_y = float(checkin_y) if checkin_y is not None else 0.0

    if df_single_gps is None or len(df_single_gps) == 0:
        return fb_x, fb_y

    gps = df_single_gps.copy()
    if checkin_ts is not None and "point_ts" in gps.columns:
        ts_checkin = pd.to_datetime(checkin_ts)
        gps["point_ts_dt"] = pd.to_datetime(gps["point_ts"])
        time_diff = (gps["point_ts_dt"] - ts_checkin).abs().dt.total_seconds()
        in_window = gps[time_diff <= window_seconds].copy()

        if len(in_window) > 1:
            dx = in_window["x"].diff()
            dy = in_window["y"].diff()
            dt = in_window["point_ts_dt"].diff().dt.total_seconds()
            spd = np.where(dt > 0, np.hypot(dx, dy) / dt, 0.0)
            in_window["speed_mps"] = spd

            dist_to_c = np.hypot(in_window["x"] - fb_x, in_window["y"] - fb_y)
            stationary = in_window[
                (in_window["speed_mps"] < max_stationary_speed_mps) & (dist_to_c <= max_checkin_dist_m)
            ]
            if len(stationary) > 0:
                x_vals = stationary["x"].to_numpy(dtype=np.float64)
                y_vals = stationary["y"].to_numpy(dtype=np.float64)
                return float(np.median(x_vals)), float(np.median(y_vals))
        elif len(in_window) == 1:
            dist_to_c = float(np.hypot(in_window["x"].iloc[0] - fb_x, in_window["y"].iloc[0] - fb_y))
            if dist_to_c <= max_checkin_dist_m:
                return float(in_window["x"].iloc[0]), float(in_window["y"].iloc[0])

    # Clean fallback to checkin coordinates
    return fb_x, fb_y


def extract_trajectory_features_batch(
    df_gps_points: pd.DataFrame,
    df_visits: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract trajectory physics, speeds, tortuosity, and dwell medoids for all visits."""
    df_gps = df_gps_points.sort_values(["visit_id", "seq"]).copy()
    df_gps["point_ts_dt"] = pd.to_datetime(df_gps["point_ts"])

    # Pre-aggregate checkin info if visits table provided
    checkin_map: Dict[str, Tuple[Any, float, float]] = {}
    if df_visits is not None:
        for _, row in df_visits.iterrows():
            checkin_map[str(row["visit_id"])] = (
                row.get("checkin_ts"),
                float(row.get("checkin_x", 0.0)),
                float(row.get("checkin_y", 0.0)),
            )

    records: List[Dict[str, Any]] = []

    for vid, group in df_gps.groupby("visit_id"):
        x_raw = group["x"].to_numpy(dtype=np.float64)
        y_raw = group["y"].to_numpy(dtype=np.float64)
        ts_raw = group["point_ts_dt"].to_numpy()
        acc_raw = group["accuracy_m"].to_numpy(dtype=np.float64) if "accuracy_m" in group.columns else None

        # 1. Apply GPS speed jump filter
        x, y, ts, num_jumps = filter_gps_jumps(x_raw, y_raw, ts_raw, max_speed_mps=40.0)

        n_pts = len(x)
        if n_pts == 0:
            continue

        # 2. Distance and displacement
        if n_pts > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            dists = np.hypot(dx, dy)
            total_dist = float(np.sum(dists))
            net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        else:
            total_dist = 0.0
            net_disp = 0.0

        # 3. Speed calculation
        if n_pts > 1 and ts is not None:
            dt_s = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
            valid_dt = dt_s > 0
            if np.any(valid_dt):
                speeds_mps = dists[valid_dt] / dt_s[valid_dt]
                mean_spd_kmh = float(np.mean(speeds_mps) * 3.6)
                max_spd_kmh = float(np.max(speeds_mps) * 3.6)
            else:
                mean_spd_kmh = 0.0
                max_spd_kmh = 0.0
        else:
            mean_spd_kmh = 0.0
            max_spd_kmh = 0.0

        # 4. Tortuosity with guards
        acc = acc_raw[:n_pts] if acc_raw is not None else None
        tortuosity = compute_path_tortuosity(x, y, accuracy_m=acc)
        is_guarded = (n_pts < 3) or (net_disp <= 15.0)

        # 5. Radius of gyration
        rad_gyr = compute_radius_of_gyration(x, y)

        # 6. Dwell medoid
        ch_info = checkin_map.get(str(vid))
        if ch_info:
            c_ts, c_x, c_y = ch_info
            dwell_x, dwell_y = build_visit_dwell_medoid(group, checkin_ts=c_ts, checkin_x=c_x, checkin_y=c_y)
        else:
            dwell_x, dwell_y = compute_spatial_medoid(x, y)

        records.append({
            "visit_id": str(vid),
            "point_count": int(n_pts),
            "total_distance_m": round(total_dist, 2),
            "net_displacement_m": round(net_disp, 2),
            "tortuosity": round(tortuosity, 3),
            "radius_of_gyration_m": round(rad_gyr, 2),
            "mean_speed_kmh": round(mean_spd_kmh, 2),
            "max_speed_kmh": round(max_spd_kmh, 2),
            "num_jumps_filtered": int(num_jumps),
            "is_trajectory_guarded": bool(is_guarded),
            "dwell_medoid_x": round(dwell_x, 2),
            "dwell_medoid_y": round(dwell_y, 2),
        })

    return pd.DataFrame(records)


def compute_address_doorstep_medoid(
    address_id: str,
    df_sanitized_visits: pd.DataFrame,
    min_integrity: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """Compute the weighted doorstep coordinate medoid across credible visits for an address.
    
    Filters for positive doorstep outcomes ('met_family', 'locked_premises', 'met_borrower', 'cash_collected')
    with operational integrity score >= min_integrity.
    
    Weights each candidate visit by w_v = I(v) * S(v).
    """
    addr_visits = df_sanitized_visits[df_sanitized_visits["address_id"] == address_id]
    if len(addr_visits) == 0:
        return None

    # 1. Prioritize credible positive doorstep visits
    positive_visits = addr_visits[addr_visits["is_doorstep_confirmed"]]
    credible_visits = positive_visits[positive_visits["visit_integrity_score"] >= min_integrity]

    if len(credible_visits) > 0:
        selected_subset = credible_visits
        has_credible = True
    elif len(positive_visits) > 0:
        selected_subset = positive_visits
        has_credible = True
    else:
        selected_subset = addr_visits
        has_credible = False

    # Checkin coordinate is the primary doorstep anchor logged at premises
    x_col = "checkin_x" if "checkin_x" in selected_subset.columns else "dwell_medoid_x"
    y_col = "checkin_y" if "checkin_y" in selected_subset.columns else "dwell_medoid_y"

    x_vals = selected_subset[x_col].to_numpy(dtype=np.float64)
    y_vals = selected_subset[y_col].to_numpy(dtype=np.float64)

    # Weights: w = I(v) * S(v)
    i_vals = selected_subset["visit_integrity_score"].to_numpy(dtype=np.float64)
    s_vals = selected_subset["spatial_evidence_strength"].to_numpy(dtype=np.float64)
    weights = np.maximum(i_vals * s_vals, 1e-4)

    med_x, med_y = compute_spatial_medoid(x_vals, y_vals, weights=weights)

    return {
        "address_id": address_id,
        "doorstep_medoid_x": round(med_x, 2),
        "doorstep_medoid_y": round(med_y, 2),
        "contributing_visit_ids": list(selected_subset["visit_id"].astype(str)),
        "contributing_count": int(len(selected_subset)),
        "total_evidence_weight": round(float(np.sum(weights)), 4),
        "has_credible_doorstep": bool(has_credible),
    }


def compute_all_address_doorstep_medoids(
    df_sanitized_visits: pd.DataFrame,
    min_integrity: float = 0.50,
) -> pd.DataFrame:
    """Compute doorstep medoids for all visited addresses in the dataset."""
    unique_addresses = df_sanitized_visits["address_id"].unique()
    records: List[Dict[str, Any]] = []

    for aid in unique_addresses:
        res = compute_address_doorstep_medoid(aid, df_sanitized_visits, min_integrity=min_integrity)
        if res:
            records.append(res)

    return pd.DataFrame(records)
