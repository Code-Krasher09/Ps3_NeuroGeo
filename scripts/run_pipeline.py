"""End-to-End Pipeline Runner: Full NeuroGeo-Field Inference & Verification.
Written using purely vectorized Pandas and NumPy operations, eliminating custom classes.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Path configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SHARED_DIR = DATA_DIR / "shared"
NEUROGEO_DIR = DATA_DIR / "ps3_geocoder"
METRICS_DIR = REPO_ROOT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def compute_distance_vectorized(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    print("=" * 60)
    print("NeuroGeo-Field: End-to-End Pure Pandas/NumPy Pipeline")
    print("=" * 60)
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    print("[1/5] Loading datasets...")
    df_addr = pd.read_csv(SHARED_DIR / "addresses.csv")
    df_gps = pd.read_csv(NEUROGEO_DIR / "visit_gps_points.csv")
    df_visits = pd.read_csv(SHARED_DIR / "field_visits.csv")
    df_pois = pd.read_csv(NEUROGEO_DIR / "landmarks_poi.csv")
    df_localities = pd.read_csv(NEUROGEO_DIR / "localities.csv")
    df_base = pd.read_csv(NEUROGEO_DIR / "baseline_geocodes.csv")

    # -------------------------------------------------------------------------
    # STEP 2: Phase 1 - Trajectory Mining & Sanitization (Pure Pandas)
    # -------------------------------------------------------------------------
    print("[2/5] Phase 1: Mining trajectories...")
    df_gps = df_gps.sort_values(["visit_id", "seq"]).copy()
    
    # Calculate distance and displacement
    df_gps["dx"] = df_gps.groupby("visit_id")["x"].diff()
    df_gps["dy"] = df_gps.groupby("visit_id")["y"].diff()
    df_gps["dist"] = np.hypot(df_gps["dx"], df_gps["dy"])
    
    df_traj = df_gps.groupby("visit_id").agg(
        total_dist_m=("dist", "sum"),
        x_first=("x", "first"),
        y_first=("y", "first"),
        x_last=("x", "last"),
        y_last=("y", "last"),
        dwell_medoid_x=("x", "median"),
        dwell_medoid_y=("y", "median"),
        point_count=("x", "count")
    ).reset_index()
    
    df_traj["net_disp_m"] = np.hypot(df_traj["x_last"] - df_traj["x_first"], df_traj["y_last"] - df_traj["y_first"])
    df_traj["tortuosity"] = np.where(
        (df_traj["net_disp_m"] > 15.0) & (df_traj["point_count"] >= 3),
        df_traj["total_dist_m"] / df_traj["net_disp_m"],
        1.0
    )
    
    # Merge visits and trajectories
    df_visits = df_visits.merge(df_traj, on="visit_id", how="left")
    df_visits["is_credible"] = df_visits["outcome"].isin(["met_family", "met_borrower", "cash_collected", "locked_premises"])
    df_visits["visit_integrity_score"] = np.where(df_visits["is_credible"], 0.9, 0.1) # simplified scoring
    df_visits["spatial_evidence_strength"] = 0.8 # simplified S(v)
    
    # Address-level Doorstep Medoid
    credible_visits = df_visits[df_visits["is_credible"]].copy()
    if len(credible_visits) > 0:
        df_medoids = credible_visits.groupby("address_id").agg(
            doorstep_medoid_x=("checkin_x", "median"),
            doorstep_medoid_y=("checkin_y", "median")
        ).reset_index()
        df_medoids["has_credible_doorstep"] = True
    else:
        df_medoids = pd.DataFrame(columns=["address_id", "doorstep_medoid_x", "doorstep_medoid_y", "has_credible_doorstep"])

    # -------------------------------------------------------------------------
    # STEP 3: Phase 2 - Text Parsing & POI Resolution (Pure Pandas)
    # -------------------------------------------------------------------------
    print("[3/5] Phase 2: Parsing addresses...")
    df_parsed = df_addr.copy()
    df_parsed["norm_text"] = df_parsed["address_text"].str.lower().str.replace(r"[^\w\s]", " ", regex=True)
    
    # Simple Locality Extraction
    df_localities["norm_loc"] = df_localities["locality_name"].str.lower().str.replace(r"[^\w\s]", " ", regex=True)
    df_parsed["locality_id_resolved"] = np.nan
    
    for _, row in df_localities.iterrows():
        loc_id = str(row["locality_id"])
        norm_loc = str(row["norm_loc"])
        if len(norm_loc) > 3:
            mask = df_parsed["norm_text"].str.contains(norm_loc, na=False, regex=False)
            df_parsed.loc[mask & df_parsed["locality_id_resolved"].isna(), "locality_id_resolved"] = loc_id
            
    # Simple POI Extraction
    df_pois["norm_poi"] = df_pois["name"].str.lower().str.replace(r"[^\w\s]", " ", regex=True)
    df_parsed["landmark_poi_id"] = np.nan
    for _, row in df_pois.iterrows():
        poi_id = str(row["poi_id"])
        norm_poi = str(row["norm_poi"])
        if len(norm_poi) > 4:
            mask = df_parsed["norm_text"].str.contains(norm_poi, na=False, regex=False)
            df_parsed.loc[mask & df_parsed["landmark_poi_id"].isna(), "landmark_poi_id"] = poi_id
            
    # Street Extraction
    df_parsed["street_normalized"] = df_parsed["norm_text"].str.extract(r"(\d+(?:st|nd|rd|th)?\s+(?:cross|main|road))")[0]
    
    # -------------------------------------------------------------------------
    # STEP 4: Phase 3 & 4 - Candidate Scorer (Pure Pandas)
    # -------------------------------------------------------------------------
    print("[4/5] Phase 4: Candidate Scoring...")
    df_wide = df_addr[["address_id", "account_id"]].copy()
    
    # C1: Baseline
    df_wide = df_wide.merge(df_base[["address_id", "geocoder_x", "geocoder_y"]], on="address_id", how="left")
    df_wide["C1_x"] = df_wide["geocoder_x"]
    df_wide["C1_y"] = df_wide["geocoder_y"]
    df_wide["C1_avail"] = df_wide["C1_x"].notna()
    
    # C2: Medoid
    df_wide = df_wide.merge(df_medoids[["address_id", "doorstep_medoid_x", "doorstep_medoid_y", "has_credible_doorstep"]], on="address_id", how="left")
    df_wide["has_credible_doorstep"] = df_wide["has_credible_doorstep"].fillna(False)
    df_wide["C2_x"] = df_wide["doorstep_medoid_x"]
    df_wide["C2_y"] = df_wide["doorstep_medoid_y"]
    df_wide["C2_avail"] = df_wide["C2_x"].notna() & df_wide["has_credible_doorstep"]
    
    # C3: POI
    df_parsed["landmark_poi_id"] = df_parsed["landmark_poi_id"].astype(str)
    df_pois["poi_id"] = df_pois["poi_id"].astype(str)
    df_wide = df_wide.merge(df_parsed[["address_id", "landmark_poi_id"]], on="address_id", how="left")
    df_wide = df_wide.merge(df_pois[["poi_id", "x", "y"]].rename(columns={"poi_id": "landmark_poi_id", "x": "poi_x", "y": "poi_y"}), on="landmark_poi_id", how="left")
    df_wide["C3_x"] = df_wide["poi_x"]
    df_wide["C3_y"] = df_wide["poi_y"]
    df_wide["C3_avail"] = df_wide["C3_x"].notna()
    
    # C4: Street Centroid
    df_streets = df_wide.merge(df_parsed[["address_id", "street_normalized", "locality_id_resolved"]], on="address_id", how="left")
    street_centroids = df_streets.dropna(subset=["geocoder_x", "geocoder_y"]).groupby(["locality_id_resolved", "street_normalized"]).agg(
        C4_x=("geocoder_x", "mean"),
        C4_y=("geocoder_y", "mean"),
        neighbor_count=("address_id", "count")
    ).reset_index()
    df_wide = df_wide.merge(df_parsed[["address_id", "locality_id_resolved", "street_normalized"]], on="address_id", how="left")
    df_wide = df_wide.merge(street_centroids, on=["locality_id_resolved", "street_normalized"], how="left")
    df_wide["C4_avail"] = df_wide["C4_x"].notna() & (df_wide["neighbor_count"] > 1)
    
    # C5: Locality Centroid
    df_wide["locality_id_resolved"] = df_wide["locality_id_resolved"].astype(str)
    df_localities["locality_id"] = df_localities["locality_id"].astype(str)
    df_wide = df_wide.merge(df_localities[["locality_id", "centroid_x", "centroid_y"]].rename(columns={"locality_id": "locality_id_resolved"}), on="locality_id_resolved", how="left")
    df_wide["C5_x"] = df_wide["centroid_x"]
    df_wide["C5_y"] = df_wide["centroid_y"]
    df_wide["C5_avail"] = df_wide["C5_x"].notna()

    # Reshape to long
    candidates = []
    for c_type in ["C1", "C2", "C3", "C4", "C5"]:
        c_df = df_wide[["address_id", f"{c_type}_x", f"{c_type}_y", f"{c_type}_avail", "geocoder_x", "geocoder_y"]].copy()
        c_df = c_df[c_df[f"{c_type}_avail"] == True].copy()
        c_df.rename(columns={f"{c_type}_x": "x", f"{c_type}_y": "y"}, inplace=True)
        c_df["candidate_type"] = c_type
        
        # TextMatch Base
        tm_map = {"C1": 0.85, "C2": 1.0, "C3": 0.90, "C4": 0.90, "C5": 0.70}
        c_df["TextMatch"] = tm_map[c_type]
        candidates.append(c_df)
        
    df_long = pd.concat(candidates, ignore_index=True)
    
    # Features
    sigma_b, sigma_v = 200.0, 50.0
    for col in ["x", "y", "geocoder_x", "geocoder_y"]:
        df_long[col] = pd.to_numeric(df_long[col], errors="coerce")
        
    df_long["BaselineAgreement"] = np.exp(-(compute_distance_vectorized(df_long["x"], df_long["y"], df_long["geocoder_x"], df_long["geocoder_y"])**2) / (2.0 * sigma_b**2)).fillna(0.0)
    
    df_v = df_visits.groupby("address_id").agg(mean_v_x=("checkin_x", "mean"), mean_v_y=("checkin_y", "mean")).reset_index()
    df_long = df_long.merge(df_v, on="address_id", how="left")
    for col in ["mean_v_x", "mean_v_y"]:
        df_long[col] = pd.to_numeric(df_long[col], errors="coerce")
        
    dist_v = compute_distance_vectorized(df_long["x"], df_long["y"], df_long["mean_v_x"], df_long["mean_v_y"])
    df_long["DirectEvidence"] = np.where(df_long["mean_v_x"].notna(), np.exp(-(dist_v**2) / (2.0 * sigma_v**2)), 0.0)
    df_long["NeighborSupport"] = 0.5
    df_long["NegativeSearchPenalty"] = 0.0
    
    # Score
    W = {"w_text": 1.0, "w_visit": 2.5, "w_neigh": 1.5, "w_base": 0.5, "w_neg": 2.0}
    df_long["total_score"] = (
        W["w_text"] * df_long["TextMatch"] + W["w_visit"] * df_long["DirectEvidence"] +
        W["w_neigh"] * df_long["NeighborSupport"] + W["w_base"] * df_long["BaselineAgreement"] -
        W["w_neg"] * df_long["NegativeSearchPenalty"]
    )
    
    priority_map = {"C2": 1, "C4": 2, "C3": 3, "C1": 4, "C5": 5}
    df_long["priority"] = df_long["candidate_type"].map(priority_map)
    df_long.sort_values(by=["address_id", "total_score", "priority"], ascending=[True, False, True], inplace=True)
    df_long["rank"] = df_long.groupby("address_id").cumcount() + 1
    
    df_top1 = df_long[df_long["rank"] == 1].copy()

    # -------------------------------------------------------------------------
    # STEP 5: Phase 5 - Calibration & Export
    # -------------------------------------------------------------------------
    print("[5/5] Phase 5: Calibration & Export...")
    df_preds = df_top1[["address_id", "x", "y", "candidate_type", "total_score"]].rename(
        columns={"x": "predicted_x", "y": "predicted_y", "candidate_type": "selected_candidate"}
    )
    df_preds["confidence_radius_m"] = np.select(
        [df_preds["selected_candidate"] == "C2", df_preds["selected_candidate"] == "C4", df_preds["selected_candidate"] == "C3"],
        [15.0, 50.0, 150.0], default=500.0
    )
    df_preds["status_tier"] = np.select(
        [df_preds["selected_candidate"] == "C2", df_preds["selected_candidate"] == "C4"],
        ["DOORSTEP_VERIFIED", "STREET_RESOLVED"], default="BASELINE_FALLBACK"
    )
    
    # Output
    out_csv = METRICS_DIR / "pipeline_predictions.csv"
    df_preds.to_csv(out_csv, index=False)
    
    # Create final JSON artifacts to replace parquet needs
    out_json = METRICS_DIR / "final_report.json"
    report = {
        "execution_time_s": round(time.time() - t0, 2),
        "addresses_processed": len(df_preds),
        "status_distribution": df_preds["status_tier"].value_counts().to_dict(),
        "candidate_distribution": df_preds["selected_candidate"].value_counts().to_dict(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Pipeline complete in {report['execution_time_s']}s.")
    print(f"Predictions written to {out_csv}")
    print(f"Report written to {out_json}")

if __name__ == "__main__":
    main()
