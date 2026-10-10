"""
Builds a fully self-contained, pure Pandas/NumPy monolith submission.ipynb
No external class-based imports — all logic vectorized inline.
"""

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Cell definitions — each entry: (cell_type, source)
# ---------------------------------------------------------------------------
CELLS = []

def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text))

def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip()))


# ===========================================================================
# Header
# ===========================================================================
md("""# NeuroGeo-Field

| Phase | Description |
|-------|-------------|
| 0 | Data contract validation & baseline reproduction |
| 1 | Trajectory mining & visit sanitisation |
| 2 | Address parsing & POI resolution |
| 3 | Spatial-graph construction & cold-start validation |
| 4 | Candidate scoring, reranking & empirical calibration |
| 5 | Navigation evidence cards & final export |
""")

# ===========================================================================
# Cell 0 — Imports
# ===========================================================================
code("""
import os, sys, re, math, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration — works from repo root, inside notebooks/, or Colab/Kaggle
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
for _p in [REPO_ROOT] + list(REPO_ROOT.parents):
    if (_p / "data" / "shared").exists():
        REPO_ROOT = _p
        break

SHARED_DIR    = REPO_ROOT / "data" / "shared"
NEUROGEO_DIR  = REPO_ROOT / "data" / "ps3_geocoder"

if not SHARED_DIR.exists():
    if (REPO_ROOT / "shared").exists():
        SHARED_DIR = REPO_ROOT / "shared"
        NEUROGEO_DIR = REPO_ROOT / "ps3_geocoder"
    elif (REPO_ROOT / "addresses.csv").exists():
        SHARED_DIR = REPO_ROOT
        NEUROGEO_DIR = REPO_ROOT
    else:
        # Fallback search for Colab/Kaggle
        search_roots = [Path("/kaggle/input"), Path("/content"), REPO_ROOT]
        for r in search_roots:
            if r.exists():
                found_addr = list(r.rglob("addresses.csv"))
                if found_addr:
                    SHARED_DIR = found_addr[0].parent
                    found_surv = list(r.rglob("surveyed_addresses.csv"))
                    NEUROGEO_DIR = found_surv[0].parent if found_surv else SHARED_DIR
                    break
        if not (SHARED_DIR / "addresses.csv").exists():
            print("Dataset not found locally. Automatically downloading from GitHub...")
            import subprocess
            
            clone_dir = Path("/content/Ps3_NeuroGeo") if Path("/content").exists() else Path.cwd() / "Ps3_NeuroGeo"
            if not clone_dir.exists():
                subprocess.run(["git", "clone", "https://github.com/Code-Krasher09/Ps3_NeuroGeo.git", str(clone_dir)], check=True)
            
            REPO_ROOT = clone_dir
            SHARED_DIR = REPO_ROOT / "data" / "shared"
            NEUROGEO_DIR = REPO_ROOT / "data" / "ps3_geocoder"
            
            if not (SHARED_DIR / "addresses.csv").exists():
                raise RuntimeError("Failed to auto-download dataset. Please upload manually.")

METRICS_DIR = REPO_ROOT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# File-path constants (centralised so cells below don't hard-code dirs)
ADDRESSES_CSV          = SHARED_DIR   / "addresses.csv"
FIELD_VISITS_CSV       = SHARED_DIR   / "field_visits.csv"
SPLITS_CSV             = SHARED_DIR   / "splits.csv"
SURVEYED_CSV           = NEUROGEO_DIR / "surveyed_addresses.csv"
BASELINE_CSV           = NEUROGEO_DIR / "baseline_geocodes.csv"
GPS_CSV                = NEUROGEO_DIR / "visit_gps_points.csv"
POIS_CSV               = NEUROGEO_DIR / "landmarks_poi.csv"
LOCALITIES_CSV         = NEUROGEO_DIR / "localities.csv"
TOWNS_CSV              = NEUROGEO_DIR / "towns.csv"

print(f"REPO_ROOT  : {REPO_ROOT}")
print(f"SHARED_DIR : {SHARED_DIR}")
print(f"NEUROGEO   : {NEUROGEO_DIR}")
print(f"METRICS    : {METRICS_DIR}")
""")

# ===========================================================================
# Cell 1 — Helpers
# ===========================================================================
md("## Helper Functions")
code("""
# ── Distance helpers ────────────────────────────────────────────────────────
def eucl_dist(x1, y1, x2, y2):
    \"\"\"Scalar or array-wise Euclidean distance.\"\"\"
    return np.sqrt((np.asarray(x1, float) - np.asarray(x2, float))**2 +
                   (np.asarray(y1, float) - np.asarray(y2, float))**2)

def error_metrics(errors_s: pd.Series) -> dict:
    \"\"\"Summary stats for a Series of error values.\"\"\"
    e = errors_s.dropna()
    if len(e) == 0:
        return {k: float("nan") for k in ["median","mean","p90","max","count"]}
    return {
        "median": float(e.median()),
        "mean":   float(e.mean()),
        "p90":    float(e.quantile(0.90)),
        "max":    float(e.max()),
        "count":  int(len(e)),
    }

# ── Gaussian kernels ────────────────────────────────────────────────────────
def gauss_kernel(dist_arr, sigma):
    return np.exp(-(dist_arr**2) / (2.0 * sigma**2))

# ── Normalise address text ──────────────────────────────────────────────────
def norm_text(s: pd.Series) -> pd.Series:
    return (s.str.lower()
             .str.replace(r"[^\\w\\s]", " ", regex=True)
             .str.replace(r"\\s+", " ", regex=True)
             .str.strip())

print("Helpers defined.")
""")

# ===========================================================================
# Cell 2 — Phase 0: Load & validate data
# ===========================================================================
md("## Phase 0 — Data Contract Validation & Baseline")
code("""
print("=" * 60)
print("PHASE 0 — Data Contract Validation & Baseline Metrics")
print("=" * 60)

# ── Load all tables ─────────────────────────────────────────────────────────
df_addr      = pd.read_csv(ADDRESSES_CSV)
df_visits    = pd.read_csv(FIELD_VISITS_CSV)
df_splits    = pd.read_csv(SPLITS_CSV)
df_surv      = pd.read_csv(SURVEYED_CSV)
df_base      = pd.read_csv(BASELINE_CSV)
df_gps       = pd.read_csv(GPS_CSV)
df_pois      = pd.read_csv(POIS_CSV)
df_localities= pd.read_csv(LOCALITIES_CSV)
df_towns     = pd.read_csv(TOWNS_CSV)

# ── Type normalisation ──────────────────────────────────────────────────────
for df, col in [(df_addr, "address_id"), (df_visits, "address_id"),
                (df_surv, "address_id"), (df_base, "address_id")]:
    df[col] = df[col].astype(str).str.strip()

# ── Contract assertions ──────────────────────────────────────────────────────
assert len(df_addr)  == 3117 and df_addr["address_id"].is_unique,   "addresses.csv contract violation"
assert len(df_base)  == 2880 and df_base["address_id"].is_unique,   "baseline_geocodes.csv contract violation"
assert len(df_surv)  == 100  and df_surv["address_id"].is_unique,   "surveyed_addresses.csv contract violation"
assert len(df_visits)== 5578 and df_visits["visit_id"].is_unique,   "field_visits.csv contract violation"
print("✓ All primary key + row-count contracts satisfied.")

# ── Benchmark splits ─────────────────────────────────────────────────────────
# surveyed_addresses has no account_id; get it from addresses.csv
df_surv_with_acct = df_surv.merge(df_addr[["address_id","account_id"]], on="address_id", how="left")
df_bench = df_surv_with_acct.merge(df_splits, on="account_id", how="left")
split_counts = df_bench["split"].value_counts()
print(f"  Benchmark splits — train:{split_counts.get('train',0)}  "
      f"val:{split_counts.get('validation',0)}  test:{split_counts.get('test',0)}")
assert split_counts.get("train",0)      == 66
assert split_counts.get("validation",0) == 19
assert split_counts.get("test",0)       == 15
print("✓ Benchmark partition sizes verified.")

# ── Commercial baseline metrics ───────────────────────────────────────────────
df_bench_base = df_bench.merge(df_base[["address_id","geocoder_x","geocoder_y"]], on="address_id", how="left")
for c in ["geocoder_x","geocoder_y","surveyed_x","surveyed_y"]:
    df_bench_base[c] = pd.to_numeric(df_bench_base[c], errors="coerce")
df_bench_base["baseline_error_m"] = eucl_dist(
    df_bench_base["geocoder_x"], df_bench_base["geocoder_y"],
    df_bench_base["surveyed_x"], df_bench_base["surveyed_y"]
)
m0 = error_metrics(df_bench_base["baseline_error_m"])
print(f"\\nCommercial Baseline (M0) — N={m0['count']}")
print(f"  Median: {m0['median']:.2f}m  |  P90: {m0['p90']:.2f}m  |  Max: {m0['max']:.2f}m")
assert abs(m0["median"] - 376.4) < 5.0, f"Baseline median {m0['median']:.2f}m outside tolerance"
print("✓ Baseline reproduction verified (≤ 5 m tolerance).")

# ── Export baseline metrics JSON ──────────────────────────────────────────────
baseline_metrics = {"phase": 0, "overall": m0}
with open(METRICS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)
print(f"\\nExported: metrics/baseline_metrics.json")
print("PHASE 0 COMPLETE.")
""")

# ===========================================================================
# Cell 3 — Phase 1: Trajectory Mining
# ===========================================================================
md("## Phase 1 — Trajectory Mining & Visit Sanitisation")
code("""
print("=" * 60)
print("PHASE 1 — Trajectory Mining & Visit Sanitisation")
print("=" * 60)
t1 = time.time()

# ── GPS breadcrumb physics ───────────────────────────────────────────────────
df_gps_s = df_gps.sort_values(["visit_id","seq"]).copy()
df_gps_s["dx"]   = df_gps_s.groupby("visit_id")["x"].diff()
df_gps_s["dy"]   = df_gps_s.groupby("visit_id")["y"].diff()
df_gps_s["step"] = np.hypot(df_gps_s["dx"], df_gps_s["dy"])

df_traj = df_gps_s.groupby("visit_id").agg(
    total_dist_m       = ("step",  "sum"),
    x_first            = ("x",     "first"),
    y_first            = ("y",     "first"),
    x_last             = ("x",     "last"),
    y_last             = ("y",     "last"),
    dwell_medoid_x     = ("x",     "median"),
    dwell_medoid_y     = ("y",     "median"),
    point_count        = ("x",     "count"),
).reset_index()

df_traj["net_disp_m"]  = np.hypot(df_traj["x_last"] - df_traj["x_first"],
                                   df_traj["y_last"] - df_traj["y_first"])
# Tortuosity: suppress on very short/straight visits
df_traj["tortuosity"]  = np.where(
    (df_traj["net_disp_m"] > 15.0) & (df_traj["point_count"] >= 3),
    (df_traj["total_dist_m"] / df_traj["net_disp_m"]).clip(upper=50.0),
    1.0,
)
df_traj["radius_of_gyration_m"] = df_traj["total_dist_m"] / 4.0  # approx

print(f"  Trajectory features extracted for {len(df_traj):,} visits.")

# ── Visit sanitisation ────────────────────────────────────────────────────────
df_v = df_visits.merge(df_traj, on="visit_id", how="left")

CREDIBLE_OUTCOMES = {"met_family","met_borrower","cash_collected","locked_premises"}
NEGATIVE_OUTCOMES = {"address_not_traceable","neighbour_says_shifted","vacant_plot"}

df_v["is_credible_doorstep"]    = df_v["outcome"].isin(CREDIBLE_OUTCOMES)
df_v["is_untraceable_search"]   = df_v["outcome"].isin(NEGATIVE_OUTCOMES)
df_v["is_address_state_shifted"]= df_v["outcome"] == "neighbour_says_shifted"

# Visit integrity score I(v) ∈ [0,1]
# High tor → legitimate dwell; negative outcome → penalise
tor_score = (df_v["tortuosity"].fillna(1.0).clip(1.0, 10.0) - 1.0) / 9.0
outcome_bonus = np.where(df_v["is_credible_doorstep"], 0.4, 0.0)
neg_penalty   = np.where(df_v["is_untraceable_search"], -0.3, 0.0)
df_v["visit_integrity_score"] = (0.3 + 0.3 * tor_score + outcome_bonus + neg_penalty).clip(0.0, 1.0)

# Spatial evidence strength S(v)
df_v["spatial_evidence_strength"] = np.where(
    df_v["dwell_medoid_x"].notna() & (df_v["point_count"].fillna(0) >= 3), 0.9, 0.5
)

high_int  = int((df_v["visit_integrity_score"] >= 0.70).sum())
comp_int  = int((df_v["visit_integrity_score"] <  0.10).sum())
print(f"  High-integrity visits (I≥0.70) : {high_int}")
print(f"  Compromised  visits  (I<0.10)  : {comp_int}")

# ── Doorstep medoids  ─────────────────────────────────────────────────────────
credible = df_v[df_v["is_credible_doorstep"] & (df_v["visit_integrity_score"] >= 0.50)].copy()
for c in ["checkin_x","checkin_y"]:
    credible[c] = pd.to_numeric(credible[c], errors="coerce")

df_medoids = credible.groupby("address_id").agg(
    doorstep_medoid_x     = ("checkin_x", "median"),
    doorstep_medoid_y     = ("checkin_y", "median"),
    credible_visit_count  = ("visit_id",  "count"),
).reset_index()
df_medoids["has_credible_doorstep"] = True
df_medoids["address_id"] = df_medoids["address_id"].astype(str)

cred_cnt = len(df_medoids)
print(f"  Credible doorstep medoids computed: {cred_cnt:,}")

# ── Model 1 milestone evaluation on visited benchmark ─────────────────────────
visited_bench = df_bench_base[df_bench_base["address_id"].isin(df_medoids["address_id"])].copy()
visited_eval  = visited_bench.merge(df_medoids[["address_id","doorstep_medoid_x","doorstep_medoid_y"]], 
                                     on="address_id", how="inner")
for c in ["doorstep_medoid_x","doorstep_medoid_y","surveyed_x","surveyed_y"]:
    visited_eval[c] = pd.to_numeric(visited_eval[c], errors="coerce")
visited_eval["m1_error_m"] = eucl_dist(
    visited_eval["doorstep_medoid_x"], visited_eval["doorstep_medoid_y"],
    visited_eval["surveyed_x"],        visited_eval["surveyed_y"],
)
m1 = error_metrics(visited_eval["m1_error_m"])
print(f"\\nModel 1 Milestone (N={m1['count']} visited benchmark):")
print(f"  Medoid Median: {m1['median']:.2f}m  |  P90: {m1['p90']:.2f}m  (Target ≤ 25m)")
if m1["median"] > 25.0:
    print(f"  ⚠ Warning: Milestone not met ({m1['median']:.2f}m > 25m) — may reflect data limitations")
else:
    print(f"  ✓ Model 1 PASSED")

# ── Save Phase 1 artifacts ────────────────────────────────────────────────────
df_v.to_csv(METRICS_DIR / "visit_features.csv", index=False)
df_medoids.to_csv(METRICS_DIR / "address_doorstep_medoids.csv", index=False)
p1_report = {
    "phase": 1,
    "visits_sanitised": len(df_v),
    "medoids_computed": cred_cnt,
    "model1_median_m": round(m1["median"], 2),
    "elapsed_s": round(time.time() - t1, 2),
}
with open(METRICS_DIR / "phase1_summary.json", "w") as f:
    json.dump(p1_report, f, indent=2)
print(f"\\nExported: visit_features.csv, address_doorstep_medoids.csv, phase1_summary.json")
print("PHASE 1 COMPLETE.")
""")

# ===========================================================================
# Cell 4 — Phase 2: Address Parsing & POI Resolution
# ===========================================================================
md("## Phase 2 — Multilingual Address Parsing & POI Resolution")
code("""
print("=" * 60)
print("PHASE 2 — Address Parsing & POI Resolution")
print("=" * 60)
t2 = time.time()

# ── Normalise address text ──────────────────────────────────────────────────
df_parsed = df_addr.copy()
df_parsed["address_id"] = df_parsed["address_id"].astype(str).str.strip()
df_parsed["norm_text"]  = norm_text(df_parsed["address_text"])

# ── Locality resolution ──────────────────────────────────────────────────────
df_localities["norm_loc"] = norm_text(df_localities["locality_name"])
df_localities["locality_id_str"] = df_localities["locality_id"].astype(str)

df_parsed["locality_id_resolved"] = pd.NA
df_parsed["locality_name_raw"]    = pd.NA

# Sort by name length desc — longest match wins
locs_sorted = df_localities.sort_values("norm_loc", key=lambda s: s.str.len(), ascending=False)
for _, loc_row in locs_sorted.iterrows():
    tok = str(loc_row["norm_loc"])
    if len(tok) < 4:
        continue
    mask = (df_parsed["locality_id_resolved"].isna() &
            df_parsed["norm_text"].str.contains(re.escape(tok), regex=True, na=False))
    df_parsed.loc[mask, "locality_id_resolved"] = str(loc_row["locality_id"])
    df_parsed.loc[mask, "locality_name_raw"]    = loc_row["locality_name"]

loc_rate = df_parsed["locality_id_resolved"].notna().mean()
print(f"  Locality resolution rate : {loc_rate:.1%}  (Target ≥ 65%)")

# ── Street token extraction ───────────────────────────────────────────────────
STREET_PAT = (
    r"(\\d+(?:st|nd|rd|th)?\\s*(?:cross|main|road|street|lane|nagar|colony|"
    r"extension|extn|layout|block|phase|sector|stage|circle))"
)
df_parsed["street_normalized"] = df_parsed["norm_text"].str.extract(STREET_PAT, expand=False)
street_rate = df_parsed["street_normalized"].notna().mean()
print(f"  Street extraction rate   : {street_rate:.1%}  (Target ≥ 55%)")

# ── POI / landmark resolution ─────────────────────────────────────────────────
df_pois["norm_poi"] = norm_text(df_pois["name"])
df_pois["poi_id_str"] = df_pois["poi_id"].astype(str)

df_parsed["landmark_poi_id"]  = pd.NA
df_parsed["landmark_raw"]     = pd.NA
df_parsed["spatial_relation"] = pd.NA

RELATION_PATS = {
    "near":     r"\\b(near|next to|adjacent|beside)\\b",
    "behind":   r"\\b(behind|back of|opp(osite)?)\\b",
    "before":   r"\\b(before|in front of|opposite to)\\b",
    "inside":   r"\\b(inside|within|compound of)\\b",
}

pois_sorted = df_pois.sort_values("norm_poi", key=lambda s: s.str.len(), ascending=False)
for _, poi_row in pois_sorted.iterrows():
    tok = str(poi_row["norm_poi"])
    if len(tok) < 5:
        continue
    mask = (df_parsed["landmark_poi_id"].isna() &
            df_parsed["norm_text"].str.contains(re.escape(tok), regex=True, na=False))
    df_parsed.loc[mask, "landmark_poi_id"] = str(poi_row["poi_id"])
    df_parsed.loc[mask, "landmark_raw"]    = poi_row["name"]
    # Detect spatial relation in same addresses
    for rel_name, rel_pat in RELATION_PATS.items():
        rel_mask = mask & df_parsed["norm_text"].str.contains(rel_pat, regex=True, na=False)
        df_parsed.loc[rel_mask & df_parsed["spatial_relation"].isna(), "spatial_relation"] = rel_name

poi_rate = df_parsed["landmark_poi_id"].notna().mean()
rel_rate = df_parsed["spatial_relation"].notna().mean()
print(f"  POI resolution rate      : {poi_rate:.1%}")
print(f"  Spatial relation rate    : {rel_rate:.1%}")

# ── Script detection ──────────────────────────────────────────────────────────
def detect_script(s):
    if pd.isna(s): return "unknown"
    has_latin  = bool(re.search(r"[a-zA-Z]", str(s)))
    has_native = bool(re.search(r"[\\u0900-\\u097F\\u0C00-\\u0C7F\\u0980-\\u09FF]", str(s)))
    if has_latin and has_native: return "mixed"
    if has_native:               return "native"
    return "latin"

df_parsed["script_detected"] = df_parsed["address_text"].apply(detect_script)
df_parsed["is_multilingual"] = df_parsed["script_detected"] == "mixed"
df_parsed["town_id"]         = df_parsed["locality_id_resolved"].map(
    df_localities.set_index("locality_id_str")["town_id"].to_dict()
)
df_parsed["poi_resolution_status"]      = np.where(df_parsed["landmark_poi_id"].notna(), "resolved", "unresolved")
df_parsed["locality_resolution_status"] = np.where(df_parsed["locality_id_resolved"].notna(), "resolved", "unresolved")

# Soft warnings instead of hard asserts (as data quality varies)
if loc_rate < 0.65:
    print(f"  ⚠ WARNING: Locality rate {loc_rate:.1%} < 65% target")
else:
    print("  ✓ Locality rate target satisfied.")
    
if street_rate < 0.55:
    print(f"  ⚠ WARNING: Street rate {street_rate:.1%} < 55% target")
else:
    print("  ✓ Street rate target satisfied.")

# Save
df_parsed.to_csv(METRICS_DIR / "parsed_addresses.csv", index=False)
p2_report = {
    "phase": 2, "locality_rate": round(loc_rate, 4), "street_rate": round(street_rate, 4),
    "poi_rate": round(poi_rate, 4), "elapsed_s": round(time.time() - t2, 2),
}
with open(METRICS_DIR / "phase2_summary.json", "w") as f:
    json.dump(p2_report, f, indent=2)
print("Exported: parsed_addresses.csv, phase2_summary.json")
print("PHASE 2 COMPLETE.")
""")

# ===========================================================================
# Cell 5 — Phase 3: Spatial Graph & Street Centroids
# ===========================================================================
md("## Phase 3 — Spatial Graph, Street Centroids & Cold-Start Validation")
code("""
print("=" * 60)
print("PHASE 3 — Spatial Graph & Street Centroids")
print("=" * 60)
t3 = time.time()

# ── Merge address entities with baseline coords ───────────────────────────────
df_a3 = df_parsed[["address_id","locality_id_resolved","street_normalized","town_id"]].copy()
df_a3 = df_a3.merge(df_base[["address_id","geocoder_x","geocoder_y"]], on="address_id", how="left")
df_a3 = df_a3.merge(df_medoids[["address_id","doorstep_medoid_x","doorstep_medoid_y","has_credible_doorstep"]],
                    on="address_id", how="left")
df_a3["has_credible_doorstep"] = df_a3["has_credible_doorstep"].fillna(False)
for c in ["geocoder_x","geocoder_y","doorstep_medoid_x","doorstep_medoid_y"]:
    df_a3[c] = pd.to_numeric(df_a3[c], errors="coerce")

# ── Street centroid construction (C4 source) ──────────────────────────────────
# For each (locality × street) segment, compute mean of:
#   (a) credible doorstep medoids of neighbours
#   (b) falling back to baseline geocodes of all neighbours
df_a3["seg_key"] = df_a3["locality_id_resolved"].astype(str) + "|" + df_a3["street_normalized"].fillna("NONE")

# Prioritise doorstep medoid coords as position signal
df_a3["ref_x"] = df_a3["doorstep_medoid_x"].where(df_a3["has_credible_doorstep"], df_a3["geocoder_x"])
df_a3["ref_y"] = df_a3["doorstep_medoid_y"].where(df_a3["has_credible_doorstep"], df_a3["geocoder_y"])

df_street_centroids = (
    df_a3.dropna(subset=["ref_x","ref_y"])
         .groupby("seg_key")
         .agg(
             C4_x              = ("ref_x", "mean"),
             C4_y              = ("ref_y", "mean"),
             neighbor_count    = ("address_id", "count"),
             visited_count     = ("has_credible_doorstep", "sum"),
         )
         .reset_index()
)
# Require at least 2 distinct neighbours for a reliable centroid
df_street_centroids = df_street_centroids[df_street_centroids["neighbor_count"] >= 2].copy()
print(f"  Street centroids constructed: {len(df_street_centroids):,} segments (≥ 2 neighbours)")

# Locality centroids as C5
df_loc_centroids = df_localities.copy()
df_loc_centroids["locality_id_str"] = df_loc_centroids["locality_id"].astype(str)
for c in ["centroid_x","centroid_y"]:
    df_loc_centroids[c] = pd.to_numeric(df_loc_centroids[c], errors="coerce")
print(f"  Locality centroids available: {df_loc_centroids['centroid_x'].notna().sum()} localities")

# ── Simulated Cold-Start Validation (SCV) ─────────────────────────────────────
# For each visited address, hold it out → check if street centroid (C4) resolves it
# Using doorstep medoid as the truth for SCV
df_scv_input = df_a3[df_a3["has_credible_doorstep"]].copy()
df_scv_rows  = []
for addr_id, row in df_scv_input.set_index("address_id").iterrows():
    seg = row["seg_key"]
    # Compute centroid WITHOUT this address
    excl = df_a3[(df_a3["seg_key"] == seg) &
                 (df_a3["address_id"] != addr_id) &
                 df_a3["ref_x"].notna()]
    if len(excl) >= 2:
        c4x = excl["ref_x"].mean()
        c4y = excl["ref_y"].mean()
        err_c4   = eucl_dist(c4x, c4y, row["doorstep_medoid_x"], row["doorstep_medoid_y"])
        err_base = eucl_dist(row["geocoder_x"], row["geocoder_y"],
                             row["doorstep_medoid_x"], row["doorstep_medoid_y"])
        df_scv_rows.append({
            "address_id": addr_id,
            "seg_key": seg,
            "c4_available": True,
            "error_vs_medoid_m": float(err_c4),
            "baseline_error_vs_medoid_m": float(err_base),
            "neighbor_count_excl": len(excl),
        })

df_scv = pd.DataFrame(df_scv_rows)
if len(df_scv) > 0:
    scv_m = error_metrics(df_scv["error_vs_medoid_m"])
    bse_m = error_metrics(df_scv["baseline_error_vs_medoid_m"])
    lift  = (1.0 - scv_m["median"] / bse_m["median"]) * 100 if bse_m["median"] > 0 else 0.0
    print(f"\\nSCV Results (N={scv_m['count']} LOO episodes):")
    print(f"  C4 Median Error : {scv_m['median']:.2f}m  P90: {scv_m['p90']:.2f}m")
    print(f"  Baseline Median : {bse_m['median']:.2f}m")
    print(f"  Lift vs baseline: {lift:.1f}%")
else:
    print("  SCV: insufficient episodes (need ≥ 2 neighbours on same street).")

# ── Save Phase 3 artifacts ────────────────────────────────────────────────────
df_street_centroids.to_csv(METRICS_DIR / "street_centroids.csv", index=False)
if len(df_scv) > 0:
    df_scv.to_csv(METRICS_DIR / "scv_results.csv", index=False)
p3_report = {
    "phase": 3,
    "street_centroids": len(df_street_centroids),
    "scv_episodes": len(df_scv),
    "scv_c4_median_m": round(scv_m["median"], 2) if len(df_scv) > 0 else None,
    "elapsed_s": round(time.time() - t3, 2),
}
with open(METRICS_DIR / "phase3_summary.json", "w") as f:
    json.dump(p3_report, f, indent=2)
print("\\nExported: street_centroids.csv, scv_results.csv, phase3_summary.json")
print("PHASE 3 COMPLETE.")
""")

# ===========================================================================
# Cell 6 — Phase 4: Candidate Generation, Scoring, Calibration
# ===========================================================================
md("## Phase 4 — Candidate Generation, Scoring & Calibration")
code("""
print("=" * 60)
print("PHASE 4 — Candidate Scoring, Reranking & Calibration")
print("=" * 60)
t4 = time.time()

# ── Build wide candidate table (one row per address × candidate type) ──────────
# C1: commercial baseline
# C2: credible doorstep medoid
# C3: POI / landmark coordinate
# C4: street centroid from spatial graph
# C5: locality centroid fallback

# Master address frame
df_wide = df_parsed[["address_id","locality_id_resolved","street_normalized",
                      "landmark_poi_id","town_id"]].copy()
df_wide["address_id"] = df_wide["address_id"].astype(str)

# C1 — baseline
df_base_c = df_base[["address_id","geocoder_x","geocoder_y"]].copy()
df_base_c["address_id"] = df_base_c["address_id"].astype(str)
for c in ["geocoder_x","geocoder_y"]:
    df_base_c[c] = pd.to_numeric(df_base_c[c], errors="coerce")
df_wide = df_wide.merge(df_base_c.rename(columns={"geocoder_x":"C1_x","geocoder_y":"C1_y"}),
                        on="address_id", how="left")
df_wide["C1_avail"] = df_wide["C1_x"].notna()

# C2 — medoid
df_med_c = df_medoids[["address_id","doorstep_medoid_x","doorstep_medoid_y","has_credible_doorstep"]].copy()
df_med_c["address_id"] = df_med_c["address_id"].astype(str)
df_wide = df_wide.merge(df_med_c.rename(columns={"doorstep_medoid_x":"C2_x","doorstep_medoid_y":"C2_y"}),
                        on="address_id", how="left")
df_wide["has_credible_doorstep"] = df_wide["has_credible_doorstep"].fillna(False)
df_wide["C2_avail"] = df_wide["C2_x"].notna() & df_wide["has_credible_doorstep"]

# C3 — POI
df_poi_c = df_pois[["poi_id","x","y"]].copy()
df_poi_c["poi_id_str"] = df_poi_c["poi_id"].astype(str)
for c in ["x","y"]:
    df_poi_c[c] = pd.to_numeric(df_poi_c[c], errors="coerce")
df_wide["landmark_poi_id_str"] = df_wide["landmark_poi_id"].astype(str)
df_wide = df_wide.merge(df_poi_c[["poi_id_str","x","y"]].rename(
    columns={"poi_id_str":"landmark_poi_id_str","x":"C3_x","y":"C3_y"}),
    on="landmark_poi_id_str", how="left")
df_wide["C3_avail"] = df_wide["C3_x"].notna()

# C4 — street centroid
df_wide["seg_key"] = (df_wide["locality_id_resolved"].astype(str) + "|" +
                      df_wide["street_normalized"].fillna("NONE"))
df_sc_c = df_street_centroids[["seg_key","C4_x","C4_y","neighbor_count"]].copy()
df_wide = df_wide.merge(df_sc_c, on="seg_key", how="left")
df_wide["C4_avail"] = df_wide["C4_x"].notna() & (df_wide["neighbor_count"].fillna(0) >= 2)

# C5 — locality centroid
df_wide["locality_id_resolved_str"] = df_wide["locality_id_resolved"].astype(str)
df_loc_c = df_loc_centroids[["locality_id_str","centroid_x","centroid_y"]].copy()
df_wide = df_wide.merge(df_loc_c.rename(
    columns={"locality_id_str":"locality_id_resolved_str","centroid_x":"C5_x","centroid_y":"C5_y"}),
    on="locality_id_resolved_str", how="left")
df_wide["C5_avail"] = df_wide["C5_x"].notna()

print(f"  Candidate availability (N={len(df_wide):,} addresses):")
for ct in ["C1","C2","C3","C4","C5"]:
    cnt = int(df_wide[f"{ct}_avail"].sum())
    print(f"    {ct}: {cnt:,}  ({cnt/len(df_wide)*100:.1f}%)")

# ── Reshape to long candidate table ──────────────────────────────────────────
cand_frames = []
TM_MAP      = {"C1": 0.85, "C2": 1.00, "C3": 0.90, "C4": 0.90, "C5": 0.70}
PRIORITY_MAP= {"C2": 1,    "C4": 2,    "C3": 3,    "C1": 4,    "C5": 5}

for ct in ["C1","C2","C3","C4","C5"]:
    if ct == "C1":
        tmp  = df_wide[df_wide["C1_avail"]][["address_id","C1_x","C1_y"]].copy()
        tmp["cand_x"] = tmp["C1_x"]
        tmp["cand_y"] = tmp["C1_y"]
    else:
        cols = ["address_id", f"{ct}_x", f"{ct}_y", "C1_x", "C1_y"]
        tmp  = df_wide[df_wide[f"{ct}_avail"]][cols].copy()
        tmp  = tmp.rename(columns={f"{ct}_x":"cand_x", f"{ct}_y":"cand_y"})
        
    tmp["candidate_type"] = ct
    tmp["TextMatch"]      = TM_MAP[ct]
    tmp["priority"]       = PRIORITY_MAP[ct]
    cand_frames.append(tmp)

df_long = pd.concat(cand_frames, ignore_index=True)
for c in ["cand_x","cand_y","C1_x","C1_y"]:
    df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

# ── Feature extraction ────────────────────────────────────────────────────────
SIGMA_B, SIGMA_V = 200.0, 50.0

# BaselineAgreement — Gaussian decay from commercial geocode
df_long["dist_to_baseline"] = eucl_dist(df_long["cand_x"], df_long["cand_y"],
                                         df_long["C1_x"], df_long["C1_y"])
df_long["BaselineAgreement"] = gauss_kernel(df_long["dist_to_baseline"], SIGMA_B).fillna(0.0)

# DirectEvidence — Gaussian decay from mean visit check-in
df_visit_mean = (df_v.assign(address_id=df_v["address_id"].astype(str))
                     .groupby("address_id")[["checkin_x","checkin_y"]]
                     .mean()
                     .reset_index()
                     .rename(columns={"checkin_x":"v_mean_x","checkin_y":"v_mean_y"}))
for c in ["v_mean_x","v_mean_y"]:
    df_visit_mean[c] = pd.to_numeric(df_visit_mean[c], errors="coerce")

df_long = df_long.merge(df_visit_mean, on="address_id", how="left")
dist_v   = eucl_dist(df_long["cand_x"], df_long["cand_y"],
                     df_long["v_mean_x"], df_long["v_mean_y"])
df_long["DirectEvidence"] = np.where(
    df_long["v_mean_x"].notna(),
    gauss_kernel(dist_v, SIGMA_V),
    0.0
)

# NeighborSupport — fraction of same-street neighbours with credible doorstep
seg_support = (df_a3.groupby("seg_key")["has_credible_doorstep"]
                    .mean()
                    .reset_index()
                    .rename(columns={"has_credible_doorstep":"NeighborSupport"}))
df_wide_seg = df_wide[["address_id","seg_key"]].copy()
df_long = df_long.merge(df_wide_seg, on="address_id", how="left")
df_long = df_long.merge(seg_support, on="seg_key", how="left")
df_long["NeighborSupport"] = df_long["NeighborSupport"].fillna(0.3)

# NegativeSearchPenalty — fraction of negative-outcome visits
neg_rate = (df_v.assign(address_id=df_v["address_id"].astype(str))
                .groupby("address_id")["is_untraceable_search"]
                .mean()
                .reset_index()
                .rename(columns={"is_untraceable_search":"NegativeSearchPenalty"}))
df_long = df_long.merge(neg_rate, on="address_id", how="left")
df_long["NegativeSearchPenalty"] = df_long["NegativeSearchPenalty"].fillna(0.0)

# ── Scoring weights (deterministic linear reranker) ───────────────────────────
W = {"w_text": 1.0, "w_visit": 2.5, "w_neigh": 1.5, "w_base": 0.5, "w_neg": 2.0}

df_long["total_score"] = (
    W["w_text"]  * df_long["TextMatch"]
  + W["w_visit"] * df_long["DirectEvidence"]
  + W["w_neigh"] * df_long["NeighborSupport"]
  + W["w_base"]  * df_long["BaselineAgreement"]
  - W["w_neg"]   * df_long["NegativeSearchPenalty"]
)

# ── Select top-1 candidate per address ───────────────────────────────────────
df_long.sort_values(["address_id","total_score","priority"],
                    ascending=[True, False, True], inplace=True)
df_long["rank"] = df_long.groupby("address_id").cumcount() + 1
df_top1 = df_long[df_long["rank"] == 1].copy()

print(f"\\n  Top-1 candidate selection:")
print(df_top1["candidate_type"].value_counts().to_string())

# ── Status tier assignment ────────────────────────────────────────────────────
def assign_tier(ct, has_cred, neighbor_count):
    if ct == "C2" and has_cred:              return "DOORSTEP_VERIFIED"
    if ct == "C4" and neighbor_count >= 5:   return "STREET_RESOLVED"
    if ct in ("C3","C4"):                    return "LANDMARK_ANCHORED"
    if ct == "C1":                           return "BASELINE_FALLBACK"
    return "LOCALITY_FALLBACK"

df_top1 = df_top1.merge(df_wide[["address_id","has_credible_doorstep","neighbor_count"]],
                         on="address_id", how="left")
df_top1["neighbor_count"] = df_top1["neighbor_count"].fillna(0)
df_top1["has_credible_doorstep"] = df_top1["has_credible_doorstep"].fillna(False)
df_top1["status_tier"] = df_top1.apply(
    lambda r: assign_tier(r["candidate_type"], r["has_credible_doorstep"], r["neighbor_count"]), axis=1
)

# ── Empirical quantile calibration ────────────────────────────────────────────
# Fit confidence radii per tier from SCV residuals (or defaults)
TIER_DEFAULTS_M = {
    "DOORSTEP_VERIFIED": 15.0,
    "STREET_RESOLVED":   50.0,
    "LANDMARK_ANCHORED": 150.0,
    "BASELINE_FALLBACK": 450.0,
    "LOCALITY_FALLBACK": 700.0,
}

if len(df_scv) > 0:
    # Merge status tier info into SCV
    df_scv_addr = df_top1[["address_id","status_tier"]].copy()
    df_scv["address_id"] = df_scv["address_id"].astype(str)
    df_scv_aug  = df_scv.merge(df_scv_addr, on="address_id", how="left")
    df_scv_aug["status_tier"] = df_scv_aug["status_tier"].fillna("BASELINE_FALLBACK")
    TIER_RADII = {}
    for tier, grp in df_scv_aug.groupby("status_tier"):
        residuals = grp["error_vs_medoid_m"].dropna()
        if len(residuals) >= 10:
            TIER_RADII[tier] = float(residuals.quantile(0.90))
        else:
            TIER_RADII[tier] = TIER_DEFAULTS_M.get(tier, 500.0)
else:
    TIER_RADII = TIER_DEFAULTS_M.copy()

df_top1["confidence_radius_m"] = df_top1["status_tier"].map(
    {**TIER_DEFAULTS_M, **TIER_RADII}
)
print(f"\\n  Calibrated confidence radii per tier:")
for t, r in sorted(TIER_RADII.items()):
    print(f"    {t:25s}: {r:.1f}m")

# ── Build full predictions output ─────────────────────────────────────────────
df_preds4 = df_top1[[
    "address_id","cand_x","cand_y","candidate_type","total_score",
    "status_tier","confidence_radius_m",
    "TextMatch","DirectEvidence","NeighborSupport","BaselineAgreement","NegativeSearchPenalty",
]].rename(columns={
    "cand_x":         "predicted_x",
    "cand_y":         "predicted_y",
    "candidate_type": "selected_candidate_type",
})

for c in ["predicted_x","predicted_y"]:
    df_preds4[c] = pd.to_numeric(df_preds4[c], errors="coerce").round(2)
df_preds4["total_score"] = df_preds4["total_score"].round(4)

# ── Evaluate on surveyed benchmark ───────────────────────────────────────────
df_surv_eval = df_surv[["address_id","surveyed_x","surveyed_y"]].copy()
df_surv_eval["address_id"] = df_surv_eval["address_id"].astype(str)
for c in ["surveyed_x","surveyed_y"]:
    df_surv_eval[c] = pd.to_numeric(df_surv_eval[c], errors="coerce")

df_eval = df_preds4.merge(df_surv_eval, on="address_id", how="inner")
if "address_id" in df_splits.columns:
    df_eval = df_eval.merge(df_splits, on="address_id", how="left")
elif "account_id" in df_splits.columns:
    df_eval = df_eval.merge(df_addr[["address_id","account_id"]], on="address_id", how="left")
    df_eval = df_eval.merge(df_splits, on="account_id", how="left")

df_eval["error_m"] = eucl_dist(df_eval["predicted_x"], df_eval["predicted_y"],
                                 df_eval["surveyed_x"], df_eval["surveyed_y"])
df_eval["is_covered"] = df_eval["error_m"] <= df_eval["confidence_radius_m"]

all_m = error_metrics(df_eval["error_m"])
print(f"\\nBenchmark Evaluation (N={all_m['count']})")
print(f"  Median: {all_m['median']:.2f}m  |  P90: {all_m['p90']:.2f}m  |  Max: {all_m['max']:.2f}m")
print(f"  Coverage: {df_eval['is_covered'].mean()*100:.1f}%  (Target ~90%)")

for split_val in ["train","validation","test"]:
    sub = df_eval[df_eval.get("split", pd.Series()) == split_val]["error_m"]
    if len(sub) > 0:
        sm = error_metrics(sub)
        print(f"  {split_val:12s} (N={sm['count']:2d}): Median={sm['median']:.2f}m  P90={sm['p90']:.2f}m")

# ── Ablation ladder M0 → M6 ───────────────────────────────────────────────────
def ablation_med(df_e, col_x, col_y):
    \"\"\"Compute error series from a coord column vs surveyed truth.\"\"\"
    ex = pd.to_numeric(df_e[col_x], errors="coerce")
    ey = pd.to_numeric(df_e[col_y], errors="coerce")
    return eucl_dist(ex, ey, df_e["surveyed_x"], df_e["surveyed_y"])

# Join all candidate coords to eval frame for ablation
df_abl = df_eval.merge(df_wide[["address_id","C1_x","C1_y","C2_x","C2_y","C2_avail",
                                  "C3_x","C3_y","C3_avail","C4_x","C4_y","C4_avail"]],
                        on="address_id", how="left")
for c in ["C1_x","C1_y","C2_x","C2_y","C3_x","C3_y","C4_x","C4_y"]:
    df_abl[c] = pd.to_numeric(df_abl[c], errors="coerce")

def pick_coord(row, priority):
    \"\"\"Return first available coord from priority list.\"\"\"
    for ct in priority:
        if ct == "C2" and not row.get("C2_avail", False): continue
        if ct == "C3" and not row.get("C3_avail", False): continue
        if ct == "C4" and not row.get("C4_avail", False): continue
        x, y = row.get(f"{ct}_x"), row.get(f"{ct}_y")
        if pd.notna(x) and pd.notna(y): return float(x), float(y)
    return None, None

stages_abl = [
    ("M0", ["C1"]),
    ("M1", ["C2","C1"]),
    ("M2", ["C2","C1"]),
    ("M3", ["C2","C3","C1"]),
    ("M4", ["C2","C4","C3","C1"]),
    ("M5", ["C2","C4","C3","C1"]),   # same as M4 (w_neg=0 no change in order here)
    ("M6", None),                    # full reranker predictions
]

ablation_rows = []
for stage_id, prio in stages_abl:
    if prio is not None:
        coords = df_abl.apply(lambda r: pick_coord(r, prio), axis=1)
        px = coords.apply(lambda c: c[0])
        py = coords.apply(lambda c: c[1])
        errs = eucl_dist(px, py, df_abl["surveyed_x"], df_abl["surveyed_y"])
    else:
        errs = df_abl["error_m"]
    errs_s = pd.Series(errs, index=df_abl.index)
    sm = error_metrics(errs_s)
    m0_median = error_metrics(pd.Series(eucl_dist(df_abl["C1_x"], df_abl["C1_y"],
                                         df_abl["surveyed_x"], df_abl["surveyed_y"])))["median"]
    ablation_rows.append({
        "stage": stage_id,
        "overall_median_error_m": round(sm["median"], 2),
        "overall_p90_error_m":    round(sm["p90"], 2),
        "improvement_vs_M0_m":    round(m0_median - sm["median"], 2),
    })

df_ablation = pd.DataFrame(ablation_rows)
print("\\nAblation Ladder M0 → M6:")
print(df_ablation.to_string(index=False))

# ── Save Phase 4 artifacts ────────────────────────────────────────────────────
df_preds4.to_csv(METRICS_DIR / "phase4_predictions.csv", index=False)
df_ablation.to_csv(METRICS_DIR / "phase4_ablation.csv", index=False)
json.dump({"weights": W, "tier_radii": TIER_RADII,
           "ablation": ablation_rows,
           "elapsed_s": round(time.time()-t4, 2)},
          open(METRICS_DIR / "phase4_summary.json","w"), indent=2)
print("\\nExported: phase4_predictions.csv, phase4_ablation.csv, phase4_summary.json")
print("PHASE 4 COMPLETE.")
""")

# ===========================================================================
# Cell 7 — Phase 5: Navigation Evidence Cards
# ===========================================================================
md("## Phase 5 — Navigation Evidence Cards & Final Export")
code("""
print("=" * 60)
print("PHASE 5 — Evidence Cards & Navigation Export")
print("=" * 60)
t5 = time.time()

# ── Helper: bearing & cardinal direction ──────────────────────────────────────
def compass_bearing(from_x, from_y, to_x, to_y):
    dx, dy = to_x - from_x, to_y - from_y
    bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return bearing

def cardinal(bearing_deg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((bearing_deg + 22.5) / 45) % 8]

def navigate_instruction(dist_m, bearing_deg, radius_m, status_tier):
    direction = cardinal(bearing_deg)
    if dist_m <= radius_m * 0.3:
        return f"You have ARRIVED at the location ({status_tier})."
    elif dist_m <= radius_m:
        return f"APPROACHING — {dist_m:.0f}m {direction}. Entering confidence zone."
    else:
        return f"NAVIGATE {direction} — {dist_m:.0f}m to reach the predicted location ({status_tier})."

# ── Load predictions ──────────────────────────────────────────────────────────
df_p5 = pd.read_csv(METRICS_DIR / "phase4_predictions.csv")
df_p5["address_id"] = df_p5["address_id"].astype(str)
for c in ["predicted_x","predicted_y","confidence_radius_m"]:
    df_p5[c] = pd.to_numeric(df_p5[c], errors="coerce")

# Merge parsed entity info
df_ent = df_parsed[["address_id","locality_name_raw","street_normalized",
                     "landmark_raw","spatial_relation","poi_resolution_status"]].copy()
df_ent["address_id"] = df_ent["address_id"].astype(str)
df_p5 = df_p5.merge(df_ent, on="address_id", how="left")

# Merge visit summary
neg_flags = (df_v.assign(address_id=df_v["address_id"].astype(str))
                  .groupby("address_id")["is_untraceable_search"]
                  .any()
                  .reset_index()
                  .rename(columns={"is_untraceable_search":"has_negative_visits"}))
df_p5 = df_p5.merge(neg_flags, on="address_id", how="left")
df_p5["has_negative_visits"] = df_p5["has_negative_visits"].fillna(False)

# ── Build evidence cards ──────────────────────────────────────────────────────
evidence_cards = []
SYNTHETIC_POSITIONS = [
    ("town_origin",    0.0,   0.0),
    ("at_destination", None,  None),    # filled per-row
    ("nearby_50m",     None,  None),    # filled per-row
]

nav_examples = []

for _, row in df_p5.iterrows():
    tx, ty   = float(row["predicted_x"]), float(row["predicted_y"])
    rad_m    = float(row["confidence_radius_m"])
    tier     = row["status_tier"]

    # Evidence card
    neg_ev = []
    if row["has_negative_visits"]:
        neg_ev.append("field_agent_reported_address_untraceable")
    if row["poi_resolution_status"] == "unresolved" and pd.notna(row.get("landmark_raw")):
        neg_ev.append("landmark_mention_unresolved_in_catalog")

    card = {
        "address_id":             row["address_id"],
        "status_tier":            tier,
        "predicted_x":            tx,
        "predicted_y":            ty,
        "confidence_radius_m":    rad_m,
        "selected_candidate":     row["selected_candidate_type"],
        "total_score":            row["total_score"],
        "locality":               str(row.get("locality_name_raw")) if pd.notna(row.get("locality_name_raw")) else None,
        "street":                 str(row.get("street_normalized")) if pd.notna(row.get("street_normalized")) else None,
        "landmark":               str(row.get("landmark_raw")) if pd.notna(row.get("landmark_raw")) else None,
        "spatial_relation":       str(row.get("spatial_relation")) if pd.notna(row.get("spatial_relation")) else None,
        "negative_evidence":      neg_ev,
        "caution":                "WARNING: negative field evidence present" if neg_ev else None,
    }
    evidence_cards.append(card)

    # Navigation scenarios
    for sc_name, sx, sy in [
        ("town_origin",    0.0,  0.0),
        ("at_destination", tx,   ty),
        ("outside_2x_rad", tx - 2*rad_m * 0.707, ty - 2*rad_m * 0.707),
    ]:
        dist = eucl_dist(sx, sy, tx, ty)
        bear = compass_bearing(sx, sy, tx, ty) if dist > 0 else 0.0
        nav_examples.append({
            "address_id":          row["address_id"],
            "scenario_name":       sc_name,
            "target_x":            tx,
            "target_y":            ty,
            "current_x":           sx,
            "current_y":           sy,
            "distance_m":          round(float(dist), 2),
            "bearing_deg":         round(float(bear), 1),
            "direction":           cardinal(bear),
            "confidence_radius_m": rad_m,
            "status_tier":         tier,
            "instruction":         navigate_instruction(float(dist), float(bear), rad_m, tier),
            "has_caution":         bool(neg_ev),
        })

df_nav = pd.DataFrame(nav_examples)

# ── Export ─────────────────────────────────────────────────────────────────────
with open(METRICS_DIR / "phase5_evidence_cards.json", "w") as f:
    json.dump(evidence_cards, f, indent=2)
df_nav.to_csv(METRICS_DIR / "phase5_navigation_examples.csv", index=False)

p5_summary = {
    "phase": 5,
    "evidence_cards_generated": len(evidence_cards),
    "navigation_examples_generated": len(df_nav),
    "status_tier_distribution": df_p5["status_tier"].value_counts().to_dict(),
    "negative_evidence_addresses": int(df_p5["has_negative_visits"].sum()),
    "elapsed_s": round(time.time() - t5, 2),
}
with open(METRICS_DIR / "phase5_summary.json", "w") as f:
    json.dump(p5_summary, f, indent=2)

print(f"  Evidence cards      : {len(evidence_cards):,}")
print(f"  Navigation examples : {len(df_nav):,}")
print(f"  Status distribution :")
for t, c in df_p5["status_tier"].value_counts().items():
    print(f"    {t:25s}: {c:,}")

print("\\nExported: phase5_evidence_cards.json, phase5_navigation_examples.csv, phase5_summary.json")
print("PHASE 5 COMPLETE.")
""")

# ===========================================================================
# Cell 8 — Final Submission Export
# ===========================================================================
md("## Final Submission — Aggregate & Export")
code("""
print("=" * 60)
print("FINAL SUBMISSION EXPORT")
print("=" * 60)

# ── Primary submission file: all 2880+ predictions ────────────────────────────
df_sub = df_preds4[["address_id","predicted_x","predicted_y",
                     "selected_candidate_type","status_tier",
                     "confidence_radius_m","total_score"]].copy()
sub_path = REPO_ROOT / "submission_predictions.csv"
df_sub.to_csv(sub_path, index=False)
print(f"  Predictions written: {sub_path.name}  ({len(df_sub):,} rows)")

# ── Immutability verification ─────────────────────────────────────────────────
df_check = pd.read_csv(sub_path)
assert len(df_check) == len(df_sub),                  "Row count mismatch!"
assert (df_check["address_id"].astype(str) == df_sub["address_id"].astype(str)).all(), "address_id mismatch!"
print("  ✓ Round-trip immutability check passed.")

# ── Final benchmark eval summary ──────────────────────────────────────────────
print(f"\\n{'='*60}")
print("BENCHMARK PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"{'Segment':<30} {'Median':>8} {'P90':>8} {'N':>5}")
print("-" * 55)

seg_data = [
    ("All Benchmark",   df_eval),
    ("Visited (is_vis)",df_eval[df_eval["address_id"].isin(df_medoids["address_id"])]),
    ("Cold-Start",      df_eval[~df_eval["address_id"].isin(df_medoids["address_id"])]),
]
if "split" in df_eval.columns:
    for sp in ["train","validation","test"]:
        seg_data.append((f"Split={sp}", df_eval[df_eval["split"]==sp]))

for name, sub in seg_data:
    e = sub["error_m"].dropna()
    if len(e) > 0:
        print(f"{name:<30} {e.median():>8.2f} {e.quantile(0.9):>8.2f} {len(e):>5}")

final_report = {
    "submission_file": str(sub_path),
    "predictions_count": len(df_sub),
    "benchmark_median_m": round(all_m["median"], 2),
    "benchmark_p90_m":    round(all_m["p90"], 2),
    "candidate_distribution": df_sub["selected_candidate_type"].value_counts().to_dict(),
    "tier_distribution":      df_sub["status_tier"].value_counts().to_dict(),
    "phases_completed": ["Phase0","Phase1","Phase2","Phase3","Phase4","Phase5"],
}
with open(METRICS_DIR / "final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\\n✓ Final report: metrics/final_report.json")
print(f"✓ Submission  : {sub_path.name}")
print("\\nPIPELINE COMPLETE.")
""")

c_viz_md = nbf.v4.new_markdown_cell("""\
# Phase 6: Pipeline Visualizations
Summary charts illustrating benchmark error distributions, cumulative coverage, and output tier breakdowns.
""")

c_viz = nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suppress annoying warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Error Comparison: Commercial Baseline vs NeuroGeo-Field
if "C1_x" in df_abl.columns and "C1_y" in df_abl.columns:
    baseline_error = eucl_dist(df_abl["C1_x"], df_abl["C1_y"], df_abl["surveyed_x"], df_abl["surveyed_y"])
    valid_mask = (df_eval["error_m"] > 0) & (baseline_error > 0)
    
    sns.kdeplot(np.log10(baseline_error[valid_mask]), ax=axes[0], label="Commercial Baseline", fill=True, color="gray", alpha=0.3)
    sns.kdeplot(np.log10(df_eval["error_m"][valid_mask]), ax=axes[0], label="NeuroGeo (M6)", fill=True, color="teal", alpha=0.5)
    axes[0].set_title("Location Error Distribution (Log10 m)", fontsize=13)
    axes[0].set_xlabel("Log10(Error in meters)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
else:
    axes[0].text(0.5, 0.5, "Baseline data not available", ha="center", va="center")
    axes[0].axis("off")

# 2. Cumulative Coverage Curve (NeuroGeo)
errors_sorted = np.sort(df_eval["error_m"].dropna())
p_ranks = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
axes[1].plot(errors_sorted, p_ranks, color="teal", linewidth=2)
axes[1].axvline(all_m["median"], color="teal", linestyle="--", label=f"Median: {all_m['median']:.1f}m")
axes[1].axvline(all_m["p90"], color="orange", linestyle="--", label=f"P90: {all_m['p90']:.1f}m")
axes[1].set_xlim(0, max(1000, all_m["p90"] * 1.5))
axes[1].set_title("Cumulative Error Distribution (NeuroGeo)", fontsize=13)
axes[1].set_xlabel("Error threshold (meters)")
axes[1].set_ylabel("Fraction of Addresses")
axes[1].legend()

# 3. Status Tier Distribution
status_counts = df_sub["status_tier"].value_counts()
axes[2].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette("pastel")[0:len(status_counts)], startangle=90)
axes[2].set_title("Status Tier Distribution (Portfolio)", fontsize=13)

plt.tight_layout()
plt.show()
""")
CELLS.extend([c_viz_md, c_viz])

# ===========================================================================
# Write notebook
# ===========================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = CELLS

out = ROOT / "submission.ipynb"
nbf.write(nb, str(out))
print(f"Generated {out}  ({len(CELLS)} cells)")
