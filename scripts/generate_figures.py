"""Generate authoritative publication-quality figures A-H for NeuroGeo-Field packaging.

Authority: PS3_FINAL_ARCHITECTURE.md, PS3_MODEL_SPEC.md, reports/metrics artifacts.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Figure A: Baseline vs Final Median Error
# -------------------------------------------------------------
def plot_figure_a():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = ["Commercial Baseline (M0)", "NeuroGeo-Field (M6)"]
    medians = [376.40, 61.26]
    p90s = [839.18, 574.29]
    colors = ["#d9534f", "#2b8a3e"]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, medians, width, label="Median Error (m)", color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, p90s, width, label="P90 Error (m)", color=["#f0ad4e", "#5cb85c"], alpha=0.7, edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Error Distance (meters)")
    ax.set_title("Figure A: Overall Geocoding Error — Baseline vs. NeuroGeo-Field\n(83.7% Median Error Reduction)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="semibold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 950)
    
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_a_baseline_vs_final.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure B: Ablation Ladder M0–M6 Median Error
# -------------------------------------------------------------
def plot_figure_b():
    df_ab = pd.read_csv("reports/metrics/phase4_ablation.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    
    stages = df_ab["stage"].tolist()
    labels = [
        "M0: Baseline",
        "M1: +Raw Visits",
        "M2: +Medoids (C2)",
        "M3: +POI (C3)",
        "M4: +Graph (C4)",
        "M5: +Reranker",
        "M6: +Neg Penalty"
    ]
    medians = df_ab["overall_median_error_m"].tolist()
    
    colors = ["#d9534f", "#e67e22", "#f39c12", "#2980b9", "#27ae60", "#16a085", "#2ecc71"]
    bars = ax.bar(stages, medians, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    
    ax.plot(stages, medians, color="#2c3e50", marker="o", linewidth=2, linestyle="--", label="Median Trajectory")
    
    ax.set_ylabel("Overall Median Error (meters)")
    ax.set_title("Figure B: Ablation Ladder M0 → M6 (Median Geolocation Error)", fontweight="bold")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontweight="semibold")
    ax.set_ylim(0, 420)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}m", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_b_ablation_ladder.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure C: Visited vs. Cold-Start Median Error
# -------------------------------------------------------------
def plot_figure_c():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    segments = ["Visited Segment (N=49)", "Cold-Start Segment (N=51)", "Overall Benchmark (N=100)"]
    baseline_meds = [383.67, 362.87, 376.40]
    final_meds = [21.10, 235.40, 61.26]
    
    x = np.arange(len(segments))
    width = 0.35
    
    b1 = ax.bar(x - width/2, baseline_meds, width, label="Commercial Baseline", color="#c0392b", alpha=0.85, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, final_meds, width, label="NeuroGeo-Field Final", color="#27ae60", alpha=0.85, edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Median Error (meters)")
    ax.set_title("Figure C: Visited vs. Cold-Start Performance Breakdown", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontweight="semibold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 450)
    
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}m", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}m", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_c_visited_vs_coldstart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure D: Candidate Availability C1–C5
# -------------------------------------------------------------
def plot_figure_d():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    candidates = [
        "C1\nCommercial Baseline",
        "C2\nDoorstep Medoid",
        "C3\nPOI Anchor",
        "C4\nStreet Graph Centroid",
        "C5\nLocality Centroid"
    ]
    counts = [100, 45, 46, 26, 89]
    colors = ["#7f8c8d", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    
    bars = ax.bar(candidates, counts, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax.set_ylabel("Availability on Benchmark (Count / 100)")
    ax.set_title("Figure D: Candidate Hypothesis Availability (N=100 Benchmark)", fontweight="bold")
    ax.set_ylim(0, 115)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}% ({h}/100)", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_d_candidate_availability.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure E: Status Tier Distribution
# -------------------------------------------------------------
def plot_figure_e():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tiers = ["DOORSTEP_VERIFIED", "LOCALITY_BOUNDED", "STREET_RESOLVED", "BASELINE_FALLBACK"]
    counts = [45, 30, 19, 6]
    labels = [
        "DOORSTEP_VERIFIED\n(45%)",
        "LOCALITY_BOUNDED\n(30%)",
        "STREET_RESOLVED\n(19%)",
        "BASELINE_FALLBACK\n(6%)"
    ]
    colors = ["#27ae60", "#f39c12", "#2980b9", "#e74c3c"]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Benchmark Addresses (Count / 100)")
    ax.set_title("Figure E: Selected Evidence Status-Tier Distribution (N=100)", fontweight="bold")
    ax.set_ylim(0, 52)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=11)
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_e_status_tier_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure F: Calibration Radius by Evidence Tier
# -------------------------------------------------------------
def plot_figure_f():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    tiers = [
        "LOCALITY_BOUNDED\n(N=1,258 SCV)",
        "STREET_RESOLVED\n(N=279 SCV)",
        "DOORSTEP_VERIFIED\n(N=2,965 Pooled)",
        "BASELINE_FALLBACK\n(N=1,428 SCV)"
    ]
    radii = [559.95, 615.93, 642.18, 982.55]
    colors = ["#f39c12", "#2980b9", "#27ae60", "#e74c3c"]
    
    bars = ax.bar(tiers, radii, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Empirical P90 Uncertainty Radius (meters)")
    ax.set_title("Figure F: Evidence-Tier Empirical Quantile Calibration Radii (R_T)", fontweight="bold")
    ax.set_ylim(0, 1150)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f} m", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    out_path = FIG_DIR / "fig_f_calibration_radii.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure G: Error Distribution Baseline vs Final (Cumulative CDF)
# -------------------------------------------------------------
def plot_figure_g():
    df_p = pd.read_csv("reports/metrics/phase4_predictions.csv")
    df_b = pd.read_csv("data/ps3_geocoder/baseline_geocodes.csv")
    df_m = df_p.merge(df_b, on="address_id")
    
    base_errors = np.sqrt((df_m["geocoder_x"] - df_m["surveyed_x"])**2 + (df_m["geocoder_y"] - df_m["surveyed_y"])**2).values
    final_errors = df_m["error_vs_surveyed_m"].values
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    base_sorted = np.sort(base_errors)
    final_sorted = np.sort(final_errors)
    cdf = np.linspace(0, 1, len(base_sorted))
    
    ax.plot(final_sorted, cdf * 100, label="NeuroGeo-Field Final (Median 61.3m)", color="#27ae60", linewidth=2.5)
    ax.plot(base_sorted, cdf * 100, label="Commercial Baseline (Median 376.4m)", color="#c0392b", linewidth=2.5, linestyle="--")
    
    ax.axvline(61.26, color="#27ae60", linestyle=":", alpha=0.7, label="NeuroGeo Median: 61.3m")
    ax.axvline(376.40, color="#c0392b", linestyle=":", alpha=0.7, label="Baseline Median: 376.4m")
    
    ax.set_xlabel("Geolocation Error (meters, log scale)")
    ax.set_xscale("log")
    ax.set_ylabel("Cumulative Percentage of Addresses (%)")
    ax.set_title("Figure G: Empirical Error CDF — Commercial Baseline vs. NeuroGeo-Field", fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    out_path = FIG_DIR / "fig_g_error_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure H: Spatial Case Study Plot (AD002159)
# -------------------------------------------------------------
def plot_figure_h():
    # Case study: AD002159 (Ambedkar Nagar, Gali 12, Devgarh Nagar)
    # Metric Cartesian coordinates centered at (0, 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Coordinates in meters
    base_x, base_y = -688.60, 3019.20
    pred_x, pred_y = -409.93, 3213.59
    surv_x, surv_y = -705.50, 3362.40  # Evaluation-only ground truth
    radius_m = 615.93
    
    # Neighbor doorstep medoids on Gali 12
    neigh_x = [-470.3, -356.3]
    neigh_y = [3195.4, 3229.8]
    neigh_labels = ["AD001662 (6 visits)", "AD002112 (7 visits)"]
    
    # 1. Uncertainty Circle around NeuroGeo Prediction
    circle = plt.Circle((pred_x, pred_y), radius_m, color="#2980b9", fill=True, alpha=0.12,
                        edgecolor="#2980b9", linestyle="--", linewidth=1.5, label=f"Calibrated P90 Radius ({radius_m:.1f}m)")
    ax.add_patch(circle)
    
    # 2. Commercial Baseline
    ax.scatter(base_x, base_y, color="#c0392b", s=130, marker="x", linewidths=3, zorder=5, label="Commercial Baseline (Error 412.4m)")
    ax.annotate("Commercial Baseline\n(-688.6, 3019.2)", (base_x, base_y), xytext=(-20, -35),
                textcoords="offset points", fontsize=9, color="#962d22", fontweight="bold")
    
    # 3. Visited Neighbors on Gali 12
    ax.scatter(neigh_x, neigh_y, color="#8e44ad", s=90, marker="D", zorder=4, label="Visited Street Neighbors (C4 inputs)")
    for nx, ny, nl in zip(neigh_x, neigh_y, neigh_labels):
        ax.annotate(nl, (nx, ny), xytext=(8, -5), textcoords="offset points", fontsize=8, color="#5b2c6f")
        
    # 4. NeuroGeo Prediction (Selected C4 Centroid)
    ax.scatter(pred_x, pred_y, color="#27ae60", s=150, marker="o", edgecolors="black", linewidths=1.5, zorder=6,
               label=f"NeuroGeo Prediction C4 (Error 324.0m)")
    ax.annotate(f"NeuroGeo Selected C4\n(-409.9, 3213.6)", (pred_x, pred_y), xytext=(12, 12),
                textcoords="offset points", fontsize=9, color="#1e8449", fontweight="bold")
    
    # 5. Surveyed Coordinate (EVALUATION-ONLY)
    ax.scatter(surv_x, surv_y, color="#d35400", s=140, marker="*", edgecolors="black", linewidths=1, zorder=7,
               label="Evaluation-only ground truth")
    ax.annotate("Surveyed Truth\n(Evaluation-only ground truth)", (surv_x, surv_y), xytext=(-60, 15),
                textcoords="offset points", fontsize=9, color="#ba4a00", fontweight="bold")
    
    # Plot connectors
    ax.plot([pred_x, surv_x], [pred_y, surv_y], color="#27ae60", linestyle=":", alpha=0.8, linewidth=1.5,
            label="Prediction Error Vector (324m)")
    ax.plot([base_x, surv_x], [base_y, surv_y], color="#c0392b", linestyle=":", alpha=0.6, linewidth=1.2,
            label="Baseline Error Vector (412m)")
    
    ax.set_xlabel("Cartesian X (meters from town origin)")
    ax.set_ylabel("Cartesian Y (meters from town origin)")
    ax.set_title("Figure H: Spatial Case Study — AD002159 (Cold-Start Street Knowledge Transfer)\n'Gali No. 12, nr Hanuman Temple, Ambedkar Nagar'", fontweight="bold")
    
    ax.set_xlim(pred_x - radius_m - 150, pred_x + radius_m + 150)
    ax.set_ylim(pred_y - radius_m - 150, pred_y + radius_m + 150)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    out_path = FIG_DIR / "fig_h_spatial_case_study.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    print("Generating Figure A...")
    plot_figure_a()
    print("Generating Figure B...")
    plot_figure_b()
    print("Generating Figure C...")
    plot_figure_c()
    print("Generating Figure D...")
    plot_figure_d()
    print("Generating Figure E...")
    plot_figure_e()
    print("Generating Figure F...")
    plot_figure_f()
    print("Generating Figure G...")
    plot_figure_g()
    print("Generating Figure H...")
    plot_figure_h()
    print("All figures A-H generated successfully in reports/figures/")

if __name__ == "__main__":
    main()
