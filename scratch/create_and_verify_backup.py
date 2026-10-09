import os
import zipfile
from pathlib import Path

repo_root = Path("d:/OpenIIT '26").resolve()
backup_zip_path = Path("d:/OpenIIT_26_backup_20261009_1525.zip").resolve()

print(f"Repo root: {repo_root}")
print(f"Backup target: {backup_zip_path}")
assert repo_root.exists(), f"Repo root does not exist: {repo_root}"

if backup_zip_path.exists():
    raise FileExistsError(f"Backup file {backup_zip_path} already exists!")

EXCLUDE_PARTS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    "ruff_cache",
}

def should_exclude(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in EXCLUDE_PARTS:
            return True
        if part.startswith(".venv"):
            return True
    if rel_path.name in {".DS_Store", "Thumbs.db"}:
        return True
    return False

all_files = []
for p in repo_root.rglob("*"):
    if p.is_file():
        rel = p.relative_to(repo_root)
        if not should_exclude(rel):
            all_files.append((p, rel))

print(f"Total files eligible for backup: {len(all_files)}")

with zipfile.ZipFile(backup_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for abs_p, rel_p in all_files:
        arcname = rel_p.as_posix()
        zf.write(abs_p, arcname=arcname)

print("Backup zip creation complete.")

# Verification Step 1: Check existence
assert backup_zip_path.exists(), "Backup file does not exist!"
zip_size_mb = backup_zip_path.stat().st_size / (1024 * 1024)
print(f"Backup zip size: {zip_size_mb:.2f} MB ({backup_zip_path.stat().st_size:,} bytes)")

# Verification Step 2: Integrity check
with zipfile.ZipFile(backup_zip_path, "r") as zf:
    bad_file = zf.testzip()
    if bad_file is not None:
        raise RuntimeError(f"Archive corruption detected at: {bad_file}")
    print("[PASS] Archive integrity test passed (testzip returned None, 0 corrupt files).")
    
    zip_namelist = set(zf.namelist())
    print(f"Total files archived in zip: {len(zip_namelist)}")

# Verification Step 3: Compare manifest
expected_set = {rel_p.as_posix() for _, rel_p in all_files}
missing = expected_set - zip_namelist
extra = zip_namelist - expected_set

print(f"Manifest check: Missing files = {len(missing)}, Extra files = {len(extra)}")
assert len(missing) == 0, f"Missing files in archive: {missing}"
assert len(extra) == 0, f"Extra files in archive: {extra}"
print("[PASS] Manifest comparison passed (100% exact match with expected project files).")

# Verification Step 4: Key files check
KEY_CHECKS = [
    "README_SUBMISSION.md",
    "PS3_FINAL_ARCHITECTURE.md",
    "Decisions.md",
    "summary_log.md",
    "PS3_DATA_CONTRACT.md",
    "PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md",
    "PS3_MODEL_SPEC.md",
    "PS3_IMPLEMENTATION_PLAN.md",
    "src/neurogeo/__init__.py",
    "src/neurogeo/candidate_ranker.py",
    "src/neurogeo/calibrator.py",
    "src/neurogeo/navigator.py",
    "notebooks/01_master_pipeline.ipynb",
    "notebooks/02_interactive_dashboard.ipynb",
    "tests/test_data_contract.py",
    "tests/test_candidate_ranker.py",
    "data/shared/addresses.csv",
    "data/shared/field_visits.csv",
    "data/ps3_geocoder/visit_gps_points.csv",
    "data/ps3_geocoder/baseline_geocodes.csv",
    "data/ps3_geocoder/landmarks_poi.csv",
    "data/ps3_geocoder/localities.csv",
    "reports/metrics/phase4_predictions.csv",
    "reports/metrics/scv_results.csv",
    "reports/metrics/calibration_quantiles.json",
    "reports/metrics/phase5_evidence_cards.json",
    "reports/figures/fig_a_baseline_vs_final.png",
    "reports/figures/fig_b_ablation_ladder.png",
    "reports/figures/fig_h_spatial_case_study.png",
]

for k in KEY_CHECKS:
    assert k in zip_namelist, f"Key file missing from archive: {k}"

print(f"[PASS] All {len(KEY_CHECKS)} sample critical key files and directories confirmed present.")

# Directory prefix coverage checks
prefixes = {
    "src/neurogeo/": 0,
    "tests/": 0,
    "notebooks/": 0,
    "data/": 0,
    "reports/metrics/": 0,
    "reports/figures/": 0,
    "scripts/": 0,
    "configs/": 0,
    "docs/": 0,
}

for name in zip_namelist:
    for pref in prefixes:
        if name.startswith(pref):
            prefixes[pref] += 1

print("\nDirectory breakdown in archive:")
for pref, count in prefixes.items():
    print(f"  * {pref}: {count} files")
