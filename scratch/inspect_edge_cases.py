"""Examine details of landmark and locality categories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
from scratch.run_full_analysis import res_df

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("TOWN BREAKDOWN FOR EACH COMBINATION")
print("=" * 80)

for (has_lm, has_loc, has_pin), grp in res_df.groupby(["has_landmark", "has_locality_name", "has_mapped_pin"]):
    print(f"\nLandmark: {has_lm} | Locality Name: {has_loc} | Mapped PIN: {has_pin} (Total = {len(grp)})")
    print("  Town counts:", grp["town_id"].value_counts().to_dict())

# 1. Inspect the 18 addresses: Has Landmark=True, Locality Name=False, Mapped PIN=False
print("\n" + "=" * 80)
print("18 ADDRESSES: Landmark=True, Locality Name=False, Mapped PIN=False")
print("=" * 80)
g18 = res_df[(res_df["has_landmark"]) & (~res_df["has_locality_name"]) & (~res_df["has_mapped_pin"])]
for _, r in g18.iterrows():
    print(f"  {r['address_id']} ({r['town_id']}): {r['address_text']} [Landmarks: {r['landmarks']}]")

# 2. Inspect the 1 address with landmark_count == 2
print("\n" + "=" * 80)
print("ADDRESS WITH MULTIPLE LANDMARKS (landmark_count == 2)")
print("=" * 80)
g2 = res_df[res_df["landmark_count"] > 1]
for _, r in g2.iterrows():
    print(f"  {r['address_id']} ({r['town_id']}): {r['address_text']} [Landmarks: {r['landmarks']}]")

# 3. Inspect Case 2 (Landmark=True, Locality Name=False, Mapped PIN=True)
print("\n" + "=" * 80)
print("SAMPLE ADDRESSES: Landmark=True, Locality Name=False, Mapped PIN=True (Total = 224)")
print("=" * 80)
g224 = res_df[(res_df["has_landmark"]) & (~res_df["has_locality_name"]) & (res_df["has_mapped_pin"])]
for _, r in g224.head(10).iterrows():
    print(f"  {r['address_id']} ({r['town_id']}): {r['address_text']} [Landmarks: {r['landmarks']}, PIN: {r['mapped_pin']}]")
