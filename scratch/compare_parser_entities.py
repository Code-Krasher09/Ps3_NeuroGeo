"""Compare address_entities.parquet against regex analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import json

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df_ent = pd.read_parquet("reports/metrics/address_entities.parquet")
df_addr = pd.read_csv("data/shared/addresses.csv")
df_loc = pd.read_csv("data/ps3_geocoder/localities.csv")
pincodes_mapped = set(df_loc["pincode"].unique())

from scratch.run_full_analysis import match_mapped_pincode

# Check how df_ent defines has_landmark and has_locality
df_ent["extracted_pin"] = df_addr["address_text"].apply(match_mapped_pincode)
df_ent["has_mapped_pin"] = df_ent["extracted_pin"].notna()
df_ent["has_locality_resolved"] = df_ent["locality_id_resolved"].notna()
df_ent["has_landmark_extracted"] = df_ent["landmark_raw"].notna()

print("=" * 80)
print("DF_ENT (OFFICIAL PARSER ARTIFACT) BREAKDOWN")
print("=" * 80)
print("Locality resolved:", df_ent["has_locality_resolved"].sum())
print("Mapped PIN present:", df_ent["has_mapped_pin"].sum())
print("Locality Resolved OR Mapped PIN:", ((df_ent["has_locality_resolved"]) | (df_ent["has_mapped_pin"])).sum())
print("Landmark extracted:", df_ent["has_landmark_extracted"].sum())

print("\nCROSS TABULATION IN DF_ENT:")
print(pd.crosstab(
    df_ent["has_landmark_extracted"].rename("Has Landmark (df_ent)"),
    [df_ent["has_locality_resolved"].rename("Locality Resolved"), df_ent["has_mapped_pin"].rename("Mapped PIN")]
))
