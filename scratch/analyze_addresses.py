"""Deep analysis of address_text for landmarks, localities, and PIN codes.
"""

import sys
from pathlib import Path
import re
import unicodedata
import pandas as pd
from typing import Dict, List, Set, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df_addr = pd.read_csv("data/shared/addresses.csv")
df_loc = pd.read_csv("data/ps3_geocoder/localities.csv")
df_pois = pd.read_csv("data/ps3_geocoder/landmarks_poi.csv")

print(f"Total addresses: {len(df_addr)}")
print(f"Town breakdown: {df_addr['town_id'].value_counts().to_dict()}")

# 1. Localities and Pincodes in localities.csv
valid_pins = set(df_loc["pincode"].unique())
print(f"Localities count: {len(df_loc)}")
print(f"Valid pincodes in localities.csv: {sorted(list(valid_pins))}")

# Map each pincode to its localities
pin_to_locs = {}
for _, row in df_loc.iterrows():
    pin = int(row["pincode"])
    pin_to_locs.setdefault(pin, []).append((row["locality_id"], row["locality_name"], row["town_id"]))

print("\nPincode to Locality Mapping in localities.csv:")
for pin, locs in sorted(pin_to_locs.items()):
    loc_names = [f"{lid}:{name}({tid})" for lid, name, tid in locs]
    print(f"  PIN {pin}: {', '.join(loc_names)}")

# Let's inspect town_id = OUT
out_addrs = df_addr[df_addr["town_id"] == "OUT"]
print(f"\nOUT addresses count: {len(out_addrs)}")
print("Sample OUT addresses:")
for _, r in out_addrs.head(5).iterrows():
    print(f"  {r['address_id']}: {r['address_text']}")
