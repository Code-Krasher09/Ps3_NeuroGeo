"""Inspect in-town addresses where locality name is not resolved.
"""

import sys
import pandas as pd
import re

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df_ent = pd.read_parquet("reports/metrics/address_entities.parquet")
df_addr = pd.read_csv("data/shared/addresses.csv")
df_loc = pd.read_csv("data/ps3_geocoder/localities.csv")

m = df_ent.merge(df_addr[["address_id", "address_text"]], on="address_id")
unres = m[(m["town_id"] != "OUT") & (m["locality_id_resolved"].isna())]
print(f"In-town unresolved locality count in df_ent: {len(unres)}")
print("\nSample in-town addresses with unresolved locality:")
for _, r in unres.head(25).iterrows():
    print(f"  {r['address_id']} ({r['town_id']}): {r['address_text']}")
