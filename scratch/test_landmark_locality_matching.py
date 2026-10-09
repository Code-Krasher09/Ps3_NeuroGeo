"""Analyze addresses.csv for landmarks, localities, and PIN codes.
"""

import sys
import re
import unicodedata
import pandas as pd
from collections import Counter

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df_addr = pd.read_csv("data/shared/addresses.csv")
df_loc = pd.read_csv("data/ps3_geocoder/localities.csv")
df_pois = pd.read_csv("data/ps3_geocoder/landmarks_poi.csv")

# 1. Catalog of Localities
# Normalized locality names and aliases
def normalize(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"['\"`]", "", s)
    s = re.sub(r"\bngrs?\b|\bnagr\b", "nagar", s)
    s = re.sub(r"\blyt\b|\blayt\b|\blayou\b", "layout", s)
    s = re.sub(r"\bcol\b|\bcolny\b|\bclny\b", "colony", s)
    s = re.sub(r"\bencl\b", "enclave", s)
    s = re.sub(r"\bbadvn\b", "badavane", s)
    s = re.sub(r"\bbaste\b", "basti", s)
    s = re.sub(r"\blakevew\b", "lakeview", s)
    s = re.sub(r"\bsilvur\b", "silver", s)
    s = re.sub(r"\banjeneya\b", "anjaneya", s)
    s = re.sub(r"\bneru\b", "nehru", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

loc_names_by_town = {}
for _, r in df_loc.iterrows():
    tid = r["town_id"]
    lname = r["locality_name"]
    norm_name = normalize(lname)
    pincode = int(r["pincode"])
    loc_names_by_town.setdefault(tid, []).append((r["locality_id"], lname, norm_name, pincode))

# Pincode map
pincode_to_localities = {}
for _, r in df_loc.iterrows():
    pin = int(r["pincode"])
    pincode_to_localities.setdefault(pin, []).append((r["locality_id"], r["locality_name"], r["town_id"]))

print("Localities by town:")
for tid, locs in loc_names_by_town.items():
    print(f"  {tid}: {len(locs)} localities")

# Check all pincodes appearing in df_addr
def get_pincode_from_address(txt):
    m = re.findall(r"\b(\d{6})\b", txt)
    if m:
        return int(m[-1]) # standard Indian address ends with PIN
    # check 7-digit typos
    m7 = re.findall(r"\b(\d{7})\b", txt)
    if m7:
        s = m7[-1]
        # Check if collapsing duplicate digit yields a valid pincode
        for pin in pincode_to_localities.keys():
            spin = str(pin)
            # check edit distance or substring
            if spin[:3] == s[:3] and spin[-2:] == s[-2:]:
                return pin
    return None

df_addr["extracted_pin"] = df_addr["address_text"].apply(get_pincode_from_address)
df_addr["has_valid_pin"] = df_addr["extracted_pin"].apply(lambda p: p in pincode_to_localities if p else False)

print("\nPincode extraction stats:")
print(f"Total with valid mapped pincode: {df_addr['has_valid_pin'].sum()} / {len(df_addr)}")
print(f"By town_id:")
print(df_addr.groupby("town_id")["has_valid_pin"].value_counts().to_string())
