"""Comprehensive analysis of landmarks, localities, and pincodes in addresses.csv
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

# 1. Normalization function
def normalize(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"['\"`]", "", s)
    # Common abbreviations
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
    s = re.sub(r"\brd\b", "road", s)
    s = re.sub(r"\bmn\b", "main", s)
    s = re.sub(r"\bcrs\b|\bcr\b", "cross", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

# 2. Locality catalog & matching
# We map all 36 localities in localities.csv
localities_list = []
for _, r in df_loc.iterrows():
    lid = r["locality_id"]
    tid = r["town_id"]
    name = r["locality_name"]
    norm_name = normalize(name)
    pin = int(r["pincode"])
    localities_list.append({
        "locality_id": lid,
        "town_id": tid,
        "name": name,
        "norm_name": norm_name,
        "pin": pin
    })

# Add known typo variants for fuzzy locality matching if needed
# E.g. "Idnira Col" -> Indira Colony, "Viya Nagar" -> Vidya Nagar
locality_patterns = [
    # T1
    ("T1-L01", "vinayaka nagar", r"\bvinayaka?\s*nagar\b"),
    ("T1-L02", "basava nagar", r"\bbasava?\s*nagar\b"),
    ("T1-L03", "kuvempu layout", r"\bkuvempu\s*layout\b"),
    ("T1-L04", "gandhi nagar", r"\bgandhi\s*nagar\b"),
    ("T1-L05", "ashoka layout", r"\bashoka?\s*layout\b"),
    ("T1-L06", "siddeshwara colony", r"\bsidd[eh]+shwara?\s*colony\b"),
    ("T1-L07", "anjaneya badavane", r"\b(anjaneya|anjeneya)\s*badavane\b"),
    ("T1-L08", "vidya nagar", r"\b(vidya|viya)\s*nagar\b"),
    ("T1-L09", "shanthi nagar", r"\bshanth?i\s*nagar\b"),
    ("T1-L10", "kalyana nagar", r"\bkalyana?\s*nagar\b"),
    ("T1-L11", "nehru colony", r"\b(nehru|neru)\s*colony\b"),
    ("T1-L12", "hosa badavane", r"\bhosa\s*badavane\b"),
    # T2
    ("T2-L01", "shivaji nagar", r"\bshivaji\s*nagar\b"),
    ("T2-L02", "indira colony", r"\b(indira|idnira)\s*colony\b"),
    ("T2-L03", "ambedkar nagar", r"\bambedkar\s*nagar\b"),
    ("T2-L04", "ram nagar", r"\bram\s*nagar\b"),
    ("T2-L05", "subhash nagar", r"\bsubhash\s*nagar\b"),
    ("T2-L06", "nehru colony", r"\b(nehru|neru)\s*colony\b"),
    ("T2-L07", "gandhi basti", r"\bgandhi\s*basti\b"),
    ("T2-L08", "azad mohalla", r"\bazad\s*mohalla\b"),
    ("T2-L09", "shastri nagar", r"\bshastri\s*nagar\b"),
    ("T2-L10", "krishna puri", r"\bkrishna\s*puri\b|\bkrishnapuri\b"),
    ("T2-L11", "patel nagar", r"\bpatel\s*nagar\b"),
    ("T2-L12", "tilak nagar", r"\btilak\s*nagar\b"),
    # T3
    ("T3-L01", "green park layout", r"\bgreen\s*park\s*layout\b"),
    ("T3-L02", "sunrise enclave", r"\bsunrise\s*enclave\b"),
    ("T3-L03", "hmt colony", r"\bhmt\s*colony\b"),
    ("T3-L04", "defence colony", r"\bdefence\s*colony\b"),
    ("T3-L05", "lakeview phase 2", r"\blakeview\s*(phase\s*2|ph\s*2)?\b"),
    ("T3-L06", "silver oak layout", r"\bsilver\s*oak\s*layout\b"),
    ("T3-L07", "railway colony", r"\brailway\s*colony\b"),
    ("T3-L08", "teachers colony", r"\bteachers\s*colony\b"),
    ("T3-L09", "royal gardens", r"\broyal\s*gardens?\b"),
    ("T3-L10", "palm meadows", r"\bpalm\s*meadows?\b"),
    ("T3-L11", "orchid enclave", r"\borchid\s*enclave\b"),
    ("T3-L12", "unity nagar", r"\bunity\s*nagar\b"),
]

# Check locality name presence
def match_locality_name(text, town_id):
    norm_text = normalize(text)
    # Search patterns
    for lid, lname, pat in locality_patterns:
        # If town_id is known and matches
        if town_id != "OUT" and not lid.startswith(town_id):
            continue
        if re.search(pat, norm_text, flags=re.IGNORECASE):
            return lid, lname
    return None, None

# 3. Pincode matching with localities.csv
pincodes_mapped = set(df_loc["pincode"].unique())

def match_mapped_pincode(text):
    # Match any 6 digit pin
    m = re.findall(r"\b(\d{6})\b", text)
    if m:
        for p in m:
            ip = int(p)
            if ip in pincodes_mapped:
                return ip
    # Check 7-digit typos
    m7 = re.findall(r"\b(\d{7})\b", text)
    if m7:
        for s in m7:
            for p in pincodes_mapped:
                sp = str(p)
                if sp[:3] == s[:3] and sp[-2:] == s[-2:]:
                    return p
    return None

# 4. Landmark patterns dictionary
LANDMARK_REGEXES = {
    "ration_shop": r"\b(ration\s*shop|ration\s*ki\s*duka+n|ration\s*dukan|rasan\s*dukan|nyaya\s*bele\s*angadi|nyayabele\s*angadi|pds\s*shop|fair\s*price\s*shop)\b|ನ್ಯಾಯಬೆಲೆ|ರಾಶನ್|राशन\s*(की\s*)?दुकान",
    "bus_stop": r"\b(bus\s*stop|bus\s*stand|bus\s*adda|bus\s*stp|busstop)\b|ಬಸ್\s*ನಿಲ್ದಾಣ|ಬಸ್|बस\s*(स्टॉप|अड्डा|स्टैंड)",
    "govt_school": r"\b(govt\s*school|government\s*school|sarkari\s*school|sarkari\s*sha+le|primary\s*school|high\s*school)\b|ಸರ್ಕಾರಿ\s*ಶಾಲೆ|ಶಾಲೆ|सरकारी\s*(स्कूल|विद्यालय)",
    "masjid": r"\b(masjid|mosque|jama\s*masjid|madina\s*masjid|noor\s*masjid)\b|ಮಸೀದಿ|मस्जिद",
    "hanuman_temple": r"\b(hanuman\s*temple|hanuman\s*mandir|anjaneya\s*temple|maruti\s*mandir|anjaneya\s*gudi|maruti\s*temple)\b|ಆಂಜನೇಯ\s*ದೇವಸ್ಥಾನ|ಹನುಮಾನ್\s*ದೇವಸ್ಥಾನ|हनुमान\s*मंदिर|मारुति\s*मंदिर",
    "medical_store": r"\b(medical\s*store|medical\s*shop|pharmacy|chemist|dawa\s*ki\s*duka+n|dawai\s*ki\s*duka+n|medicals|med\s*store|medical\s*shp)\b|ಮೆಡಿಕಲ್|मेडिकल\s*स्टोर|दवा(ई)?\s*की\s*दुकान",
    "ganesha_temple": r"\b(ganesh\s*temple|ganesha\s*temple|ganapathi\s*temple|ganapathi\s*gudi|ganapati\s*devasthana|vinayaka\s*temple|ganapathi\s*gu+di|ganesh\s*mandir|ganapathi\s*mandir)\b|ಗಣಪತಿ\s*ದೇವಸ್ಥಾನ|ವಿನಾಯಕ\s*ದೇವಸ್ಥಾನ|गणेश\s*मंदिर|गणपति\s*मंदिर",
    "water_tank": r"\b(water\s*tank|pa+ni\s*ki\s*tanki|neerina\s*tank|overhead\s*tank|water\s*tanki|tanki)\b|ನೀರಿನ\s*ಟ್ಯಾಂಕ್|ಟ್ಯಾಂಕ್|पानी\s*की\s*टंकी|पानी\s*टंकी",
    "milk_dairy": r"\b(milk\s*dairy|ha+lu\s*dairy|ha+lina\s*dairy|doodh\s*dairy|milk\s*booth|ha+lu\s*kendra)\b|ಹಾಲಿನ\s*ಡೈರಿ|ಹಾಲು\s*ಡೈರಿ|दूध\s*डेयरी|मिल्क\s*डेयरी",
    "community_hall": r"\b(community\s*hall|kalyana\s*mantap[a]?|kalyan\s*mantap|barat\s*gh+ar|samudaya\s*bhavana?|function\s*hall)\b|ಕಲ್ಯಾಣ\s*ಮಂಟಪ|ಸಮುದಾಯ\s*ಭವನ|बारात\s*घर|कल्याण\s*मंडप",
    "park": r"\b(park|public\s*park)\b|ಉದ್ಯಾನವನ|ಪಾರ್ಕ್|पार्क|बगीचा",  # note: avoid 'green park layout' collision by checking layout context
    "church": r"\b(church|st\s*marys\s*church)\b|ಚರ್ಚ್|चर्च",
    "post_office": r"\b(post\s*office|dak\s*ghar|anche\s*kacheri|post\s*off)\b|ಅಂಚೆ\s*ಕಚೇರಿ|ಅಂಚೆ|डाक\s*घर|डाकघर",
    "petrol_bunk": r"\b(petrol\s*bunk|petrol\s*pump|fuel\s*station)\b|ಪೆಟ್ರೋಲ್\s*ಬಂಕ್|ಪೆಟ್ರೋಲ್|पेट्रोल\s*पंप"
}

# Special care for 'park': in 'green park layout', 'park' is part of the locality name, NOT a landmark!
def match_landmarks(text):
    norm_text = normalize(text)
    # Check if 'green park layout' is present
    has_green_park_locality = "green park" in norm_text
    
    found_landmarks = []
    for lm_type, regex in LANDMARK_REGEXES.items():
        if lm_type == "park" and has_green_park_locality:
            # Check if there is another mention of park outside 'green park layout'
            text_without_gp = re.sub(r"green\s*park\s*(layout)?", "", norm_text)
            m = re.search(regex, text_without_gp, flags=re.IGNORECASE)
            if m:
                found_landmarks.append((lm_type, m.group(0)))
        else:
            # Check in normalized text and original text
            m = re.search(regex, norm_text, flags=re.IGNORECASE)
            if not m:
                m = re.search(regex, text)
            if m:
                found_landmarks.append((lm_type, m.group(0)))
    return found_landmarks

# Run analysis on all addresses
results = []
for _, r in df_addr.iterrows():
    aid = r["address_id"]
    tid = r["town_id"]
    txt = str(r["address_text"])
    
    lid, lname = match_locality_name(txt, tid)
    pin = match_mapped_pincode(txt)
    lms = match_landmarks(txt)
    
    results.append({
        "address_id": aid,
        "town_id": tid,
        "address_text": txt,
        "has_locality_name": lid is not None,
        "locality_id": lid,
        "locality_name": lname,
        "has_mapped_pin": pin is not None,
        "mapped_pin": pin,
        "landmark_count": len(lms),
        "landmarks": [lm[0] for lm in lms],
        "has_landmark": len(lms) > 0,
    })

res_df = pd.DataFrame(results)

print("=" * 70)
print("ADDRESS ANALYSIS RESULTS (Total = 3,117)")
print("=" * 70)
print(f"Total Addresses: {len(res_df)}")
print(f"Addresses with Locality Name mentioned: {res_df['has_locality_name'].sum()} ({res_df['has_locality_name'].mean()*100:.1f}%)")
print(f"Addresses with Mapped PIN code mentioned: {res_df['has_mapped_pin'].sum()} ({res_df['has_mapped_pin'].mean()*100:.1f}%)")
print(f"Addresses with EITHER Locality Name OR Mapped PIN: {((res_df['has_locality_name']) | (res_df['has_mapped_pin'])).sum()} ({((res_df['has_locality_name']) | (res_df['has_mapped_pin'])).mean()*100:.1f}%)")
print(f"Addresses with NEITHER Locality Name NOR Mapped PIN: {((~res_df['has_locality_name']) & (~res_df['has_mapped_pin'])).sum()} ({((~res_df['has_locality_name']) & (~res_df['has_mapped_pin'])).mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("LANDMARK MENTIONS DISTRIBUTION")
print("=" * 70)
print(res_df["landmark_count"].value_counts().sort_index())

print("\n" + "=" * 70)
print("CROSS-TABULATION: LANDMARK vs LOCALITY NAME vs PIN CODE")
print("=" * 70)
print(pd.crosstab(
    res_df["has_landmark"].rename("Has Landmark"),
    [res_df["has_locality_name"].rename("Locality Name"), res_df["has_mapped_pin"].rename("Mapped PIN")]
))
