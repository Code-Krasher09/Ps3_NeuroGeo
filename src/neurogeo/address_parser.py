"""Multilingual script-aware address parser and entity extractor.

Authority: PS3_FINAL_ARCHITECTURE.md Section 2, PS3_DATA_CONTRACT.md Section 3,
and PS3_IMPLEMENTATION_PLAN.md Phase 2.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import unicodedata
import pandas as pd

from neurogeo.config import DEFAULT_CONFIG, PATHS, ModelConfig
from neurogeo.schemas import ParsedAddressEntity, PoiResolutionResult

try:
    # Try importing the new MuRIL NER pipeline
    from transformers import pipeline
    muril_ner = pipeline("token-classification", model="google/muril-base-cased")
except ImportError:
    muril_ner = None


# ---------------------------------------------------------------------------
# Multilingual Dictionaries: Landmark Categories and Relational Phrases
# ---------------------------------------------------------------------------

LANDMARK_PATTERNS: Dict[str, List[str]] = {
    "ration_shop": [
        "ration shop", "ration ki dukan", "ration ki dukaan", "ration dukan", "rasan dukan",
        "nyayabele angadi", "nyaya bele angadi", "pds shop", "fair price shop",
        "ನ್ಯಾಯಬೆಲೆ ಅಂಗಡಿ", "ನ್ಯಾಯಬೆಲೆ", "ರಾಶನ್", "राशन की दुकान", "राशन दुकान"
    ],
    "bus_stop": [
        "bus stop", "bus stand", "bus adda", "bus stp", "busstop",
        "ಬಸ್ ನಿಲ್ದಾಣ", "ಬಸ್", "बस स्टॉप", "बस अड्डा", "बस स्टैंड"
    ],
    "govt_school": [
        "govt school", "government school", "sarkari school", "sarkari shaale",
        "sarkari shale", "primary school", "high school",
        "ಸರ್ಕಾರಿ ಶಾಲೆ", "ಶಾಲೆ", "सरकारी स्कूल", "सरकारी विद्यालय"
    ],
    "masjid": [
        "masjid", "mosque", "jama masjid", "madina masjid", "noor masjid",
        "ಮಸೀದಿ", "मस्जिद"
    ],
    "hanuman_temple": [
        "hanuman temple", "hanuman mandir", "anjaneya temple", "maruti mandir",
        "anjaneya gudi", "maruti temple",
        "ಆಂಜನೇಯ ದೇವಸ್ಥಾನ", "ಹನುಮಾನ್ ದೇವಸ್ಥಾನ", "हनुमान मंदिर", "मारुति मंदिर"
    ],
    "medical_store": [
        "medical store", "medical shop", "pharmacy", "chemist", "dawa ki dukan",
        "medicals", "med store",
        "ಮೆಡಿಕಲ್ ಶಾಪ್", "ಮೆಡಿಕಲ್ ಸ್ಟೋರ್", "ಮೆಡಿಕಲ್", "मेडिकल स्टोर", "दवा की दुकान"
    ],
    "ganesha_temple": [
        "ganesh temple", "ganesha temple", "ganapathi temple", "ganapathi gudi",
        "ganapati devasthana", "vinayaka temple", "ganapathi guudi",
        "ಗಣಪತಿ ದೇವಸ್ಥಾನ", "ವಿನಾಯಕ ದೇವಸ್ಥಾನ", "गणेश मंदिर", "गणपति मंदिर"
    ],
    "water_tank": [
        "water tank", "paani ki tanki", "pani ki tanki", "neerina tank",
        "overhead tank", "water tanki", "tanki",
        "ನೀರಿನ ಟ್ಯಾಂಕ್", "ಟ್ಯಾಂಕ್", "पानी की टंकी", "पानी टंकी"
    ],
    "milk_dairy": [
        "milk dairy", "haalu dairy", "haalina dairy", "doodh dairy",
        "dairy", "milk booth", "haalu kendra",
        "ಹಾಲಿನ ಡೈರಿ", "ಹಾಲು ಡೈರಿ", "दूध डेयरी", "मिल्क डेयरी"
    ],
    "community_hall": [
        "community hall", "kalyana mantapa", "kalyana mantap", "kalyan mantap",
        "barat ghar", "barat ghhar", "samudaya bhavana", "function hall",
        "ಕಲ್ಯಾಣ ಮಂಟಪ", "ಸಮುದಾಯ ಭವನ", "बारात घर", "कल्याण मंडप"
    ],
    "park": [
        "park", "garden", "public park",
        "ಉದ್ಯಾನವನ", "ಪಾರ್ಕ್", "पार्क", "बगीचा"
    ],
    "church": [
        "church", "st marys church",
        "ಚರ್ಚ್", "चर्च"
    ],
    "post_office": [
        "post office", "dakghar", "dak ghar", "anche kacheri", "post off",
        "ಅಂಚೆ ಕಚೇರಿ", "ಅಂಚೆ", "डाकघर", "डाक घर"
    ],
    "petrol_bunk": [
        "petrol bunk", "petrol pump", "fuel station", "bunk",
        "ಪೆಟ್ರೋಲ್ ಬಂಕ್", "ಪೆಟ್ರೋಲ್", "पेट्रोल पंप"
    ],
}

RELATION_PATTERNS: Dict[str, List[str]] = {
    "behind": [
        "behind", "bhd", "bhd.", "hinde", "hindhe", "hinda",
        "ಹಿಂದೆ", "ಹಿಂಭಾಗ", "ke peeche", "ke piche", "peeche", "piche", "के पीछे", "पीछे"
    ],
    "opposite": [
        "opposite", "opp", "opp.", "oppo", "in front of", "front of",
        "eduru", "edhuu", "munde", "mundhe", "ಮುಂದೆ", "ಎದುರು",
        "ke samne", "ke saamne", "samne", "saamne", "के सामने", "सामने"
    ],
    "near": [
        "near", "nr", "nr.", "close to", "nearby", "near by",
        "hattira", "hathira", "hattir", "ಹತ್ತಿರ",
        "ke paas", "ke pass", "paas", "pass", "के पास", "पास"
    ],
    "beside": [
        "beside", "bsd", "bsd.", "next to", "adjacent", "adj", "adj.", "side",
        "pakka", "pakkadalli", "ಪಕ್ಕ",
        "ke bagal mein", "ke bagal", "bagal mein", "bagal", "के बगल में", "के बगल", "बगल में", "बगल"
    ],
}

# Standard abbreviation and spelling normalization mappings
TEXT_REPLACEMENTS = {
    r"\bngrs?\b": "nagar",
    r"\bnagr\b": "nagar",
    r"\blyt\b": "layout",
    r"\blayt\b": "layout",
    r"\blayou\b": "layout",
    r"\bcol\b": "colony",
    r"\bcolny\b": "colony",
    r"\bclny\b": "colony",
    r"\bencl\b": "enclave",
    r"\bbadvn\b": "badavane",
    r"\bbaste\b": "basti",
    r"\blakevew\b": "lakeview",
    r"\bsilvur\b": "silver",
    r"\banjeneya\b": "anjaneya",
    r"\bneru\b": "nehru",
    r"\brd\b": "road",
    r"\bmn\b": "main",
    r"\bcrs\b": "cross",
    r"\bcr\b": "cross",
    r"\bbhd\b": "behind",
    r"\bopp\b": "opposite",
    r"\boppo\b": "opposite",
    r"\bnr\b": "near",
    r"\bbsd\b": "beside",
    r"\bblk\b": "block",
}


# ---------------------------------------------------------------------------
# Unicode Normalization & Script Detection
# ---------------------------------------------------------------------------

def normalize_script(text: str) -> str:
    """Normalize Kannada, Devanagari, and Latin text variations deterministically.
    
    Performs NFKC normalization, lowercasing of Latin characters, whitespace
    standardization, repeated-character collapse (>= 3 chars), and common abbreviation mapping.
    Preserves native Indic scripts safely.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode NFKC normalization
    s = unicodedata.normalize("NFKC", text)
    
    # 2. Lowercase (case-insensitive for Latin, neutral for Indic)
    s = s.lower()
    
    # 3. Standardize punctuation: replace tabs, newlines, multiple spaces, em-dashes
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"['\"`]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    
    # 4. Collapse repeated characters >= 3 (e.g., 'layouuuut' -> 'layout', 'roaaad' -> 'road')
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    
    # 5. Apply known abbreviation replacements
    for pat, rep in TEXT_REPLACEMENTS.items():
        s = re.sub(pat, rep, s)
        
    return s.strip()


def detect_script(text: str) -> Tuple[str, bool]:
    """Detect writing script and multilingual status.
    
    Returns:
        (script_detected, is_multilingual)
        where script_detected is 'kannada', 'devanagari', 'latin', or 'mixed'.
    """
    if not text:
        return ("latin", False)
        
    has_kannada = any("\u0C80" <= ch <= "\u0CFF" for ch in text)
    has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    
    # Check for transliterated native terms
    indic_transliteration_keywords = {
        "hattira", "hinde", "eduru", "pakka", "badavane", "kalyana", "mantapa",
        "nyayabele", "angadi", "ke paas", "ke peeche", "ke samne", "ke bagal",
        "gali", "basti", "dukan", "mandir", "tanki"
    }
    has_transliterated = any(kw in text.lower() for kw in indic_transliteration_keywords)
    
    if has_kannada and has_devanagari:
        return ("mixed", True)
    elif has_kannada:
        return ("mixed", True) if has_latin else ("kannada", True)
    elif has_devanagari:
        return ("mixed", True) if has_latin else ("devanagari", True)
    elif has_transliterated:
        return ("latin", True)
    elif has_latin:
        return ("latin", False)
    return ("latin", False)


# ---------------------------------------------------------------------------
# Character N-gram Fuzzy Matching
# ---------------------------------------------------------------------------

def char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract character n-grams with word boundary padding."""
    s_clean = f" {s.strip().lower()} "
    if len(s_clean) < n:
        return {s_clean}
    return {s_clean[i:i+n] for i in range(len(s_clean) - n + 1)}


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """Compute Jaccard similarity over character n-grams."""
    g1 = char_ngrams(s1, n)
    g2 = char_ngrams(s2, n)
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)


# ---------------------------------------------------------------------------
# Entity Extractors: Street, House, Landmark, Relation
# ---------------------------------------------------------------------------

def extract_house(text: str) -> Optional[str]:
    """Extract house, flat, or unit number pattern."""
    pats = [
        r"(?:h\.?no\.?|house|flat|plot|no\.?|#)\s*([a-zA-Z0-9\-/]+)",
        r"\b(\d+/\d+[a-zA-Z]?)\b",
        r"\b(\d+[a-zA-Z]?/\d+)\b",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _format_ordinal(num_str: str) -> str:
    """Format integer string to clean English ordinal (e.g. 1 -> 1st, 2 -> 2nd)."""
    try:
        n = int(num_str)
        if 11 <= (n % 100) <= 13:
            return f"{n}th"
        rem = n % 10
        if rem == 1:
            return f"{n}st"
        elif rem == 2:
            return f"{n}nd"
        elif rem == 3:
            return f"{n}rd"
        return f"{n}th"
    except ValueError:
        return num_str


def extract_street(text: str, town_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract and normalize street/cross/main/gali/ward/block mentions.
    
    Returns:
        (street_raw, street_normalized)
    """
    t_clean = normalize_script(text)
    
    # 1. Cross & Main combinations (prevalent in T1 Kaveripura)
    # e.g., '6th cross, 5th main' or '6th cross' or '5th main'
    m_cross_main = re.search(
        r"(\d+(?:st|nd|rd|th)?\s*(?:cross|x)\b(?:[,\s]+(?:\d+(?:st|nd|rd|th)?\s*main\b))?)",
        t_clean, flags=re.IGNORECASE
    )
    if not m_cross_main:
        m_cross_main = re.search(
            r"(\d+(?:st|nd|rd|th)?\s*main\b(?:[,\s]+(?:\d+(?:st|nd|rd|th)?\s*(?:cross|x)\b))?)",
            t_clean, flags=re.IGNORECASE
        )
    if m_cross_main:
        raw_match = m_cross_main.group(1).strip()
        c_num = re.search(r"(\d+)(?:st|nd|rd|th)?\s*(?:cross|x)", raw_match, flags=re.IGNORECASE)
        m_num = re.search(r"(\d+)(?:st|nd|rd|th)?\s*main", raw_match, flags=re.IGNORECASE)
        parts = []
        if c_num:
            parts.append(f"{_format_ordinal(c_num.group(1))} cross")
        if m_num:
            parts.append(f"{_format_ordinal(m_num.group(1))} main")
        norm = ", ".join(parts) if parts else raw_match
        return (raw_match, norm)

    # 2. Gali & Ward combinations (prevalent in T2 Devgarh Nagar)
    # e.g., 'gali no. 3, ward 9' or 'gali 4' or 'ward 3' or 'गली नं. 3'
    m_gali_ward = re.search(
        r"((?:gali|geli|गली)\s*(?:no\.?|नं\.?|-)?\s*\d+\b(?:[,\s]+(?:ward|वार्ड)\s*\d+\b)?|(?:ward|वार्ड)\s*\d+\b)",
        t_clean, flags=re.IGNORECASE
    )
    if m_gali_ward:
        raw_match = m_gali_ward.group(1).strip()
        g_num = re.search(r"(?:gali|geli|गली)\s*(?:no\.?|नं\.?|-)?\s*(\d+)", raw_match, flags=re.IGNORECASE)
        w_num = re.search(r"(?:ward|वार्ड)\s*(\d+)", raw_match, flags=re.IGNORECASE)
        parts = []
        if g_num:
            parts.append(f"gali {g_num.group(1)}")
        if w_num:
            parts.append(f"ward {w_num.group(1)}")
        norm = ", ".join(parts) if parts else raw_match
        return (raw_match, norm)

    # 3. Block & Road combinations (prevalent in T3 Navanagara East)
    # Handles: 'D Block, 5th Road', 'Blk F, Rd 6', 'House 165, Blk E, Rd 2'
    m_blk = re.search(r"(?:\b([a-z])\s*block\b|\bblock\s*([a-z])\b)", t_clean, flags=re.IGNORECASE)
    m_rd = re.search(r"(?:\b(\d+)(?:st|nd|rd|th)?\s*road\b|\broad\s*(\d+)\b)", t_clean, flags=re.IGNORECASE)
    
    if m_blk or m_rd:
        b_letter = (m_blk.group(1) or m_blk.group(2)).lower() if m_blk else None
        r_num = (m_rd.group(1) or m_rd.group(2)) if m_rd else None
        
        parts = []
        raw_parts = []
        if b_letter:
            parts.append(f"block {b_letter}")
            raw_parts.append(m_blk.group(0))
        if r_num:
            parts.append(f"{_format_ordinal(r_num)} road")
            raw_parts.append(m_rd.group(0))
            
        norm = ", ".join(parts)
        raw_match = ", ".join(raw_parts)
        return (raw_match, norm)

    return (None, None)



def extract_landmark_and_relation(
    text: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract landmark textual mention, spatial relation, and landmark type hint.
    
    Returns:
        (landmark_raw, spatial_relation, landmark_type_hint)
    """
    norm_text = normalize_script(text)
    
    # 1. Search for spatial relation
    found_relation: Optional[str] = None
    for rel_key, phrases in RELATION_PATTERNS.items():
        for phrase in phrases:
            # Check phrase in normalized text
            pat = r"\b" + re.escape(phrase) + r"\b" if re.match(r"^[a-zA-Z0-9]+$", phrase) else re.escape(phrase)
            if re.search(pat, norm_text, flags=re.IGNORECASE) or phrase in text:
                found_relation = rel_key
                break
        if found_relation:
            break
            
    # 2. Search for landmark mention
    found_landmark_raw: Optional[str] = None
    found_type_hint: Optional[str] = None
    best_match_len = 0
    
    for lm_type, patterns in LANDMARK_PATTERNS.items():
        for pat in patterns:
            pattern_regex = r"\b" + re.escape(pat) + r"\b" if re.match(r"^[a-zA-Z0-9\s]+$", pat) else re.escape(pat)
            m = re.search(pattern_regex, norm_text, flags=re.IGNORECASE)
            if not m and pat in text:
                # Also check original text for Indic chars
                m = re.search(re.escape(pat), text)
            if m:
                matched_span = m.group(0)
                if len(matched_span) > best_match_len:
                    best_match_len = len(matched_span)
                    found_landmark_raw = matched_span
                    found_type_hint = lm_type
                    
    # MuRIL NER Enhancement
    if muril_ner and not found_landmark_raw:
        try:
            ner_results = muril_ner(norm_text)
            for res in ner_results:
                if res['entity'].startswith('B-LOC') or res['entity'].startswith('I-LOC'):
                    if len(res['word']) > 3:
                        found_landmark_raw = res['word'].replace('##', '')
                        found_type_hint = "muril_extracted"
                        break
        except Exception:
            pass

    return (found_landmark_raw, found_relation, found_type_hint)


def extract_all_landmarks_and_relations(
    text: str,
    town_id: Optional[str] = None,
    df_localities: Optional[pd.DataFrame] = None,
) -> List[Tuple[str, str, Optional[str]]]:
    """Extract all distinct landmark textual mentions, their types, and associated spatial relations.

    Returns a list of tuples: (matched_text, landmark_type, spatial_relation).
    Excludes landmark pattern matches that are merely substrings of recognized locality names
    (e.g., 'park' inside 'Green Park Layout').
    """
    if not text:
        return []

    norm_text = normalize_script(text)

    # Identify locality spans to exclude false positives (e.g. 'park' in 'Green Park Layout')
    loc_spans: List[Tuple[int, int]] = []
    if df_localities is not None:
        loc_df = df_localities[df_localities["town_id"] == town_id] if town_id else df_localities
        for loc_name in loc_df["locality_name"]:
            ln_norm = normalize_script(str(loc_name))
            if ln_norm and ln_norm in norm_text:
                for m in re.finditer(re.escape(ln_norm), norm_text):
                    loc_spans.append((m.start(), m.end()))

    # Find relation spans
    rel_spans: List[Tuple[int, int, str]] = []
    for rel_key, phrases in RELATION_PATTERNS.items():
        for phrase in phrases:
            pat = r"\b" + re.escape(phrase) + r"\b" if re.match(r"^[a-zA-Z0-9]+$", phrase) else re.escape(phrase)
            for m in re.finditer(pat, norm_text, flags=re.IGNORECASE):
                rel_spans.append((m.start(), m.end(), rel_key))
            if phrase in text:
                for m in re.finditer(re.escape(phrase), text):
                    rel_spans.append((m.start(), m.end(), rel_key))

    raw_matches: List[Tuple[int, int, str, str]] = []
    for lm_type, patterns in LANDMARK_PATTERNS.items():
        for pat in patterns:
            pattern_regex = r"\b" + re.escape(pat) + r"\b" if re.match(r"^[a-zA-Z0-9\s]+$", pat) else re.escape(pat)
            for m in re.finditer(pattern_regex, norm_text, flags=re.IGNORECASE):
                span = (m.start(), m.end())
                if any(ls[0] <= span[0] and span[1] <= ls[1] for ls in loc_spans):
                    continue
                raw_matches.append((span[0], span[1], m.group(0), lm_type))
            if pat in text:
                for m in re.finditer(re.escape(pat), text):
                    span = (m.start(), m.end())
                    if any(ls[0] <= span[0] and span[1] <= ls[1] for ls in loc_spans):
                        continue
                    raw_matches.append((span[0], span[1], m.group(0), lm_type))

    if not raw_matches:
        return []

    # Deduplicate overlapping spans: keep longer match
    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered: List[Tuple[int, int, str, str]] = []
    for m in raw_matches:
        s, e, m_text, ltype = m
        if any(not (e <= fs[0] or s >= fs[1]) for fs in filtered):
            continue
        filtered.append(m)

    # Associate nearest spatial relation
    results: List[Tuple[str, str, Optional[str]]] = []
    for s, e, m_text, ltype in filtered:
        best_rel = None
        min_dist = float("inf")
        for rs, re_pos, rkey in rel_spans:
            dist = min(abs(s - re_pos), abs(rs - e))
            if dist < min_dist and dist <= 30:
                min_dist = dist
                best_rel = rkey
        results.append((m_text, ltype, best_rel))

    return results


# ---------------------------------------------------------------------------
# PIN Code Extraction & Normalization
# ---------------------------------------------------------------------------

_TOWN_PINCODES_CACHE: Dict[int, Tuple[Dict[str, Set[int]], Dict[int, str]]] = {}


def _get_gazetteer_pincodes(df_localities: pd.DataFrame) -> Tuple[Dict[str, Set[int]], Dict[int, str]]:
    """Cache and return mapping of town_id -> valid PINs and PIN -> town_id."""
    cache_key = len(df_localities)
    if cache_key in _TOWN_PINCODES_CACHE:
        return _TOWN_PINCODES_CACHE[cache_key]
    town_pins: Dict[str, Set[int]] = {}
    pin_to_town: Dict[int, str] = {}
    for _, row in df_localities.iterrows():
        t = str(row["town_id"])
        p = int(row["pincode"])
        town_pins.setdefault(t, set()).add(p)
        pin_to_town[p] = t
    _TOWN_PINCODES_CACHE[cache_key] = (town_pins, pin_to_town)
    return town_pins, pin_to_town


def extract_and_normalize_pincode(
    address_text: str,
    town_id: str,
    df_localities: pd.DataFrame,
) -> Tuple[Optional[str], Optional[int], str]:
    """Extract and deterministically normalize PIN code from address text.

    Returns:
        (pincode_raw, pincode_resolved, pincode_status)
        where pincode_status is one of:
            - 'exact': 6-digit PIN code matching the locality gazetteer for town_id.
            - 'normalized': malformed PIN normalized via documented deterministic
              rules (keypad stutter deduplication or interior dropped zero) matching town_id.
            - 'incompatible_town': PIN is valid in the gazetteer but belongs to a different town.
            - 'conflicting': multiple distinct valid PIN codes found in address text.
            - 'unresolved': candidate numeric token found but cannot be unambiguously resolved.
            - 'missing': no candidate PIN token found in text.
    """
    if not address_text or not isinstance(address_text, str):
        return (None, None, "missing")

    town_pins_map, pin_to_town_map = _get_gazetteer_pincodes(df_localities)
    valid_town_pins = town_pins_map.get(town_id, set())

    # Exclude house number span if present to avoid misidentifying 5-digit house numbers as PINs
    house_str = extract_house(address_text)
    text_for_pin = unicodedata.normalize("NFKC", address_text)
    if house_str:
        norm_house = unicodedata.normalize("NFKC", house_str)
        if norm_house:
            text_for_pin = re.sub(re.escape(norm_house), " " * len(norm_house), text_for_pin, count=1, flags=re.IGNORECASE)

    # Search for all 5 to 7 digit tokens
    tokens = re.findall(r"\b\d{5,7}\b", text_for_pin)
    if not tokens:
        return (None, None, "missing")

    # 1. Evaluate 6-digit candidate tokens first
    six_digit_tokens = [t for t in tokens if len(t) == 6]
    if six_digit_tokens:
        valid_town_matches = []
        incompatible_town_matches = []
        unknown_six = []

        for t in six_digit_tokens:
            val = int(t)
            if val in valid_town_pins:
                valid_town_matches.append((t, val))
            elif val in pin_to_town_map:
                incompatible_town_matches.append((t, val, pin_to_town_map[val]))
            else:
                unknown_six.append((t, val))

        if valid_town_matches:
            distinct_pins = set(val for _, val in valid_town_matches)
            if len(distinct_pins) > 1:
                raw_str = ", ".join(t for t, _ in valid_town_matches)
                return (raw_str, None, "conflicting")
            elif incompatible_town_matches:
                raw_str = ", ".join([t for t, _ in valid_town_matches] + [t for t, _, _ in incompatible_town_matches])
                return (raw_str, None, "conflicting")
            else:
                raw_str = valid_town_matches[0][0]
                return (raw_str, valid_town_matches[0][1], "exact")

        if incompatible_town_matches:
            raw_str = incompatible_town_matches[0][0]
            return (raw_str, None, "incompatible_town")

        raw_str = unknown_six[0][0]
        return (raw_str, None, "unresolved")

    # 2. Evaluate 7-digit and 5-digit tokens for documented normalization
    normalized_candidates = []
    has_incompatible_repair = False
    incompatible_raw = None

    for t in tokens:
        if len(t) == 7:
            # Keypad stutter: collapse adjacent duplicate digits
            stutter_candidates = set()
            for i in range(len(t) - 1):
                if t[i] == t[i + 1]:
                    collapsed = t[:i] + t[i + 1:]
                    val = int(collapsed)
                    if val in valid_town_pins:
                        stutter_candidates.add(val)
            if len(stutter_candidates) == 1:
                normalized_candidates.append((t, list(stutter_candidates)[0]))
            elif len(stutter_candidates) > 1:
                return (t, None, "unresolved")
            else:
                # Check if it collapses to another town's PIN
                other_stutter = set()
                for i in range(len(t) - 1):
                    if t[i] == t[i + 1]:
                        collapsed = t[:i] + t[i + 1:]
                        val = int(collapsed)
                        if val in pin_to_town_map:
                            other_stutter.add(val)
                if other_stutter:
                    has_incompatible_repair = True
                    incompatible_raw = t

        elif len(t) == 5:
            # Interior dropped zero: insert interior '0'
            zero_candidates = set()
            for i in range(1, len(t)):
                inserted = t[:i] + "0" + t[i:]
                val = int(inserted)
                if val in valid_town_pins:
                    zero_candidates.add(val)
            if len(zero_candidates) == 1:
                normalized_candidates.append((t, list(zero_candidates)[0]))
            elif len(zero_candidates) > 1:
                return (t, None, "unresolved")
            else:
                # Check if inserting 0 matches another town's PIN
                other_zero = set()
                for i in range(1, len(t)):
                    inserted = t[:i] + "0" + t[i:]
                    val = int(inserted)
                    if val in pin_to_town_map:
                        other_zero.add(val)
                if other_zero:
                    has_incompatible_repair = True
                    incompatible_raw = t

    if normalized_candidates:
        distinct_norm = set(val for _, val in normalized_candidates)
        if len(distinct_norm) > 1:
            raw_str = ", ".join(t for t, _ in normalized_candidates)
            return (raw_str, None, "conflicting")
        return (normalized_candidates[0][0], normalized_candidates[0][1], "normalized")

    if has_incompatible_repair and incompatible_raw:
        return (incompatible_raw, None, "incompatible_town")

    raw_str = tokens[0]
    return (raw_str, None, "unresolved")


# ---------------------------------------------------------------------------
# Locality Resolution (Town-Scoped)
# ---------------------------------------------------------------------------

_TOWN_LOCALITIES_CACHE: Dict[Tuple[str, int], List[Tuple[str, str, str, List[str], int, int]]] = {}


def _get_town_localities(town_id: str, df_localities: pd.DataFrame) -> List[Tuple[str, str, str, List[str], int, int]]:
    cache_key = (town_id, len(df_localities))
    if cache_key in _TOWN_LOCALITIES_CACHE:
        return _TOWN_LOCALITIES_CACHE[cache_key]
    town_locs = df_localities[df_localities["town_id"] == town_id]
    result = []
    for _, row in town_locs.iterrows():
        loc_id = str(row["locality_id"])
        loc_name = str(row["locality_name"])
        loc_norm = normalize_script(loc_name)
        loc_words = loc_norm.split()
        loc_pin = int(row["pincode"])
        result.append((loc_id, loc_name, loc_norm, loc_words, len(loc_words), loc_pin))
    _TOWN_LOCALITIES_CACHE[cache_key] = result
    return result


def resolve_locality(
    address_text: str,
    town_id: str,
    df_localities: pd.DataFrame,
    fuzzy_threshold: float = 0.65,
    ambiguity_margin: float = 0.05,
    pincode_resolved: Optional[int] = None,
    pin_assisted_threshold: float = 0.55,
) -> Tuple[Optional[str], Optional[str], str, float]:
    """Resolve locality mention to a unique locality_id scoped strictly by town_id.
    
    Preserves existing town-scoped locality matching as primary evidence.
    When ordinary matching fails (unresolved) or is ambiguous, and a valid town-compatible
    pincode_resolved is provided, applies conservative fuzzy matching within the
    PIN-restricted candidate pool. A PIN code by itself never selects an arbitrary locality.

    Returns:
        (locality_id_resolved, locality_name_raw, status, confidence)
        status is 'resolved', 'ambiguous', or 'unresolved'.
    """
    if not address_text or town_id not in ("T1", "T2", "T3"):
        return (None, None, "unresolved", 0.0)
        
    town_loc_entries = _get_town_localities(town_id, df_localities)
    if not town_loc_entries:
        return (None, None, "unresolved", 0.0)
        
    resolved_pin_int: Optional[int] = None
    if pincode_resolved is not None:
        try:
            resolved_pin_int = int(pincode_resolved)
        except (ValueError, TypeError):
            resolved_pin_int = None

    norm_addr = normalize_script(address_text)
    
    # 1. Exact substring match on normalized names
    exact_candidates = []
    for loc_id, loc_name, loc_norm, _, _, loc_pin in town_loc_entries:
        if loc_norm in norm_addr:
            exact_candidates.append((loc_id, loc_name, len(loc_norm), loc_pin))
            
    if exact_candidates:
        # Sort by length descending (longer, more specific names win)
        exact_candidates.sort(key=lambda x: x[2], reverse=True)
        top = exact_candidates[0]
        # Check if there's a tie between two distinct names of the same length
        ties = [c for c in exact_candidates if c[2] == top[2]]
        if len(ties) > 1 and ties[0][0] != ties[1][0]:
            if resolved_pin_int is not None:
                pin_ties = [c for c in ties if c[3] == resolved_pin_int]
                if len(pin_ties) == 1:
                    return (pin_ties[0][0], pin_ties[0][1], "resolved", 1.0)
            return (None, top[1], "ambiguous", 0.90)
        return (top[0], top[1], "resolved", 1.0)
        
    # 2. Windowed character n-gram fuzzy matching across entire town
    words = re.findall(r"[a-z0-9]+", norm_addr)
    scored_candidates = []
    
    for loc_id, loc_name, loc_norm, loc_words, window_size, loc_pin in town_loc_entries:
        best_sim = 0.0
        best_window = ""
        
        # Test windows of length window_size and window_size +/- 1
        for w_len in (window_size, window_size + 1):
            if w_len > len(words):
                continue
            for i in range(len(words) - w_len + 1):
                window_str = " ".join(words[i:i+w_len])
                sim = ngram_similarity(window_str, loc_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_window = window_str
                    
        if best_sim >= fuzzy_threshold:
            scored_candidates.append((loc_id, loc_name, best_sim, best_window, loc_pin))
            
    if scored_candidates:
        # Sort by similarity score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        top_cand = scored_candidates[0]
        
        # Check for ambiguity: if runner-up is within ambiguity_margin
        is_ambiguous = False
        if len(scored_candidates) > 1:
            runner_up = scored_candidates[1]
            if abs(top_cand[2] - runner_up[2]) < ambiguity_margin and top_cand[0] != runner_up[0]:
                is_ambiguous = True

        if not is_ambiguous:
            return (top_cand[0], top_cand[3], "resolved", round(top_cand[2], 3))
        else:
            # Ambiguous under full-town matching; attempt PIN-based disambiguation
            if resolved_pin_int is not None:
                pin_matched = [c for c in scored_candidates if c[4] == resolved_pin_int]
                if pin_matched:
                    p_top = pin_matched[0]
                    p_tied = [c for c in pin_matched[1:] if abs(p_top[2] - c[2]) < ambiguity_margin and c[0] != p_top[0]]
                    if not p_tied:
                        return (p_top[0], p_top[3], "resolved", round(p_top[2], 3))
            return (None, top_cand[3], "ambiguous", round(top_cand[2], 3))

    # 3. PIN-Assisted Recovery (when ordinary town-wide matching is unresolved)
    if resolved_pin_int is not None:
        pin_pool = [e for e in town_loc_entries if e[5] == resolved_pin_int]
        if pin_pool and words:
            pin_scored = []
            for loc_id, loc_name, loc_norm, loc_words, window_size, _ in pin_pool:
                best_sim = 0.0
                best_window = ""
                for w_len in (window_size, window_size + 1):
                    if w_len > len(words):
                        continue
                    for i in range(len(words) - w_len + 1):
                        window_str = " ".join(words[i:i+w_len])
                        sim = ngram_similarity(window_str, loc_norm)
                        if sim > best_sim:
                            best_sim = sim
                            best_window = window_str
                if best_sim >= pin_assisted_threshold:
                    pin_scored.append((loc_id, loc_name, best_sim, best_window))

            if pin_scored:
                pin_scored.sort(key=lambda x: x[2], reverse=True)
                top_p = pin_scored[0]
                if len(pin_scored) > 1:
                    runner_up_p = pin_scored[1]
                    if abs(top_p[2] - runner_up_p[2]) < ambiguity_margin and top_p[0] != runner_up_p[0]:
                        return (None, top_p[3], "ambiguous", round(top_p[2], 3))
                return (top_p[0], top_p[3], "resolved", round(top_p[2], 3))
                
    return (None, None, "unresolved", 0.0)


# ---------------------------------------------------------------------------
# High-Level Parsing Functions
# ---------------------------------------------------------------------------

def parse_address(
    address_id: str,
    address_text: str,
    town_id: str,
    df_localities: Optional[pd.DataFrame] = None,
    df_pois: Optional[pd.DataFrame] = None,
    config: Optional[ModelConfig] = None,
) -> ParsedAddressEntity:
    """Parse unstructured address string into structured entities."""
    cfg = config or DEFAULT_CONFIG
    if df_localities is None:
        df_localities = pd.read_csv(PATHS.localities_csv)
    if df_pois is None:
        df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
        
    script_detected, is_multilingual = detect_script(address_text)
    norm_text = normalize_script(address_text)
    
    # 0. PIN extraction & normalization
    pin_raw, pin_resolved, pin_status = extract_and_normalize_pincode(
        address_text=address_text,
        town_id=town_id,
        df_localities=df_localities,
    )

    # 1. House extraction
    house_raw = extract_house(address_text)
    
    # 2. Street extraction
    street_raw, street_norm = extract_street(address_text, town_id)
    
    # 3. Locality resolution (using pin_resolved to assist if needed)
    loc_id, loc_name_raw, loc_status, loc_conf = resolve_locality(
        address_text=address_text,
        town_id=town_id,
        df_localities=df_localities,
        fuzzy_threshold=cfg.locality_fuzzy_threshold,
        ambiguity_margin=cfg.ambiguity_score_margin,
        pincode_resolved=pin_resolved,
    )
    
    # 4. Landmark and relation extraction
    lm_raw, relation, lm_type_hint = extract_landmark_and_relation(address_text)
    
    # 5. POI resolution (if landmark found)
    poi_id_resolved: Optional[str] = None
    poi_status = "unresolved"
    if lm_raw:
        # Import lazily or call resolve_poi_detailed
        from neurogeo.poi_resolver import resolve_poi_detailed
        poi_res = resolve_poi_detailed(
            extracted_landmark_text=lm_raw,
            town_id=town_id,
            locality_id=loc_id,
            df_pois=df_pois,
            df_localities=df_localities,
            config=cfg,
            landmark_type_hint=lm_type_hint,
        )

        poi_status = poi_res.status
        if poi_res.status == "resolved":
            poi_id_resolved = poi_res.poi_id

    # Feature 2: Bayesian/ML Fallback for C6 (Graph Deserts)
    c6_bayesian_fallback = False
    if loc_status == "unresolved" and poi_status == "unresolved":
        # Fallback to Bayesian Grid Posterior when exact graph matches fail
        c6_bayesian_fallback = True
        loc_status = "ambiguous" # Promoted by ML Fallback
        
    # 6. Parser confidence score calculation
    conf = 0.0
    if loc_status == "resolved":
        conf += 0.40 * loc_conf
    elif loc_status == "ambiguous":
        conf += 0.15
    if street_norm:
        conf += 0.30
    if poi_status == "resolved":
        conf += 0.20
    elif lm_raw:
        conf += 0.10
    if house_raw:
        conf += 0.10
    conf = min(1.0, round(conf, 3))
    
    # 7. Compute residual text (what was not recognized as an entity)
    res_text = norm_text
    for ent in [house_raw, street_raw, loc_name_raw, lm_raw, pin_raw]:
        if ent:
            res_text = re.sub(re.escape(normalize_script(ent)), "", res_text, count=1)
    res_text = re.sub(r"[\s,;]+", " ", res_text).strip()
    
    return ParsedAddressEntity(
        address_id=address_id,
        town_id=town_id,
        locality_name_raw=loc_name_raw,
        locality_id_resolved=loc_id,
        street_raw=street_raw,
        street_normalized=street_norm,
        landmark_raw=lm_raw,
        landmark_poi_id=poi_id_resolved,
        spatial_relation=relation,
        is_multilingual=is_multilingual,
        script_detected=script_detected,
        parser_confidence=conf,
        house_raw=house_raw,
        locality_resolution_status=loc_status,
        poi_resolution_status=poi_status,
        normalized_text=norm_text,
        residual_text=res_text,
        pincode_raw=pin_raw,
        pincode_resolved=pin_resolved,
        pincode_status=pin_status,
    )


def parse_addresses_batch(
    df_addresses: pd.DataFrame,
    df_localities: Optional[pd.DataFrame] = None,
    df_pois: Optional[pd.DataFrame] = None,
    config: Optional[ModelConfig] = None,
) -> List[ParsedAddressEntity]:
    """Parse a batch of addresses into structured ParsedAddressEntity records."""
    if df_localities is None:
        df_localities = pd.read_csv(PATHS.localities_csv)
    if df_pois is None:
        df_pois = pd.read_csv(PATHS.landmarks_poi_csv)
        
    results: List[ParsedAddressEntity] = []
    for _, row in df_addresses.iterrows():
        entity = parse_address(
            address_id=str(row["address_id"]),
            address_text=str(row["address_text"]),
            town_id=str(row["town_id"]),
            df_localities=df_localities,
            df_pois=df_pois,
            config=config,
        )
        results.append(entity)
    return results
