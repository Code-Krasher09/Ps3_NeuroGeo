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
                    
    return (found_landmark_raw, found_relation, found_type_hint)


# ---------------------------------------------------------------------------
# Locality Resolution (Town-Scoped)
# ---------------------------------------------------------------------------

_TOWN_LOCALITIES_CACHE: Dict[Tuple[str, int], List[Tuple[str, str, str, List[str], int]]] = {}


def _get_town_localities(town_id: str, df_localities: pd.DataFrame) -> List[Tuple[str, str, str, List[str], int]]:
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
        result.append((loc_id, loc_name, loc_norm, loc_words, len(loc_words)))
    _TOWN_LOCALITIES_CACHE[cache_key] = result
    return result


def resolve_locality(
    address_text: str,
    town_id: str,
    df_localities: pd.DataFrame,
    fuzzy_threshold: float = 0.65,
    ambiguity_margin: float = 0.05,
) -> Tuple[Optional[str], Optional[str], str, float]:
    """Resolve locality mention to a unique locality_id scoped strictly by town_id.
    
    Returns:
        (locality_id_resolved, locality_name_raw, status, confidence)
        status is 'resolved', 'ambiguous', or 'unresolved'.
    """
    if not address_text or town_id not in ("T1", "T2", "T3"):
        return (None, None, "unresolved", 0.0)
        
    town_loc_entries = _get_town_localities(town_id, df_localities)
    if not town_loc_entries:
        return (None, None, "unresolved", 0.0)
        
    norm_addr = normalize_script(address_text)
    
    # 1. Exact substring match on normalized names
    exact_candidates = []
    for loc_id, loc_name, loc_norm, _, _ in town_loc_entries:
        if loc_norm in norm_addr:
            exact_candidates.append((loc_id, loc_name, len(loc_norm)))
            
    if exact_candidates:
        # Sort by length descending (longer, more specific names win)
        exact_candidates.sort(key=lambda x: x[2], reverse=True)
        top = exact_candidates[0]
        # Check if there's a tie between two distinct names of the same length
        ties = [c for c in exact_candidates if c[2] == top[2]]
        if len(ties) > 1 and ties[0][0] != ties[1][0]:
            return (None, top[1], "ambiguous", 0.90)
        return (top[0], top[1], "resolved", 1.0)
        
    # 2. Windowed character n-gram fuzzy matching
    words = re.findall(r"[a-z0-9]+", norm_addr)
    scored_candidates = []
    
    for loc_id, loc_name, loc_norm, loc_words, window_size in town_loc_entries:
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
            scored_candidates.append((loc_id, loc_name, best_sim, best_window))
            
    if scored_candidates:
        # Sort by similarity score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        top_cand = scored_candidates[0]
        
        # Check for ambiguity: if runner-up is within ambiguity_margin
        if len(scored_candidates) > 1:
            runner_up = scored_candidates[1]
            if abs(top_cand[2] - runner_up[2]) < ambiguity_margin and top_cand[0] != runner_up[0]:
                return (None, top_cand[3], "ambiguous", round(top_cand[2], 3))
                
        return (top_cand[0], top_cand[3], "resolved", round(top_cand[2], 3))
        
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
    
    # 1. House extraction
    house_raw = extract_house(address_text)
    
    # 2. Street extraction
    street_raw, street_norm = extract_street(address_text, town_id)
    
    # 3. Locality resolution
    loc_id, loc_name_raw, loc_status, loc_conf = resolve_locality(
        address_text=address_text,
        town_id=town_id,
        df_localities=df_localities,
        fuzzy_threshold=cfg.locality_fuzzy_threshold,
        ambiguity_margin=cfg.ambiguity_score_margin,
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
    for ent in [house_raw, street_raw, loc_name_raw, lm_raw]:
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
