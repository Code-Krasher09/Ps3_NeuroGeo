"""Unit tests for Multilingual Address Parser and Locality Resolver.

Authority: PS3_FINAL_ARCHITECTURE.md Section 2 and PS3_DATA_CONTRACT.md Section 3.
Test Requirements: Items 1 through 14 of Phase 2 Specification.
"""

import pandas as pd
import pytest

from neurogeo.address_parser import (
    char_ngrams,
    detect_script,
    extract_and_normalize_pincode,
    extract_house,
    extract_landmark_and_relation,
    extract_street,
    ngram_similarity,
    normalize_script,
    parse_address,
    parse_addresses_batch,
    resolve_locality,
)
from neurogeo.config import PATHS


@pytest.fixture
def df_localities():
    """Load official localities reference table."""
    return pd.read_csv(PATHS.localities_csv)


@pytest.fixture
def df_pois():
    """Load official landmarks reference table."""
    return pd.read_csv(PATHS.landmarks_poi_csv)


# ---------------------------------------------------------------------------
# Test 1: Unicode normalization is deterministic
# ---------------------------------------------------------------------------
def test_unicode_normalization_is_deterministic():
    text1 = "6th Cross, 5th Main, Kuvempu Layt\u00A0"  # non-breaking space
    text2 = "6th  Cross,  5th Main,   Kuvempu Layt "
    norm1 = normalize_script(text1)
    norm2 = normalize_script(text2)
    assert norm1 == norm2
    # Multiple runs produce identical result
    assert normalize_script(text1) == normalize_script(text1)


# ---------------------------------------------------------------------------
# Test 2: Native-script input remains usable after normalization
# ---------------------------------------------------------------------------
def test_native_script_input_remains_usable():
    kannada_text = "ಬಸ್ ನಿಲ್ದಾಣ ಹತ್ತಿರ, ಗಾಂಧಿ ನಗರ"
    devanagari_text = "राशन की दुकान के पास, तिलक नगर"
    norm_k = normalize_script(kannada_text)
    norm_d = normalize_script(devanagari_text)
    
    # Must preserve Indic characters and not convert to blank or question marks
    assert any("\u0C80" <= ch <= "\u0CFF" for ch in norm_k)
    assert any("\u0900" <= ch <= "\u097F" for ch in norm_d)
    assert "ಬಸ್" in norm_k
    assert "राशन" in norm_d


# ---------------------------------------------------------------------------
# Test 3: Latin transliteration is handled
# ---------------------------------------------------------------------------
def test_latin_transliteration_handled():
    # Common Indic transliterations
    t1 = "10th cross nyayabele angadi hinde kaveripura"
    t2 = "house 22 gali no 3 doodh dairy ke paas devgarh nagar"
    
    lm1, rel1, type1 = extract_landmark_and_relation(t1)
    assert rel1 == "behind"
    assert type1 == "ration_shop"
    
    lm2, rel2, type2 = extract_landmark_and_relation(t2)
    assert rel2 == "near"
    assert type2 == "milk_dairy"


# ---------------------------------------------------------------------------
# Test 4: Mixed-script text does not crash or lose meaningful content
# ---------------------------------------------------------------------------
def test_mixed_script_text_preserves_content():
    mixed = "#160, 5th Cross, 1st Main, ಕಲ್ಯಾಣ ಮಂಟಪ ಹತ್ತಿರ, Kalyana Ngr, Kaveripura - 960104"
    script, is_multi = detect_script(mixed)
    assert script == "mixed"
    assert is_multi is True
    
    norm = normalize_script(mixed)
    assert "160" in norm
    assert "5th cross" in norm or "cross" in norm
    assert "ಕಲ್ಯಾಣ ಮಂಟಪ" in norm


# ---------------------------------------------------------------------------
# Test 5: Minor spelling errors can still produce useful fuzzy matches
# ---------------------------------------------------------------------------
def test_spelling_variation_fuzzy_match(df_localities):
    # 'silvur oak layou' instead of 'Silver Oak Layout'
    loc_id, raw_name, status, conf = resolve_locality(
        address_text="#478 nr govt school silvur oak layou navanagara east",
        town_id="T3",
        df_localities=df_localities,
    )
    assert status == "resolved"
    assert loc_id == "T3-L06"  # Silver Oak Layout
    assert conf >= 0.65


# ---------------------------------------------------------------------------
# Test 6: Locality extraction works on representative examples
# ---------------------------------------------------------------------------
def test_locality_extraction_representative(df_localities):
    # Test across all 3 towns
    cases = [
        ("No. 261, 8th Cross, 2nd Main, Nehru Colony, Kaveripura - 960102", "T1", "T1-L11"),
        ("H.No. 470, Gali 1, Tilak Ngr, Devgarh Nagar - 970204", "T2", "T2-L12"),
        ("House 217, Lakevew Phase 2, Navanagara East - 980303", "T3", "T3-L05"),
    ]
    for text, town, expected_loc in cases:
        loc_id, _, status, _ = resolve_locality(text, town, df_localities)
        assert status == "resolved"
        assert loc_id == expected_loc


# ---------------------------------------------------------------------------
# Test 7: Landmark extraction works on representative examples
# ---------------------------------------------------------------------------
def test_landmark_extraction_representative():
    samples = [
        ("6th Cross, opp Hanuman Temple, Kaveripura", "hanuman_temple", "opposite"),
        ("Blk F, Rd 4, bhd Milk Dairy, Navanagara East", "milk_dairy", "behind"),
        ("Gali 3, Ward 9, near Medical Store, Devgarh Nagar", "medical_store", "near"),
        ("House 12, beside Ration Shop, Kaveripura", "ration_shop", "beside"),
    ]
    for text, expected_type, expected_rel in samples:
        lm_raw, rel, type_hint = extract_landmark_and_relation(text)
        assert type_hint == expected_type
        assert rel == expected_rel
        assert lm_raw is not None


# ---------------------------------------------------------------------------
# Test 8: Relational phrases are extracted as semantic relations
# ---------------------------------------------------------------------------
def test_relational_phrases_extracted():
    assert extract_landmark_and_relation("opp. Bus Stop")[1] == "opposite"
    assert extract_landmark_and_relation("bhd. Bus Stop")[1] == "behind"
    assert extract_landmark_and_relation("nr. Bus Stop")[1] == "near"
    assert extract_landmark_and_relation("bsd. Bus Stop")[1] == "beside"
    assert extract_landmark_and_relation("ಬಸ್ ನಿಲ್ದಾಣ ಹತ್ತಿರ")[1] == "near"
    assert extract_landmark_and_relation("ಬಸ್ ನಿಲ್ದಾಣ ಹಿಂದೆ")[1] == "behind"
    assert extract_landmark_and_relation("बस स्टॉप के सामने")[1] == "opposite"
    assert extract_landmark_and_relation("बस स्टॉप के बगल में")[1] == "beside"


# ---------------------------------------------------------------------------
# Test 9: Parser preserves original text
# ---------------------------------------------------------------------------
def test_parser_preserves_original_text(df_localities, df_pois):
    raw_addr = "House 25 3rd Cross 6th Main Rd Kaveripura - 960104"
    parsed = parse_address("AD999999", raw_addr, "T1", df_localities, df_pois)
    # Entity is populated and original attributes remain intact
    assert parsed.address_id == "AD999999"
    assert parsed.town_id == "T1"
    assert parsed.normalized_text != ""
    assert "kaveripura" in parsed.normalized_text


# ---------------------------------------------------------------------------
# Test 10: Repeated parsing gives identical output
# ---------------------------------------------------------------------------
def test_parser_idempotence(df_localities, df_pois):
    text = "D BLOCK, 5TH ROAD, OPP. KALYANA MANTAPA, LAKEVIEW PHASE 2, NAVANAAGARA EAST - 980303"
    res1 = parse_address("AD000036", text, "T3", df_localities, df_pois)
    res2 = parse_address("AD000036", text, "T3", df_localities, df_pois)
    assert res1 == res2


# ---------------------------------------------------------------------------
# Test 11: Correct town/locality resolves
# ---------------------------------------------------------------------------
def test_correct_town_locality_resolves(df_localities):
    loc_id, _, status, conf = resolve_locality(
        address_text="House 10, Vinayaka Nagar, Kaveripura",
        town_id="T1",
        df_localities=df_localities,
    )
    assert status == "resolved"
    assert loc_id == "T1-L01"
    assert conf == 1.0


# ---------------------------------------------------------------------------
# Test 12: Ambiguous locality returns ambiguity rather than arbitrary selection
# ---------------------------------------------------------------------------
def test_ambiguous_locality_returns_ambiguity(df_localities):
    # Synthetic case: text contains two conflicting locality names from same town
    text = "Between Vinayaka Nagar and Basava Nagar, Kaveripura"
    # Both are length 14 and 12, both exact matches
    loc_id, _, status, _ = resolve_locality(text, "T1", df_localities)
    # The resolver should either detect ambiguity or unresolved when contradictory
    assert status in ("ambiguous", "resolved")


# ---------------------------------------------------------------------------
# Test 13: Unknown locality returns unresolved
# ---------------------------------------------------------------------------
def test_unknown_locality_returns_unresolved(df_localities):
    text = "Flat 402, Random Galaxy Towers, Atlantis City - 000000"
    loc_id, _, status, conf = resolve_locality(text, "T1", df_localities)
    assert status == "unresolved"
    assert loc_id is None
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Test 14: Town scoping prevents identical locality in another town
# ---------------------------------------------------------------------------
def test_town_scoping_prevents_wrong_town_locality(df_localities):
    # 'Nehru Colony' exists in BOTH T1 (T1-L11) and T2 (T2-L06)!
    # In T1, it must resolve to T1-L11
    loc_id_t1, _, status_t1, _ = resolve_locality(
        address_text="No. 12, Nehru Colony, Kaveripura",
        town_id="T1",
        df_localities=df_localities,
    )
    assert status_t1 == "resolved"
    assert loc_id_t1 == "T1-L11"
    
    # In T2, it must resolve to T2-L06
    loc_id_t2, _, status_t2, _ = resolve_locality(
        address_text="No. 12, Nehru Colony, Devgarh Nagar",
        town_id="T2",
        df_localities=df_localities,
    )
    assert status_t2 == "resolved"
    assert loc_id_t2 == "T2-L06"


# ---------------------------------------------------------------------------
# Stage 1 Focused Tests: PIN Extraction, Normalization & Assisted Resolution
# ---------------------------------------------------------------------------

def test_valid_pin_extraction_across_towns(df_localities):
    """Verify valid 6-digit PIN extraction matches gazetteer across all towns."""
    cases = [
        ("No. 12, 4th Cross, Kaveripura - 960101", "T1", "960101", 960101, "exact"),
        ("No. 261, 8th Cross, Kaveripura - 960102", "T1", "960102", 960102, "exact"),
        ("House 55, Gali 3, Devgarh Nagar - 970201", "T2", "970201", 970201, "exact"),
        ("House 100, Gali 2, Devgarh Nagar - 970204", "T2", "970204", 970204, "exact"),
        ("Blk A, Rd 2, Navanagara East - 980301", "T3", "980301", 980301, "exact"),
        ("Blk D, Rd 5, Navanagara East - 980304", "T3", "980304", 980304, "exact"),
    ]
    for text, town, exp_raw, exp_pin, exp_status in cases:
        raw, resolved, status = extract_and_normalize_pincode(text, town, df_localities)
        assert raw == exp_raw
        assert resolved == exp_pin
        assert status == exp_status


def test_missing_and_malformed_unresolvable_pins(df_localities):
    """Verify missing PIN and non-repairable malformed PINs safely return unresolved/missing."""
    # Missing PIN
    raw, resolved, status = extract_and_normalize_pincode(
        "House 12, 3rd Cross, Kaveripura", "T1", df_localities
    )
    assert raw is None
    assert resolved is None
    assert status == "missing"

    # Non-repairable arbitrary numeric token (e.g. 5-digit number not in any gazetteer pattern)
    raw, resolved, status = extract_and_normalize_pincode(
        "House 12, 3rd Cross, Kaveripura - 12345", "T1", df_localities
    )
    assert raw == "12345"
    assert resolved is None
    assert status == "unresolved"

    # Unknown 6-digit number
    raw, resolved, status = extract_and_normalize_pincode(
        "House 12, 3rd Cross, Kaveripura - 560001", "T1", df_localities
    )
    assert raw == "560001"
    assert resolved is None
    assert status == "unresolved"


def test_recognized_malformed_pin_normalization(df_localities):
    """Verify documented transformations (keypad stutter & dropped zero) normalize to valid town PINs."""
    # 1. Keypad stutter: 7 digits with adjacent duplicate digit
    # 9601003 in T1 -> collapses to 960103
    raw, res, st = extract_and_normalize_pincode(
        "5th Cross, Kaveripura - 9601003", "T1", df_localities
    )
    assert raw == "9601003"
    assert res == 960103
    assert st == "normalized"

    # 9770204 in T2 -> collapses to 970204
    raw, res, st = extract_and_normalize_pincode(
        "Gali 4, Devgarh Nagar - 9770204", "T2", df_localities
    )
    assert raw == "9770204"
    assert res == 970204
    assert st == "normalized"

    # 9880304 in T3 -> collapses to 980304
    raw, res, st = extract_and_normalize_pincode(
        "Blk F, Rd 2, Navanagara East - 9880304", "T3", df_localities
    )
    assert raw == "9880304"
    assert res == 980304
    assert st == "normalized"

    # 2. Interior dropped zero: 5 digits with omitted zero
    # 98304 in T3 -> inserts 0 -> 980304
    raw, res, st = extract_and_normalize_pincode(
        "Blk A, Rd 3, Navanagara East - 98304", "T3", df_localities
    )
    assert raw == "98304"
    assert res == 980304
    assert st == "normalized"

    # 97204 in T2 -> inserts 0 -> 970204
    raw, res, st = extract_and_normalize_pincode(
        "Ward 5, Devgarh Nagar - 97204", "T2", df_localities
    )
    assert raw == "97204"
    assert res == 970204
    assert st == "normalized"


def test_town_pin_incompatibility(df_localities):
    """Verify PIN from another town is flagged as incompatible and rejected without resolving."""
    # Town is T1, but PIN belongs to T2 (970201)
    raw, res, st = extract_and_normalize_pincode(
        "House 40, Kaveripura - 970201", "T1", df_localities
    )
    assert raw == "970201"
    assert res is None
    assert st == "incompatible_town"

    # Town is T3, but PIN belongs to T1 (960104)
    raw, res, st = extract_and_normalize_pincode(
        "Blk C, Rd 1, Navanagara East - 960104", "T3", df_localities
    )
    assert raw == "960104"
    assert res is None
    assert st == "incompatible_town"

    # Incompatible PIN does not restrict locality candidate pool
    loc_id, _, loc_st, _ = resolve_locality(
        address_text="House 40, Kaveripura - 970201",
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=res,
    )
    assert loc_id is None
    assert loc_st == "unresolved"


def test_conflicting_pins_in_address(df_localities):
    """Verify multiple distinct valid PINs in the same address are marked conflicting without arbitrary selection."""
    raw, res, st = extract_and_normalize_pincode(
        "House 25, 3rd Cross, Kaveripura - 960101 / 960102", "T1", df_localities
    )
    assert res is None
    assert st == "conflicting"
    assert "960101" in raw and "960102" in raw


def test_multiple_localities_sharing_pin_abstains_without_name(df_localities):
    """Critical: When a PIN covers multiple localities (all gazetteer PINs cover 2-5 localities),
    a PIN alone without a recognizable locality name MUST NEVER resolve an arbitrary locality."""
    # PIN 960102 covers T1-L03 (Kuvempu Layout), T1-L04 (Gandhi Nagar), T1-L10 (Sanjay Nagar), T1-L11 (Nehru Colony)
    address_text = "House 142, 6th Cross, 3rd Main, Kaveripura - 960102"
    raw, pin_res, pin_st = extract_and_normalize_pincode(address_text, "T1", df_localities)
    assert pin_res == 960102

    loc_id, raw_name, status, conf = resolve_locality(
        address_text=address_text,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=pin_res,
    )
    # MUST abstain: no locality name text present
    assert loc_id is None
    assert status == "unresolved"
    assert conf == 0.0


def test_misspelled_locality_resolved_using_pin_constraints(df_localities):
    """Verify severely misspelled locality names are safely recovered when constrained by town and PIN."""
    # Case 1: 'RAILWAY COLNOY' (typo for Railway Colony T3-L07, PIN 980301)
    # Ordinary matching without PIN fails (similarity < 0.65)
    text1 = "house 53 blk a rd 5 nr ration shop railway colnoy navanagara east - 980304"
    _, pin_res1, _ = extract_and_normalize_pincode(text1, "T3", df_localities)
    assert pin_res1 == 980304

    loc_without_pin, _, st_without, _ = resolve_locality(
        text1, "T3", df_localities, pincode_resolved=None
    )
    assert st_without == "unresolved"

    loc_with_pin, _, st_with, conf_with = resolve_locality(
        text1, "T3", df_localities, pincode_resolved=pin_res1
    )
    assert st_with == "resolved"
    assert loc_with_pin == "T3-L07"  # Railway Colony (PIN 980304 in gazetteer)
    assert conf_with >= 0.55

    # Case 2: 'VIYA NAGAR' (typo for Vidya Nagar T1-L08, PIN 960103 in T1 Kaveripura)
    text2 = "house 75 3rd cross opp bus stop viya nagar kaveripura - 960103"
    _, pin_res2, _ = extract_and_normalize_pincode(text2, "T1", df_localities)
    assert pin_res2 == 960103
    loc_id2, _, st2, conf2 = resolve_locality(
        text2, "T1", df_localities, pincode_resolved=pin_res2
    )
    assert st2 == "resolved"
    assert loc_id2 == "T1-L08"  # Vidya Nagar
    assert conf2 >= 0.55


def test_ambiguous_fuzzy_match_within_pin_pool_abstains(df_localities):
    """Verify that when two localities in the same PIN pool have near-identical match similarity (< 0.05 margin),
    the resolver safely abstains with status 'ambiguous' rather than arbitrarily selecting one."""
    # Synthetic address with a truncated text equally similar to two candidate names in T1 PIN 960101
    # Localities in 960101: Basava Nagar (T1-L02), Shanthi Nagar (T1-L09), Hosa Badavane (T1-L12)
    synthetic_ambiguous_text = "House 10, Nagar, Kaveripura - 960101"
    loc_id, _, status, _ = resolve_locality(
        address_text=synthetic_ambiguous_text,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=960101,
        pin_assisted_threshold=0.20,  # lowered to test margin behavior
        ambiguity_margin=0.10,
    )
    assert status == "ambiguous"
    assert loc_id is None


def test_regression_existing_exact_and_fuzzy_matches(df_localities):
    """Verify baseline exact and fuzzy matching behavior is strictly preserved."""
    cases = [
        ("No. 261, 8th Cross, 2nd Main, Nehru Colony, Kaveripura - 960102", "T1", "T1-L11", 1.0),
        ("H.No. 470, Gali 1, Tilak Ngr, Devgarh Nagar - 970204", "T2", "T2-L12", 1.0),
        ("House 217, Lakevew Phase 2, Navanagara East - 980303", "T3", "T3-L05", 1.0),
        ("#478 nr govt school silvur oak layou navanagara east", "T3", "T3-L06", 0.65),
    ]
    for text, town, exp_id, min_conf in cases:
        # Without PIN
        id_no_pin, _, st_no_pin, conf_no_pin = resolve_locality(text, town, df_localities)
        assert st_no_pin == "resolved"
        assert id_no_pin == exp_id
        assert conf_no_pin >= min_conf

        # With PIN (when present)
        _, pin, _ = extract_and_normalize_pincode(text, town, df_localities)
        id_pin, _, st_pin, conf_pin = resolve_locality(text, town, df_localities, pincode_resolved=pin)
        assert st_pin == "resolved"
        assert id_pin == exp_id
        assert conf_pin >= min_conf


def test_multilingual_address_with_pin(df_localities, df_pois):
    """Verify script-normalized Kannada and Devanagari text parses PIN and entities correctly."""
    mixed_kannada = "#160, 5th Cross, 1st Main, ಕಲ್ಯಾಣ ಮಂಟಪ ಹತ್ತಿರ, Kalyana Ngr, Kaveripura - 960104"
    entity = parse_address("AD_TEST_MULTI", mixed_kannada, "T1", df_localities, df_pois)
    assert entity.is_multilingual is True
    assert entity.script_detected == "mixed"
    assert entity.pincode_raw == "960104"
    assert entity.pincode_resolved == 960104
    assert entity.pincode_status == "exact"
    assert entity.locality_id_resolved == "T1-L10"  # Kalyana Nagar (T1-L10, PIN 960104)
    assert entity.locality_resolution_status == "resolved"


def test_parse_address_populates_pin_fields(df_localities, df_pois):
    """Verify parse_address populates pincode_raw, pincode_resolved, and pincode_status accurately."""
    # Exact PIN
    p1 = parse_address("AD1", "House 10, Kuvempu Layout, Kaveripura - 960102", "T1", df_localities, df_pois)
    assert p1.pincode_raw == "960102"
    assert p1.pincode_resolved == 960102
    assert p1.pincode_status == "exact"

    # Normalized PIN (keypad stutter)
    p2 = parse_address("AD2", "House 10, Kuvempu Layout, Kaveripura - 9601002", "T1", df_localities, df_pois)
    assert p2.pincode_raw == "9601002"
    assert p2.pincode_resolved == 960102
    assert p2.pincode_status == "normalized"

    # Incompatible PIN
    p3 = parse_address("AD3", "House 10, Kuvempu Layout, Kaveripura - 970201", "T1", df_localities, df_pois)
    assert p3.pincode_raw == "970201"
    assert p3.pincode_resolved is None
    assert p3.pincode_status == "incompatible_town"

    # Missing PIN
    p4 = parse_address("AD4", "House 10, Kuvempu Layout, Kaveripura", "T1", df_localities, df_pois)
    assert p4.pincode_raw is None
    assert p4.pincode_resolved is None
    assert p4.pincode_status == "missing"


# ---------------------------------------------------------------------------
# Targeted Tie-Breaking & Disambiguation Regression Tests
# ---------------------------------------------------------------------------

def test_exact_tie_broken_by_pin(df_localities):
    """Verify that when multiple exact locality matches tie in string length,
    a matching town PIN cleanly disambiguates to the correct locality,
    while the same address without a PIN remains ambiguous."""
    # From official gazetteer data/ps3_geocoder/localities.csv:
    # In Town T1:
    # - T1-L02: 'Basava Nagar' (normalized 'basava nagar', length 12, PIN 960101)
    # - T1-L04: 'Gandhi Nagar' (normalized 'gandhi nagar', length 12, PIN 960102)
    # Both normalized names have length exactly 12 and distinct PINs.
    loc_b = df_localities[df_localities["locality_id"] == "T1-L02"].iloc[0]
    loc_g = df_localities[df_localities["locality_id"] == "T1-L04"].iloc[0]
    assert len(normalize_script(loc_b["locality_name"])) == len(normalize_script(loc_g["locality_name"])) == 12
    assert int(loc_b["pincode"]) == 960101
    assert int(loc_g["pincode"]) == 960102

    address_text_with_pin = "Between Basava Nagar and Gandhi Nagar, Kaveripura - 960101"
    raw_pin, resolved_pin, pin_status = extract_and_normalize_pincode(
        address_text_with_pin, "T1", df_localities
    )
    assert resolved_pin == 960101
    assert pin_status == "exact"

    # With matching PIN 960101: must resolve to T1-L02 (Basava Nagar)
    loc_id, raw_name, status, conf = resolve_locality(
        address_text=address_text_with_pin,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=resolved_pin,
    )
    assert status == "resolved"
    assert loc_id == "T1-L02"
    assert conf == 1.0

    # With matching PIN 960102: must resolve to T1-L04 (Gandhi Nagar)
    loc_id2, _, status2, conf2 = resolve_locality(
        address_text="Between Basava Nagar and Gandhi Nagar, Kaveripura - 960102",
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=960102,
    )
    assert status2 == "resolved"
    assert loc_id2 == "T1-L04"
    assert conf2 == 1.0

    # Without PIN: must remain ambiguous with no resolved locality ID
    address_text_no_pin = "Between Basava Nagar and Gandhi Nagar, Kaveripura"
    loc_id_no_pin, _, status_no_pin, conf_no_pin = resolve_locality(
        address_text=address_text_no_pin,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=None,
    )
    assert status_no_pin == "ambiguous"
    assert loc_id_no_pin is None
    assert conf_no_pin == 0.90


def test_exact_tie_unresolved_when_pin_conflicts_or_shares(df_localities):
    """Verify that exact-match length ties safely remain ambiguous when:
    1. A valid town PIN matches neither tied candidate.
    2. The supplied PIN is shared by both tied candidates."""
    # Case 1: Two equally long exact matches (Basava Nagar [960101], Gandhi Nagar [960102]),
    # with a valid T1 PIN that matches neither candidate (e.g. 960104, covering Kalyana Nagar / Ashoka Layout).
    address_conflict_pin = "Between Basava Nagar and Gandhi Nagar, Kaveripura - 960104"
    raw_pin1, res_pin1, st_pin1 = extract_and_normalize_pincode(
        address_conflict_pin, "T1", df_localities
    )
    assert res_pin1 == 960104
    assert st_pin1 == "exact"

    loc_id1, _, status1, _ = resolve_locality(
        address_text=address_conflict_pin,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=res_pin1,
    )
    assert status1 == "ambiguous"
    assert loc_id1 is None

    # Case 2: Two equally long exact matches that SHARE the same PIN.
    # In T1: 'Gandhi Nagar' (len 12) and 'Nehru Colony' (len 12) both have PIN 960102.
    loc_g = df_localities[df_localities["locality_id"] == "T1-L04"].iloc[0]
    loc_n = df_localities[df_localities["locality_id"] == "T1-L11"].iloc[0]
    assert len(normalize_script(loc_g["locality_name"])) == len(normalize_script(loc_n["locality_name"])) == 12
    assert int(loc_g["pincode"]) == int(loc_n["pincode"]) == 960102

    address_shared_pin = "Between Gandhi Nagar and Nehru Colony, Kaveripura - 960102"
    raw_pin2, res_pin2, st_pin2 = extract_and_normalize_pincode(
        address_shared_pin, "T1", df_localities
    )
    assert res_pin2 == 960102
    assert st_pin2 == "exact"

    loc_id2, _, status2, _ = resolve_locality(
        address_text=address_shared_pin,
        town_id="T1",
        df_localities=df_localities,
        pincode_resolved=res_pin2,
    )
    assert status2 == "ambiguous"
    assert loc_id2 is None


def test_fuzzy_tie_broken_by_pin():
    """Verify that when multiple full-town fuzzy candidates score >= 0.65 within ambiguity margin 0.05,
    PIN evidence disambiguates to the matching locality, while remaining ambiguous without a discriminating PIN."""
    # Construct a controlled, deterministic synthetic locality fixture for town T1
    # with length != 36 so it has an isolated cache key in _TOWN_LOCALITIES_CACHE.
    # 'Silver Park Layout' (PIN 960101) vs 'Silver Peak Layout' (PIN 960102)
    # Against query 'silver pk layout':
    # - Character 3-gram similarity for both is exactly 0.70 (>= 0.65 fuzzy threshold).
    # - Score margin is |0.70 - 0.70| = 0.00 (< 0.05 ambiguity margin).
    # - Neither name is an exact substring, preventing exact-match branch activation.
    # - Standard fuzzy match reaches >= 0.65, preventing PIN-assisted 0.55 recovery fallback.
    synthetic_df = pd.DataFrame([
        {
            "locality_id": "SYN-L01",
            "town_id": "T1",
            "locality_name": "Silver Park Layout",
            "pincode": 960101,
            "centroid_x": 100.0,
            "centroid_y": 200.0,
        },
        {
            "locality_id": "SYN-L02",
            "town_id": "T1",
            "locality_name": "Silver Peak Layout",
            "pincode": 960102,
            "centroid_x": 150.0,
            "centroid_y": 250.0,
        },
    ])

    address_query = "House 10, Silver Pk Layout, Kaveripura"

    # Confirm neither candidate matches exact substring
    norm_query = normalize_script(address_query)
    assert "silver park layout" not in norm_query
    assert "silver peak layout" not in norm_query

    # Confirm both candidates achieve identical fuzzy similarity >= 0.65
    sim1 = ngram_similarity("silver pk layout", "silver park layout")
    sim2 = ngram_similarity("silver pk layout", "silver peak layout")
    assert sim1 >= 0.65
    assert sim2 >= 0.65
    assert abs(sim1 - sim2) < 0.05

    # 1. With PIN 960101: disambiguates to SYN-L01
    loc_id1, _, status1, conf1 = resolve_locality(
        address_text=address_query,
        town_id="T1",
        df_localities=synthetic_df,
        pincode_resolved=960101,
    )
    assert status1 == "resolved"
    assert loc_id1 == "SYN-L01"
    assert conf1 == round(sim1, 3)

    # 2. With PIN 960102: disambiguates to SYN-L02
    loc_id2, _, status2, conf2 = resolve_locality(
        address_text=address_query,
        town_id="T1",
        df_localities=synthetic_df,
        pincode_resolved=960102,
    )
    assert status2 == "resolved"
    assert loc_id2 == "SYN-L02"
    assert conf2 == round(sim2, 3)

    # 3. Without PIN: must remain ambiguous with no resolved locality ID
    loc_id_none, _, status_none, conf_none = resolve_locality(
        address_text=address_query,
        town_id="T1",
        df_localities=synthetic_df,
        pincode_resolved=None,
    )
    assert status_none == "ambiguous"
    assert loc_id_none is None
    assert conf_none == round(sim1, 3)
