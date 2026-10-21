from __future__ import annotations

import pytest

from src.linguistics import align_entities, nfc, script_ratio
from src.schemas.dataset import Script
from tests.fakes import HINDI_TEXT


def test_script_ratio_tolerates_latin_units_in_hindi() -> None:
    ratio = script_ratio(HINDI_TEXT, Script.DEVANAGARI)
    assert 0.9 < ratio < 1.0
    assert script_ratio(HINDI_TEXT, Script.LATIN) == pytest.approx(1.0 - ratio)


def test_script_ratio_ignores_digits_and_punctuation() -> None:
    assert script_ratio("500 mg, 3x!", Script.LATIN) == 1.0
    assert script_ratio("123 ...", Script.TELUGU) == 0.0


def test_alignment_produces_exact_offsets() -> None:
    result = align_entities(HINDI_TEXT, [("पैरासिटामोल", "DRUG"), ("500 mg", "DOSAGE")])
    assert not result.unaligned
    assert [e.label for e in result.entities] == ["DRUG", "DOSAGE"]
    for entity in result.entities:
        assert HINDI_TEXT[entity.start_char : entity.end_char] == entity.text


def test_every_occurrence_is_labelled() -> None:
    text = "Take paracetamol now. Paracetamol again at night."
    result = align_entities(text, [("paracetamol", "DRUG")])
    assert [(e.text, e.start_char) for e in result.entities] == [
        ("paracetamol", 5),
        ("Paracetamol", 22),
    ]


def test_longest_span_wins_and_nested_mention_is_not_an_error() -> None:
    text = "Give paracetamol 500 mg."
    result = align_entities(text, [("paracetamol", "DRUG"), ("paracetamol 500 mg", "DRUG")])
    assert [e.text for e in result.entities] == ["paracetamol 500 mg"]
    assert not result.unaligned


def test_mentions_are_nfc_normalised_before_matching() -> None:
    text = nfc("मरीज़ को दवा दें")
    result = align_entities(text, [("मरीज़", "POPULATION")])  # precomposed ZA in mention
    assert len(result.entities) == 1
    assert result.entities[0].text == text[:5]


def test_matches_must_respect_word_boundaries() -> None:
    result = align_entities("Dose is 5 mgmt units", [("mg", "DOSAGE")])
    assert not result.entities
    assert result.unaligned == [("mg", "DOSAGE")]


def test_agglutinative_suffix_allowed_only_when_configured() -> None:
    text = "రోగి పారాసిటమాల్‌ను తీసుకోవాలి"  # paracetamol + accusative suffix
    strict = align_entities(text, [("పారాసిటమాల్", "DRUG")], strict_right_boundary=True)
    relaxed = align_entities(text, [("పారాసిటమాల్", "DRUG")], strict_right_boundary=False)
    assert not strict.entities
    assert [e.text for e in relaxed.entities] == ["పారాసిటమాల్"]


def test_left_boundary_is_always_enforced() -> None:
    result = align_entities("अपैरासिटामोल", [("पैरासिटामोल", "DRUG")], strict_right_boundary=False)
    assert not result.entities


def test_numbers_are_compared_across_numeral_systems() -> None:
    from src.linguistics import extract_numbers

    assert extract_numbers("२५ मिलीग्राम, दिन में 2 बार") == {"25", "2"}
    assert extract_numbers("౫౦౦ mg") == {"500"}
    assert extract_numbers("Rs. 1,00,000 at 7.5% for 12 months.") == {"100000", "7.5", "12"}
