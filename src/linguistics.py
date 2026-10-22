"""Text helpers: Unicode normalisation, script checks and entity span alignment.

We never ask the LLM for character offsets because it gets them wrong. The
localizer returns entity strings and align_entities() finds them in the text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.schemas.dataset import EntityLabel, Script

# Unicode blocks per script (inclusive ranges). Latin covers Basic Latin letters
# plus Latin-1 Supplement / Extended-A/B, enough for romanised Indic text.
SCRIPT_RANGES: dict[Script, tuple[tuple[int, int], ...]] = {
    Script.DEVANAGARI: ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    Script.BENGALI: ((0x0980, 0x09FF),),
    Script.GURMUKHI: ((0x0A00, 0x0A7F),),
    Script.GUJARATI: ((0x0A80, 0x0AFF),),
    Script.TAMIL: ((0x0B80, 0x0BFF),),
    Script.TELUGU: ((0x0C00, 0x0C7F),),
    Script.KANNADA: ((0x0C80, 0x0CFF),),
    Script.MALAYALAM: ((0x0D00, 0x0D7F),),
    Script.LATIN: ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
}

_ZERO_WIDTH_JOINERS = frozenset({"\u200c", "\u200d"})  # ZWNJ / ZWJ occur inside Indic words
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def nfc(text: str) -> str:
    """Return the NFC-normalised form of ``text``."""
    return unicodedata.normalize("NFC", text)


def _is_letter_or_mark(ch: str) -> bool:
    # Indic vowel signs and viramas are category M*, not L*, but are part of words.
    return unicodedata.category(ch)[0] in ("L", "M")


def _char_kind(ch: str) -> str | None:
    if ch in _ZERO_WIDTH_JOINERS or _is_letter_or_mark(ch):
        return "letter"
    if unicodedata.category(ch)[0] == "N":
        return "digit"
    return None


def _same_word(a: str, b: str) -> bool:
    # Letters and digits next to each other don't form one word, so "mg" in
    # "500mg" still counts as a whole word.
    kind = _char_kind(a)
    return kind is not None and kind == _char_kind(b)


def script_ratio(text: str, script: Script) -> float:
    """Fraction of letters in ``text`` that are in ``script`` (0.0 if no letters)."""
    ranges = SCRIPT_RANGES[script]
    total = in_script = 0
    for ch in text:
        if not _is_letter_or_mark(ch):
            continue
        total += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in ranges):
            in_script += 1
    return in_script / total if total else 0.0


def extract_numbers(text: str) -> set[str]:
    """All numbers in ``text`` as ASCII strings, so "२५" and "25" compare equal.

    Thousands separators are dropped too ("1,00,000" -> "100000").
    """
    ascii_digits = "".join(
        str(unicodedata.digit(ch)) if unicodedata.category(ch) == "Nd" else ch for ch in text
    )
    return {match.replace(",", "") for match in _NUMBER.findall(ascii_digits)}


@dataclass(slots=True)
class AlignmentResult:
    entities: list[EntityLabel] = field(default_factory=list)  # sorted, non-overlapping
    unaligned: list[tuple[str, str]] = field(default_factory=list)  # (surface, label) never found
    aligned_surfaces: set[str] = field(default_factory=set)  # surfaces that matched at least once


def _has_valid_boundaries(text: str, start: int, end: int, strict_right: bool) -> bool:
    """Reject matches that start (or, if ``strict_right``, end) mid-word."""
    if start > 0 and _same_word(text[start - 1], text[start]):
        return False
    return not (strict_right and end < len(text) and _same_word(text[end - 1], text[end]))


def contains_term(text: str, term: str, *, strict_right_boundary: bool = True) -> bool:
    """Case-insensitive whole-word search, using the same rules as alignment."""
    term = nfc(term).strip()
    if not term:
        return False
    return any(
        _has_valid_boundaries(text, m.start(), m.end(), strict_right_boundary)
        for m in re.finditer(re.escape(term), text, flags=re.IGNORECASE)
    )


def align_entities(
    text: str,
    mentions: Iterable[tuple[str, str]],
    *,
    strict_right_boundary: bool = True,
) -> AlignmentResult:
    """Find character spans in ``text`` (already NFC) for each (surface, label).

    Every occurrence gets labelled, not just the first, otherwise an NER model
    trained on the data learns that repeats aren't entities. Longer mentions are
    placed first, so "paracetamol 500 mg" wins over a nested "paracetamol" and
    the shorter one isn't reported as missing. Matching ignores case and
    respects word boundaries; anything never found goes into ``unaligned``.

    Pass ``strict_right_boundary=False`` for Telugu, Tamil etc., where a case
    suffix is glued straight onto the entity.
    """
    result = AlignmentResult()
    unique: dict[tuple[str, str], None] = {}
    for surface, label in mentions:
        surface = nfc(surface).strip()
        if surface:
            unique.setdefault((surface, label), None)

    occupied: list[tuple[int, int]] = []
    for surface, label in sorted(unique, key=lambda m: len(m[0]), reverse=True):
        found = False
        for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE):
            start, end = match.span()
            if not _has_valid_boundaries(text, start, end, strict_right_boundary):
                continue
            found = True
            if any(start < o_end and o_start < end for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            result.entities.append(
                EntityLabel(text=text[start:end], label=label, start_char=start, end_char=end)
            )
        if found:
            result.aligned_surfaces.add(surface)
        else:
            result.unaligned.append((surface, label))

    result.entities.sort(key=lambda e: e.start_char)
    return result
