"""Deterministic text utilities: Unicode normalisation, script detection and
entity span alignment.

LLMs are unreliable at counting characters, so the pipeline never asks them for
offsets. The Localization Adapter returns entity *surface strings*;
:func:`align_entities` locates them in the passage and produces exact
code-point spans. This is the step that makes the published ``start_char`` /
``end_char`` values trustworthy.
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


def is_word_char(ch: str) -> bool:
    """True if ``ch`` can sit *inside* a word in any supported script."""
    return ch in _ZERO_WIDTH_JOINERS or unicodedata.category(ch)[0] in ("L", "M", "N")


def script_ratio(text: str, script: Script) -> float:
    """Share of letters/marks in ``text`` that belong to ``script``.

    Digits, punctuation and whitespace are ignored. Returns ``0.0`` for text
    without any letters.
    """
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
    """Numeric values in ``text``, canonicalised for cross-script comparison.

    Native decimal digits (Devanagari ``२५``, Telugu ``౨౫``, ...) are mapped to
    ASCII and thousands separators are dropped (``1,00,000`` -> ``100000``), so
    a dose survives the check whichever numeral system the localizer used.
    """
    ascii_digits = "".join(
        str(unicodedata.digit(ch)) if unicodedata.category(ch) == "Nd" else ch for ch in text
    )
    return {match.replace(",", "") for match in _NUMBER.findall(ascii_digits)}


@dataclass(slots=True)
class AlignmentResult:
    """Outcome of :func:`align_entities`."""

    entities: list[EntityLabel] = field(default_factory=list)
    """Aligned, sorted, non-overlapping spans."""
    unaligned: list[tuple[str, str]] = field(default_factory=list)
    """``(surface, label)`` pairs that do not occur in the text at all."""
    aligned_surfaces: set[str] = field(default_factory=set)
    """NFC surfaces that matched at least once (used for entity coverage)."""


def _has_valid_boundaries(text: str, start: int, end: int, strict_right: bool) -> bool:
    """Reject matches that start (or, if ``strict_right``, end) mid-word."""
    if start > 0 and is_word_char(text[start - 1]) and is_word_char(text[start]):
        return False
    return not (
        strict_right and end < len(text) and is_word_char(text[end]) and is_word_char(text[end - 1])
    )


def align_entities(
    text: str,
    mentions: Iterable[tuple[str, str]],
    *,
    strict_right_boundary: bool = True,
) -> AlignmentResult:
    """Compute exact character spans for entity surface forms.

    Policy:

    * Inputs are NFC-normalised; ``text`` is expected to be NFC already.
    * **Every** non-overlapping occurrence of a mention is labelled. Leaving a
      repeated entity unlabelled would teach an NER model false negatives.
    * Longer surfaces are placed first, so ``"paracetamol 500 mg"`` wins over a
      nested ``"paracetamol"``; a mention whose occurrences are all covered by
      longer spans is treated as subsumed, not as an error.
    * Latin-script matching is case-insensitive; the stored entity text is
      always the exact substring of ``text``.
    * A mention that never occurs (respecting word boundaries) is reported in
      ``unaligned`` so the Evaluator can reject the draft with precise feedback.

    Args:
        text: NFC-normalised passage.
        mentions: ``(surface, label)`` pairs as returned by the localizer.
        strict_right_boundary: if ``False``, a match may end mid-word, which
            lets a stem match an inflected form in agglutinative languages.
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
