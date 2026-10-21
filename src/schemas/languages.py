"""Registry of target languages and the linguistic rules attached to each.

Adding a language is a one-entry change here: give it a BCP-47 tag, its script,
a style guide the Localization Adapter follows, and the script-purity floor the
Evaluator enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.dataset import Script


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """Static description of one target language variety."""

    key: str
    """CLI-friendly identifier, e.g. ``"hinglish"``."""
    code: str
    """BCP-47 tag written to ``SyntheticSample.target_language``."""
    name: str
    script: Script
    style_guide: str
    """Register / code-mixing guidance injected into the localization prompt."""
    min_script_ratio: float = 0.6
    """Minimum share of letters that must be in ``script``. Below 1.0 on purpose:
    Indian technical prose legitimately keeps some Latin tokens (``mg``, ``GST``)."""
    strict_right_boundary: bool = True
    """``False`` for agglutinative languages, where case suffixes attach directly
    to an entity (Telugu ``పారాసిటమాల్‌ను`` = paracetamol + accusative), so an entity
    span may legitimately end mid-word."""


_INDIC_COMMON = (
    "Use Western Arabic numerals (0-9) for every quantity, dose, date and amount. "
    "Keep SI units and standard abbreviations (mg, ml, kg, %, Rs., GST, EMI) as commonly written."
)

LANGUAGES: dict[str, LanguageSpec] = {
    spec.key: spec
    for spec in (
        LanguageSpec(
            key="hindi",
            code="hi",
            name="Hindi",
            script=Script.DEVANAGARI,
            style_guide=(
                "Write contemporary standard Hindi as an Indian professional would explain it to an "
                "educated layperson. Prefer words people actually use (डॉक्टर, दवा, बैंक खाता) over "
                "heavily Sanskritised coinages. Transliterate international technical terms and drug "
                f"names into Devanagari when that is the common practice. {_INDIC_COMMON}"
            ),
        ),
        LanguageSpec(
            key="hinglish",
            code="hi-Latn",
            name="Hinglish (Romanised Hindi-English code-mix)",
            script=Script.LATIN,
            min_script_ratio=0.95,
            style_guide=(
                "Write Hindi grammar and function words in Roman script (hai, ke liye, karna chahiye) "
                "mixed with English technical nouns, the way urban Indians text on WhatsApp. Use common "
                "informal spellings consistently (nahi, kya, dawai, paise). Never use Devanagari. "
                f"{_INDIC_COMMON}"
            ),
        ),
        LanguageSpec(
            key="marathi",
            code="mr",
            name="Marathi",
            script=Script.DEVANAGARI,
            style_guide=f"Write natural standard Marathi as used in Maharashtra. {_INDIC_COMMON}",
        ),
        LanguageSpec(
            key="bengali",
            code="bn",
            name="Bengali",
            script=Script.BENGALI,
            style_guide=f"Write natural standard colloquial Bengali (চলিত ভাষা). {_INDIC_COMMON}",
        ),
        LanguageSpec(
            key="gujarati",
            code="gu",
            name="Gujarati",
            script=Script.GUJARATI,
            style_guide=f"Write natural standard Gujarati. {_INDIC_COMMON}",
        ),
        LanguageSpec(
            key="punjabi",
            code="pa",
            name="Punjabi",
            script=Script.GURMUKHI,
            style_guide=f"Write natural Punjabi in Gurmukhi script as used in Indian Punjab. {_INDIC_COMMON}",
        ),
        LanguageSpec(
            key="telugu",
            code="te",
            name="Telugu",
            script=Script.TELUGU,
            strict_right_boundary=False,
            style_guide=(
                "Write modern spoken-standard Telugu (వ్యావహారిక భాష), not archaic literary Telugu. "
                "Transliterate technical terms into Telugu script when that is common practice. "
                f"{_INDIC_COMMON}"
            ),
        ),
        LanguageSpec(
            key="tamil",
            code="ta",
            name="Tamil",
            script=Script.TAMIL,
            strict_right_boundary=False,
            style_guide=(
                "Write clear modern written Tamil. Avoid overly pure (தனித்தமிழ்) coinages where a "
                f"transliterated technical term is standard. {_INDIC_COMMON}"
            ),
        ),
        LanguageSpec(
            key="kannada",
            code="kn",
            name="Kannada",
            script=Script.KANNADA,
            strict_right_boundary=False,
            style_guide=f"Write natural modern standard Kannada. {_INDIC_COMMON}",
        ),
        LanguageSpec(
            key="malayalam",
            code="ml",
            name="Malayalam",
            script=Script.MALAYALAM,
            strict_right_boundary=False,
            style_guide=f"Write natural modern standard Malayalam. {_INDIC_COMMON}",
        ),
    )
}


def get_language(key_or_code: str) -> LanguageSpec:
    """Resolve a language by registry key (``"telugu"``) or BCP-47 code (``"te"``).

    Raises:
        ValueError: if the language is not registered.
    """
    needle = key_or_code.strip().lower()
    for spec in LANGUAGES.values():
        if needle in (spec.key, spec.code.lower()):
            return spec
    raise ValueError(
        f"Unknown language {key_or_code!r}. Available: {', '.join(sorted(LANGUAGES))}"
    )
