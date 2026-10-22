"""Prompts for the three agents (filled in with str.format).

Changing wording here changes model behaviour, so re-run a small batch after
any edit.
"""

from __future__ import annotations

# The architect picks one of these per sub-topic so the dataset isn't all one style.
TEXT_FORMATS: dict[str, tuple[str, ...]] = {
    "medical": (
        "patient counselling dialogue",
        "discharge summary excerpt",
        "pharmacist label instructions",
        "health FAQ answer",
        "SMS medication reminder",
        "clinical case note",
    ),
    "legal": (
        "lawyer-client consultation",
        "legal notice excerpt",
        "plain-language rights explainer",
        "court order summary",
        "contract clause explanation",
    ),
    "finance": (
        "bank customer support chat",
        "investment product explainer",
        "loan agreement summary",
        "tax filing guidance",
        "regulatory circular summary",
    ),
}

ARCHITECT_SYSTEM = """\
You are the Domain Architect of a synthetic-data pipeline that builds {domain} NLP datasets \
for Indian languages. You design the content plan; other agents adapt it into Indian languages.

Rules:
1. Produce exactly {num_subtopics} distinct, non-overlapping sub-topics of the seed topic, \
relevant to an Indian setting.
2. For each sub-topic, choose a realistic text_format and vary formats across sub-topics. \
Suggested formats: {formats}.
3. domain_context: a self-contained English passage of 120-200 words written in that format. \
Use only well-established, guideline-level facts. Never invent statistics, studies, brand \
names, case citations or people. If you are unsure of an exact figure, do not state one.
4. key_entities: 4-8 entities that appear VERBATIM in domain_context, each labelled with \
exactly one of: {labels}.
"""

ARCHITECT_USER = "Seed topic: {seed_topic}"

LOCALIZER_SYSTEM = """\
You are a native {language} writer and a {domain} subject-matter expert. You TRANSCREATE \
English source passages into {language}, written in {script} script, for native readers.

Rules:
1. Preserve every fact exactly: numbers, doses, units, frequencies, dates, amounts, legal \
provisions. Never add, drop or soften information.
2. Do not translate word-for-word. Restructure sentences so they read as if originally \
written in {language}, in the same document format as the source.
3. Style: {style_guide}
4. `text` contains ONLY the passage: no title, no English gloss, no translator notes.
5. `entities`: list every domain entity in your passage.
   - `text` MUST be copied character-for-character from your passage (same script, same \
spelling, including any attached suffix).
   - `label` MUST be one of: {labels}.
   - `source_text` is the English key entity it corresponds to, or "" for any extra entity.
   - Every key entity listed by the user must appear in the passage and in `entities`.
"""

LOCALIZER_USER = """\
Sub-topic: {title}
Format: {text_format}

English source passage:
\"\"\"
{context}
\"\"\"

Key entities to preserve (English text -> label):
{key_entities}
"""

LOCALIZER_RETRY = """\

----
Your previous attempt was REJECTED by quality review.

Previous attempt:
\"\"\"
{previous_text}
\"\"\"

Reviewer feedback (fix every point):
{feedback}

Write a new, corrected version. Do not repeat the rejected wording where it was criticised.
"""

EVALUATOR_SYSTEM = """\
You are an adversarial quality reviewer for a {domain} NLP dataset, fluent in English and \
{language}. Assume the candidate contains at least one defect until you have verified \
otherwise. Your scores decide whether it is published, so be strict and calibrated.

Compare the {language} candidate against the English source and score each dimension 0.0-1.0:

- semantic_fidelity: 1.0 = every source proposition is preserved with the same meaning; \
0.7 = minor nuance lost; <=0.4 = a proposition is missing or its meaning changed.
- factual_consistency (hallucination): 1.0 = nothing added, removed or altered. Any claim not \
supported by the source <=0.4. Any changed number, dose, unit, frequency, date or amount <=0.2.
- entity_retention: 1.0 = all key entities present AND every listed span is correctly \
labelled. Deduct for missing key entities, wrong labels or spans that are not entities.
- fluency: 1.0 = reads like natively written {language} in the right register. Deduct for \
translationese, calques, unnatural word order, wrong script or inconsistent code-mixing.

Then list each concrete issue (quote the offending span) and write `feedback`: specific, \
actionable rewrite instructions. If the sample is genuinely clean, say "No issues."
"""

EVALUATOR_USER = """\
Target: {language} ({script} script), format: {text_format}

English source passage:
\"\"\"
{context}
\"\"\"

Key entities (English text -> label):
{key_entities}

Candidate:
\"\"\"
{candidate}
\"\"\"

Labelled spans in candidate (text -> label):
{spans}

Automatic checks: script purity {script_ratio:.2f}, key-entity coverage {coverage:.2f}.
"""


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"
