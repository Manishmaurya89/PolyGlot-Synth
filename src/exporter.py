"""Writing run outputs to disk and publishing them to the Hugging Face Hub.

JSONL is written as plain UTF-8 (no \\u escapes) so the Indic text is readable
in the files and in the Hub viewer.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.schemas.dataset import (
    DOMAIN_ENTITY_LABELS,
    DatasetBatch,
    QuarantinedSample,
    SyntheticSample,
)

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class ExportError(RuntimeError):
    pass


def _atomic_write_text(path: Path, content: str) -> None:
    # Write to a temp file and rename it, so nobody ever reads a half-written file.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write one record per line and return how many were written."""
    lines = [record.model_dump_json() for record in records]
    _atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return len(lines)


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    """Load a JSONL file, validating every line. Errors name the file and line."""
    records: list[M] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise ExportError(f"{path}:{lineno}: invalid {model.__name__}: {exc}") from exc
    return records


@dataclass(frozen=True, slots=True)
class ExportPaths:
    verified: Path
    quarantine: Path
    manifest: Path


class DatasetExporter:
    """Writes verified/, quarantine/ and manifests/ files for a run."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def write_run(
        self,
        batch: DatasetBatch,
        quarantined: Sequence[QuarantinedSample],
        report: dict[str, Any],
    ) -> ExportPaths:
        paths = ExportPaths(
            verified=self.output_dir / "verified" / f"{batch.batch_id}.jsonl",
            quarantine=self.output_dir / "quarantine" / f"{batch.batch_id}.jsonl",
            manifest=self.output_dir / "manifests" / f"{batch.batch_id}.json",
        )
        try:
            write_jsonl(paths.verified, batch.samples)
            write_jsonl(paths.quarantine, quarantined)
            manifest = {
                **batch.model_dump(mode="json", exclude={"samples"}),
                "quarantined_samples": len(quarantined),
                "files": {k: str(v) for k, v in asdict(paths).items()},
                "report": report,
            }
            _atomic_write_text(
                paths.manifest, json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n"
            )
        except OSError as exc:
            raise ExportError(f"Could not write outputs to {self.output_dir}: {exc}") from exc
        logger.info("Wrote %d verified / %d quarantined samples", batch.total_samples, len(quarantined))
        return paths


def samples_frame(samples: Sequence[SyntheticSample]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": s.id,
                "language": s.target_language,
                "script": s.script.value,
                "domain": s.domain.value,
                "subtopic": s.provenance.subtopic if s.provenance else None,
                "quality_score": s.quality_score,
                "retry_count": s.provenance.retry_count if s.provenance else None,
                "num_entities": len(s.entities),
                "num_chars": len(s.raw_text),
            }
            for s in samples
        ]
    )


def language_summary(samples: Sequence[SyntheticSample]) -> pd.DataFrame:
    """Score/retry stats per language. Empty if there are no samples."""
    df = samples_frame(samples)
    if df.empty:
        return df
    return (
        df.groupby("language")
        .agg(
            samples=("id", "size"),
            mean_score=("quality_score", "mean"),
            min_score=("quality_score", "min"),
            mean_retries=("retry_count", "mean"),
            mean_entities=("num_entities", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _size_category(n: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return label
    return "100K<n<1M" if n < 1_000_000 else "1M<n<10M"


def _yaml_list(key: str, values: Iterable[str], indent: str = "") -> list[str]:
    # JSON strings are valid YAML scalars, which sidesteps quoting edge cases.
    return [f"{indent}{key}:", *(f"{indent}- {json.dumps(v, ensure_ascii=False)}" for v in values)]


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _dataset_card(
    samples: Sequence[SyntheticSample], pretty_name: str, license_id: str, repo_id: str
) -> str:
    languages = sorted({s.target_language for s in samples})
    iso_codes = {code.split("-")[0] for code in languages}
    if "hi-Latn" in languages:
        iso_codes.add("en")  # Hinglish is Hindi-English code-mixed
    domains = sorted({s.domain.value for s in samples})
    labels = sorted({label for d in {s.domain for s in samples} for label in DOMAIN_ENTITY_LABELS[d]})
    summary = language_summary(samples)

    yaml: list[str] = ["---", f"pretty_name: {json.dumps(pretty_name)}", f"license: {license_id}"]
    yaml += _yaml_list("language", sorted(iso_codes))
    yaml += _yaml_list("task_categories", ["token-classification", "text-generation"])
    yaml += _yaml_list("tags", ["synthetic", "multilingual", "indic", *domains])
    yaml += _yaml_list("size_categories", [_size_category(len(samples))])
    yaml += [
        "configs:",
        "- config_name: all",
        "  default: true",
        "  data_files:",
        "  - split: train",
        '    path: "data/*/train.jsonl"',
    ]
    for code in languages:
        yaml += [
            f"- config_name: {json.dumps(code)}",
            "  data_files:",
            "  - split: train",
            f'    path: "data/{code}/train.jsonl"',
        ]
    yaml.append("---")

    body = f"""
# {pretty_name}

Synthetic, domain-specific ({", ".join(domains)}) text in Indian languages with
character-level entity annotations, generated by the **PolyGlot-Crew** multi-agent
pipeline (Domain Architect -> Localization Adapter -> Adversarial Evaluator).

Every sample passed deterministic checks (script purity, exact entity span
alignment, label taxonomy, key-entity coverage, numeric fidelity) and an LLM critic scoring semantic
fidelity, factual consistency, entity retention and fluency.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", "all")  # every language
one = load_dataset("{repo_id}", "{languages[0]}")  # a single language config
```

## Statistics

{_markdown_table(summary)}

## Fields

| field | description |
|---|---|
| `raw_text` | NFC-normalised passage |
| `entities` | list of `{{text, label, start_char, end_char}}`; `raw_text[start_char:end_char] == text` |
| `target_language` / `script` | BCP-47 tag / ISO 15924 script code |
| `quality_score`, `quality_breakdown` | final score and per-dimension scores (0-1) |
| `provenance` | seed topic, sub-topic, English source passage, generator/evaluator models, retries |

Entity labels: {", ".join(f"`{label}`" for label in labels)}

## Limitations

This is **synthetic** data produced by language models. It may contain factual
errors and must not be used as medical, legal or financial advice. Validate on
human-annotated data before drawing conclusions from models trained on it.
"""
    return "\n".join(yaml) + "\n" + body


def build_hf_dataset(
    jsonl_files: Sequence[Path],
    out_dir: Path,
    *,
    pretty_name: str = "PolyGlot-Crew Synthetic Indic Domain Data",
    license_id: str = "cc-by-4.0",
    repo_id: str = "<username>/<dataset-name>",
) -> Path:
    """Merge verified JSONL files into a folder ready to upload to the Hub.

    Produces README.md (the dataset card) and data/<lang>/train.jsonl. Records
    are re-validated and de-duplicated by id. The folder is rebuilt every time,
    so it always matches the inputs exactly.
    """
    if not jsonl_files:
        raise ExportError("No verified JSONL files to publish")
    by_id: dict[str, SyntheticSample] = {}
    for path in jsonl_files:
        for sample in read_jsonl(Path(path), SyntheticSample):
            by_id.setdefault(sample.id, sample)
    samples = list(by_id.values())
    if not samples:
        raise ExportError("Input files contain no samples")

    out_dir = Path(out_dir)
    grouped: dict[str, list[SyntheticSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target_language].append(sample)
    for stale in (out_dir / "data").glob("*/train.jsonl") if (out_dir / "data").exists() else ():
        stale.unlink()
    for code, items in grouped.items():
        write_jsonl(out_dir / "data" / code / "train.jsonl", items)
    _atomic_write_text(out_dir / "README.md", _dataset_card(samples, pretty_name, license_id, repo_id))
    logger.info("Built HF dataset folder %s (%d samples, %d languages)", out_dir, len(samples), len(grouped))
    return out_dir


def push_to_hub(folder: Path, repo_id: str, *, token: str | None, private: bool = True) -> str:
    """Upload the folder as a dataset repo (created if needed) and return its URL."""
    from huggingface_hub import HfApi

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(folder),
            commit_message="Update PolyGlot-Crew synthetic dataset",
            delete_patterns=["data/**"],  # drop files on the Hub that are no longer in the folder
        )
    except Exception as exc:
        raise ExportError(f"Upload to {repo_id} failed: {exc}") from exc
    return f"https://huggingface.co/datasets/{repo_id}"
