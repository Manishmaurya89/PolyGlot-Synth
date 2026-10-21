from __future__ import annotations

from pathlib import Path

import pytest

from src.exporter import DatasetExporter, ExportError, build_hf_dataset, read_jsonl
from src.schemas.dataset import DatasetBatch, EntityLabel, SyntheticSample


def make_sample(language: str = "hi", text: str = "पैरासिटामोल 500 mg") -> SyntheticSample:
    script = "Latn" if language == "hi-Latn" else "Deva"
    return SyntheticSample(
        target_language=language,
        script=script,
        raw_text=text,
        entities=[EntityLabel(text="500 mg", label="DOSAGE", start_char=len(text) - 6, end_char=len(text))],
        domain="medical",
        quality_score=0.9,
    )


def test_write_run_produces_readable_utf8_jsonl(tmp_path: Path) -> None:
    batch = DatasetBatch.from_samples([make_sample(), make_sample()], seed_topic="t", domain="medical")
    paths = DatasetExporter(tmp_path).write_run(batch, [], {"verified": 2})

    raw = paths.verified.read_text(encoding="utf-8")
    assert "पैरासिटामोल" in raw and "\\u" not in raw
    assert read_jsonl(paths.verified, SyntheticSample) == batch.samples
    assert paths.quarantine.read_text() == ""
    assert '"report"' in paths.manifest.read_text()


def test_read_jsonl_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(make_sample().model_dump_json() + "\n{\"id\": 1}\n", encoding="utf-8")
    with pytest.raises(ExportError, match=r"bad\.jsonl:2"):
        read_jsonl(path, SyntheticSample)


def test_build_hf_dataset_layout_and_dedup(tmp_path: Path) -> None:
    hi, hinglish = make_sample(), make_sample("hi-Latn", "paracetamol 500 mg")
    exporter = DatasetExporter(tmp_path / "runs")
    first = exporter.write_run(DatasetBatch.from_samples([hi]), [], {}).verified
    second = exporter.write_run(DatasetBatch.from_samples([hi, hinglish]), [], {}).verified

    out = build_hf_dataset([first, second], tmp_path / "hf", repo_id="me/demo")

    assert len(read_jsonl(out / "data" / "hi" / "train.jsonl", SyntheticSample)) == 1
    assert len(read_jsonl(out / "data" / "hi-Latn" / "train.jsonl", SyntheticSample)) == 1
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    assert 'config_name: "hi-Latn"' in card and 'path: "data/*/train.jsonl"' in card
    assert '- "en"' in card  # Hinglish adds English to the language tags
    assert 'load_dataset("me/demo", "all")' in card


def test_build_hf_dataset_requires_input(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        build_hf_dataset([], tmp_path)
