"""PolyGlot-Crew command-line entry point.

Runs one generation batch end to end:

    seed topic -> Domain Architect -> (sub-topic x language) workers in parallel
    -> Localization Adapter <-> Adversarial Evaluator loop -> verified / quarantined
    -> JSONL + manifest on disk -> (optional) Hugging Face Hub

Examples:
    python main.py
    python main.py --seed-topic "Pediatric antibiotic dosing" --languages hindi tamil
    python main.py --provider anthropic --num-subtopics 5 --max-concurrency 8
    python main.py --push-to-hub your-username/polyglot-medical-indic

Exit codes: 0 success, 1 configuration/input error, 2 pipeline failure, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.config import ConfigError, Provider, Settings
from src.exporter import DatasetExporter, ExportError, build_hf_dataset, language_summary, push_to_hub
from src.graph import PipelineError, build_pipeline, initial_state
from src.llm import LLMCallError
from src.schemas.dataset import DatasetBatch, Domain, QuarantinedSample
from src.schemas.languages import LANGUAGES
from src.telemetry import UsageTracker

logger = logging.getLogger("polyglot_crew")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a verified synthetic multilingual domain dataset batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed-topic", default="Clinical Drug Dosage Diagnostics")
    parser.add_argument("--domain", choices=[d.value for d in Domain], default=Domain.MEDICAL.value)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hindi", "hinglish", "telugu"],
        metavar="LANG",
        help=f"Registry keys or BCP-47 codes. Available: {', '.join(LANGUAGES)}",
    )
    parser.add_argument("--num-subtopics", type=int, default=3)
    parser.add_argument("--provider", choices=[p.value for p in Provider], help="Overrides LLM_PROVIDER.")
    parser.add_argument("--max-concurrency", type=int, help="Overrides MAX_CONCURRENCY.")
    parser.add_argument("--output-dir", type=Path, help="Overrides OUTPUT_DIR.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--push-to-hub",
        nargs="?",
        const="",
        metavar="REPO_ID",
        help="After the run, publish all verified batches. Uses HF_REPO_ID when no value is given.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep HTTP client chatter out of the per-call token logs.
    for noisy in ("httpx", "httpcore", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_report(
    batch: DatasetBatch,
    quarantined: list[QuarantinedSample],
    tracker: UsageTracker,
    wall_time_s: float,
    planned: int,
) -> dict[str, Any]:
    """Machine-readable run statistics (also stored in the manifest)."""
    retries = [s.provenance.retry_count for s in batch.samples if s.provenance]
    reasons: dict[str, int] = {}
    for q in quarantined:
        reasons[q.reason.value] = reasons.get(q.reason.value, 0) + 1
    finished = batch.total_samples + len(quarantined)
    return {
        "planned_samples": planned,
        "verified": batch.total_samples,
        "quarantined": len(quarantined),
        "quarantine_reasons": reasons,
        "acceptance_rate": round(batch.total_samples / finished, 4) if finished else 0.0,
        "first_pass_acceptance": round(retries.count(0) / finished, 4) if finished else 0.0,
        "mean_retries_verified": round(sum(retries) / len(retries), 3) if retries else 0.0,
        "wall_time_s": round(wall_time_s, 2),
        "samples_per_minute": round(finished / wall_time_s * 60, 2) if wall_time_s else 0.0,
        "llm": tracker.totals(),
    }


def print_report(
    report: dict[str, Any],
    batch: DatasetBatch,
    tracker: UsageTracker,
    outputs: dict[str, str],
) -> None:
    rule = "=" * 78
    llm = report["llm"]
    lines = [
        "",
        rule,
        f" PolyGlot-Crew run report  |  batch {batch.batch_id}",
        rule,
        f" Seed topic : {batch.seed_topic}  ({batch.domain})",
        f" Samples    : {report['planned_samples']} planned | {report['verified']} verified | "
        f"{report['quarantined']} quarantined {report['quarantine_reasons'] or ''}",
        f" Acceptance : {report['acceptance_rate']:.1%} overall | "
        f"{report['first_pass_acceptance']:.1%} first-pass | "
        f"{report['mean_retries_verified']} mean retries (verified)",
        f" Throughput : {report['wall_time_s']}s wall | {report['samples_per_minute']} samples/min",
        f" LLM usage  : {llm['calls']} calls ({llm['failed_calls']} failed) | "
        f"{llm['input_tokens']:,} in + {llm['output_tokens']:,} out tokens | "
        f"{llm['llm_time_s']}s model time",
        "",
        " Usage by role:",
    ]
    usage = tracker.summary_frame()
    lines.append(usage.to_string(index=False) if not usage.empty else " (no LLM calls recorded)")
    summary = language_summary(batch.samples)
    lines += ["", " Quality by language:"]
    lines.append(summary.to_string(index=False) if not summary.empty else " (no verified samples)")
    lines += ["", " Outputs:", *(f"   {name:<10} -> {target}" for name, target in outputs.items()), rule]
    print("\n".join(lines))


def publish(settings: Settings, repo_id: str) -> str:
    """Build a Hub folder from every verified batch on disk and upload it."""
    verified = sorted((settings.output_dir / "verified").glob("*.jsonl"))
    folder = build_hf_dataset(verified, settings.output_dir / "hf_dataset", repo_id=repo_id)
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    return push_to_hub(folder, repo_id, token=token, private=settings.hf_private)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        settings = Settings.from_env(
            args.env_file,
            provider=args.provider,
            max_concurrency=args.max_concurrency,
            output_dir=args.output_dir,
        )
        state = initial_state(args.seed_topic, args.domain, args.languages, args.num_subtopics)
    except (ConfigError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    repo_id = None
    if args.push_to_hub is not None:
        repo_id = args.push_to_hub or settings.hf_repo_id
        if not repo_id:
            logger.error("--push-to-hub needs a REPO_ID argument or HF_REPO_ID in the environment")
            return 1

    tracker = UsageTracker()
    try:
        pipeline = build_pipeline(settings, tracker)
        logger.info(
            "Starting run: %r | %s | languages=%s | up to %d samples | concurrency=%d",
            state["seed_topic"], state["domain"], ",".join(state["language_keys"]),
            args.num_subtopics * len(state["language_keys"]),
            settings.max_concurrency,
        )
        started = time.perf_counter()
        final = pipeline.invoke(state, config={"max_concurrency": settings.max_concurrency})
        wall_time = time.perf_counter() - started
    except (PipelineError, LLMCallError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130

    batch: DatasetBatch = final["batch"]
    quarantined: list[QuarantinedSample] = final.get("quarantined_samples", [])
    planned = len(final.get("blueprint_subtopics", [])) * len(state["language_keys"])
    report = build_report(batch, quarantined, tracker, wall_time, planned)

    try:
        paths = DatasetExporter(settings.output_dir).write_run(batch, quarantined, report)
    except ExportError as exc:
        logger.error("%s", exc)
        return 2
    outputs = {
        "verified": str(paths.verified),
        "quarantine": str(paths.quarantine),
        "manifest": str(paths.manifest),
    }

    if repo_id:
        try:
            outputs["hub"] = publish(settings, repo_id)
        except ExportError as exc:
            logger.error("%s", exc)
            print_report(report, batch, tracker, outputs)
            return 2

    print_report(report, batch, tracker, outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
