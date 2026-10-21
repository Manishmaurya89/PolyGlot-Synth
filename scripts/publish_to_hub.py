"""Publish previously generated batches to the Hugging Face Hub.

Merges every ``<OUTPUT_DIR>/verified/*.jsonl`` file (or the files you pass),
re-validates and de-duplicates the samples, builds a Hub-ready folder with a
dataset card, and uploads it.

Examples:
    python scripts/publish_to_hub.py --repo-id your-username/polyglot-medical-indic
    python scripts/publish_to_hub.py --repo-id me/ds --files data/verified/batch-1.jsonl --public
    python scripts/publish_to_hub.py --dry-run   # build the folder only, no upload
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, Settings
from src.exporter import ExportError, build_hf_dataset, push_to_hub

logger = logging.getLogger("publish_to_hub")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", help="Target dataset repo. Defaults to HF_REPO_ID.")
    parser.add_argument("--files", nargs="+", type=Path, help="Verified JSONL files to publish.")
    parser.add_argument("--out-dir", type=Path, help="Staging folder (default <OUTPUT_DIR>/hf_dataset).")
    parser.add_argument("--license", default="cc-by-4.0", help="License id for the dataset card.")
    parser.add_argument("--public", action="store_true", help="Create the repo as public.")
    parser.add_argument("--dry-run", action="store_true", help="Build the folder but do not upload.")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        settings = Settings.from_env(args.env_file)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    repo_id = args.repo_id or settings.hf_repo_id
    if not repo_id and not args.dry_run:
        logger.error("Pass --repo-id or set HF_REPO_ID")
        return 1

    files = args.files or sorted((settings.output_dir / "verified").glob("*.jsonl"))
    out_dir = args.out_dir or settings.output_dir / "hf_dataset"
    try:
        folder = build_hf_dataset(
            files, out_dir, license_id=args.license, repo_id=repo_id or "<username>/<dataset-name>"
        )
        if args.dry_run:
            logger.info("Dry run: dataset folder ready at %s", folder)
            return 0
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        private = settings.hf_private and not args.public
        url = push_to_hub(folder, repo_id, token=token, private=private)
    except ExportError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Published: %s", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
