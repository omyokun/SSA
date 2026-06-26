#!/usr/bin/env python3
"""Download FineWeb-Edu text from Hugging Face into a local text file.

The default output is one document per line:

    data/fineweb_edu_text_document.txt

That file can then be tokenized into:

    data/fineweb_edu_text_document.bin
    data/fineweb_edu_text_document.idx
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalize_document(text: str) -> str:
    """Keep one source document on one output line."""
    return " ".join(text.split()).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFaceFW/fineweb-edu text")
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--config",
        default="sample-10BT",
        help="Dataset config/name. Use an empty string to omit it.",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--output",
        default="data/fineweb_edu_text_document.txt",
        help="Output text file, one document per line",
    )
    parser.add_argument("--text-field", default="text", help="Dataset field containing text")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50_000,
        help="Maximum documents to write. Use 0 or a negative value for no limit.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream from Hugging Face instead of downloading the full Arrow cache first",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets` "
            "or run inside an environment that already provides Hugging Face datasets."
        ) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.config:
        dataset_kwargs["name"] = args.config

    logger.info(
        "Loading dataset=%s config=%s split=%s streaming=%s",
        args.dataset,
        args.config or "<none>",
        args.split,
        args.streaming,
    )
    dataset = load_dataset(**dataset_kwargs)

    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None
    written = 0
    skipped = 0

    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(args.text_field)
            if not isinstance(text, str):
                skipped += 1
                continue

            text = normalize_document(text)
            if not text:
                skipped += 1
                continue

            handle.write(text)
            handle.write("\n")
            written += 1

            if written % 10_000 == 0:
                logger.info("Wrote %d documents to %s", written, output)

            if max_docs is not None and written >= max_docs:
                break

    logger.info("Done: wrote=%d skipped=%d output=%s", written, skipped, output)


if __name__ == "__main__":
    main()
