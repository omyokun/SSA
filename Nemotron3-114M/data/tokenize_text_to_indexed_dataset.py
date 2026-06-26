#!/usr/bin/env python3
"""Tokenize plain text into Megatron/NeMo indexed dataset files."""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_indexed_dataset_builder(output_bin_path: str):
    def instantiate(builder_cls):
        try:
            return builder_cls(output_bin_path)
        except TypeError:
            import numpy as np

            return builder_cls(output_bin_path, dtype=np.int32)

    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

        return instantiate(IndexedDatasetBuilder)
    except ImportError:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder

        return instantiate(MMapIndexedDatasetBuilder)


def encode_text(tokenizer, text: str):
    if hasattr(tokenizer, "text_to_ids"):
        return tokenizer.text_to_ids(text)
    return tokenizer.encode(text)


def build_indexed_dataset(
    input_path: str,
    tokenizer_name: str,
    output_prefix: str,
    max_tokens: int | None,
    append_eos: bool,
    log_every: int,
):
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
    builder = get_indexed_dataset_builder(f"{output_prefix}.bin")

    token_count = 0
    lines_processed = 0

    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue

            ids = list(encode_text(tokenizer, text))
            if append_eos and eos_id is not None:
                ids = ids + [eos_id]

            if max_tokens is not None:
                remaining = max_tokens - token_count
                if remaining <= 0:
                    break
                ids = ids[:remaining]

            if not ids:
                break

            token_count += len(ids)
            builder.add_item(torch.tensor(ids, dtype=torch.int64))
            if hasattr(builder, "end_document"):
                builder.end_document()

            lines_processed += 1
            if log_every > 0 and lines_processed % log_every == 0:
                logger.info(
                    "Processed lines=%d tokens=%d output_prefix=%s",
                    lines_processed,
                    token_count,
                    output_prefix,
                )

            if max_tokens is not None and token_count >= max_tokens:
                break

    builder.finalize(f"{output_prefix}.idx")
    logger.info(
        "Wrote indexed dataset: prefix=%s lines=%d tokens=%d",
        output_prefix,
        lines_processed,
        token_count,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize plain text into Megatron indexed dataset format"
    )
    parser.add_argument(
        "--input-txt",
        default="data/fineweb_edu_text_document.txt",
        help="Plain text input, one document per line",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/luciole_50k",
        help="Tokenizer path or name",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/fineweb_edu_text_document",
        help="Output prefix without .bin/.idx",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10_000_000_000,
        help="Maximum tokens to write. Use 0 or a negative value for no limit.",
    )
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append tokenizer EOS token after each input line when available",
    )
    parser.add_argument("--log-every", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    build_indexed_dataset(
        input_path=args.input_txt,
        tokenizer_name=args.tokenizer,
        output_prefix=args.output_prefix,
        max_tokens=max_tokens,
        append_eos=args.append_eos,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
