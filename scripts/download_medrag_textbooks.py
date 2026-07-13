"""Download the MedRAG textbooks corpus into data/corpus.

The Hugging Face dataset is already chunked into short snippets with
id/title/content fields, which makes JSONL a convenient local format for
later LlamaIndex ingestion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DEFAULT_DATASET = "MedRAG/textbooks"
DEFAULT_OUTPUT_DIR = Path("data/corpus/medrag_textbooks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MedRAG/textbooks and export it as local JSONL."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset name. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download. Defaults to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the full dataset through the Hugging Face cache before export.",
    )
    return parser.parse_args()


def iter_rows(
    dataset_name: str,
    split: str,
    *,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in dataset:
        yield dict(row)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(row.get("title"))
    content = clean_text(row.get("content"))
    contents = clean_text(row.get("contents")) or f"{title}. {content}".strip()
    return {
        "id": clean_text(row.get("id")),
        "title": title,
        "content": content,
        "contents": contents,
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "textbooks.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    streaming = not args.no_streaming

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in iter_rows(args.dataset, args.split, streaming=streaming):
            if args.limit is not None and count >= args.limit:
                break
            record = normalize_row(row)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    metadata = {
        "dataset": args.dataset,
        "source_url": f"https://huggingface.co/datasets/{args.dataset}",
        "split": args.split,
        "row_count": count,
        "output_file": str(output_path),
        "streaming": streaming,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {count} rows to {output_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
