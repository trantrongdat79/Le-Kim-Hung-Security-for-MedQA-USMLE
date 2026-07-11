"""Dataset loading utilities for the MedQA-USMLE JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.core.schema import MedQAExample

DATASET_DIR = Path("data/MedQA-USMLE-4-options")
SPLIT_FILES = {
    "train": DATASET_DIR / "phrases_no_exclude_train.jsonl",
    "test": DATASET_DIR / "phrases_no_exclude_test.jsonl",
}


def get_split_path(split: str) -> Path:
    try:
        return SPLIT_FILES[split]
    except KeyError as exc:
        valid = ", ".join(sorted(SPLIT_FILES))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {valid}") from exc


def iter_jsonl_examples(path: Path) -> Iterable[MedQAExample]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield MedQAExample.from_dict(json.loads(stripped))
            except Exception as exc:
                raise ValueError(f"Failed to parse {path}:{line_number}") from exc


def load_medqa_split(
    split: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[MedQAExample]:
    path = get_split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split file not found: {path}")
        
    examples: list[MedQAExample] = []
    for index, example in enumerate(iter_jsonl_examples(path)):
        if index < offset:
            continue
        if limit is not None and len(examples) >= limit:
            break
        examples.append(example)
    return examples
