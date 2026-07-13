"""Query the local Chroma RAG index to verify retrieval quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import chromadb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.ingest_data import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    embed_batch,
    load_embedding_config,
)


DEFAULT_QUERY = (
    "Lactose-fermenting gram-negative rods forming pink colonies on MacConkey agar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a query and retrieve nearest chunks from Chroma."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query text to retrieve evidence for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return. Defaults to 5.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"Persistent Chroma directory. Defaults to {DEFAULT_CHROMA_DIR}.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Defaults to {DEFAULT_COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=700,
        help="Maximum document preview characters per result. Defaults to 700.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    collection = client.get_collection(name=args.collection)
    count = collection.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{args.collection}' is empty. Run ingest_data.py first."
        )

    embedding_config = load_embedding_config()
    query_embeddings, usage = embed_batch([args.query], config=embedding_config)
    query_embedding = query_embeddings[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(args.top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    print("Retrieval test")
    print(f"chroma_dir={args.chroma_dir}")
    print(f"collection={args.collection}")
    print(f"collection_count={count}")
    print(f"query={args.query}")
    print(f"embedding_dimension={len(query_embedding)}")
    if usage:
        print(f"usage={usage}")
    print()

    print_results(result, preview_chars=args.preview_chars)


def print_results(result: dict[str, Any], *, preview_chars: int) -> None:
    ids = first_query_values(result, "ids")
    documents = first_query_values(result, "documents")
    metadatas = first_query_values(result, "metadatas")
    distances = first_query_values(result, "distances")

    for index, chunk_id in enumerate(ids, start=1):
        document = documents[index - 1]
        metadata = metadatas[index - 1] or {}
        distance = distances[index - 1]
        preview = compact_preview(document, preview_chars)

        print(f"[{index}] id={chunk_id}")
        print(f"    distance={distance:.6f}")
        print(f"    title={metadata.get('title', '')}")
        print(f"    row_index={metadata.get('row_index', '')}")
        print(f"    preview={preview}")
        print()


def first_query_values(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key) or [[]]
    return values[0]


def compact_preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
