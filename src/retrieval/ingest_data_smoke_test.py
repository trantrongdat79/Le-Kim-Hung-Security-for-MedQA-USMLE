"""Smoke test the configured embedding model.

This script only verifies that the embedding endpoint works before the full
RAG ingestion pipeline is implemented in ingest_data.py.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv


DEFAULT_TEXTS = [
    "Lactose-fermenting gram-negative rods form pink colonies on MacConkey agar.",
    "Escherichia coli commonly causes urinary tract infections.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test OpenRouter-compatible embedding configuration."
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Text to embed. Pass multiple times to test batch embedding.",
    )
    return parser.parse_args()


def load_embedding_config() -> tuple[str, str, str]:
    load_dotenv()

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    api_base = os.getenv("EMBEDDING_MODEL_API_BASE", "").strip()
    api_key = os.getenv("EMBEDDING_MODEL_API_KEY", "").strip()

    missing = [
        name
        for name, value in {
            "EMBEDDING_MODEL": model,
            "EMBEDDING_MODEL_API_BASE": api_base,
            "EMBEDDING_MODEL_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    return model, api_base.rstrip("/"), api_key


def request_embeddings(
    *,
    api_base: str,
    api_key: str,
    model: str,
    texts: list[str],
) -> dict[str, Any]:
    url = f"{api_base}/embeddings"
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc


def main() -> None:
    args = parse_args()
    texts = args.texts or DEFAULT_TEXTS
    model, api_base, api_key = load_embedding_config()

    response = request_embeddings(
        api_base=api_base,
        api_key=api_key,
        model=model,
        texts=texts,
    )

    embeddings = [item["embedding"] for item in response.get("data", [])]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embedding(s), got {len(embeddings)}: {response}"
        )

    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) != 1:
        raise RuntimeError(f"Embeddings have inconsistent dimensions: {dimensions}")

    print("Embedding smoke test passed")
    print(f"model={response.get('model', model)}")
    print(f"input_count={len(texts)}")
    print(f"embedding_dimension={dimensions.pop()}")
    if usage := response.get("usage"):
        print(f"usage={json.dumps(usage, sort_keys=True)}")


if __name__ == "__main__":
    main()
