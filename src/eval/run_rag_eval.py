"""Run a small Sprint 2 evaluation for the RAGMedicalReasoningAgent."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.rag_agent import RAGMedicalReasoningAgent
from src.core.dataset import load_medqa_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Sprint 2 RAGMedicalReasoningAgent on MedQA."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to evaluate. Defaults to train.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of examples to evaluate. Defaults to 5.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of examples to skip before evaluation. Defaults to 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve per question. Defaults to 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    examples = load_medqa_split(args.split, limit=args.limit, offset=args.offset)
    if not examples:
        raise SystemExit("No examples loaded. Check --split, --limit, and --offset.")

    agent = RAGMedicalReasoningAgent(top_k=args.top_k)

    correct = 0
    invalid = 0
    latencies: list[float] = []

    for index, example in enumerate(examples, start=1):
        start = time.perf_counter()
        prediction = agent.answer(example)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        is_correct = prediction.is_valid and prediction.answer_idx == example.answer_idx
        if is_correct:
            correct += 1
        if not prediction.is_valid:
            invalid += 1

        print(
            f"[{index}/{len(examples)}] "
            f"gold={example.answer_idx} "
            f"pred={prediction.answer_idx or 'INVALID'} "
            f"correct={is_correct} "
            f"latency={elapsed:.2f}s"
        )
        if not prediction.is_valid and prediction.error:
            print(f"  invalid_reason={prediction.error}")

    total = len(examples)
    accuracy = correct / total
    invalid_rate = invalid / total

    print()
    print("Sprint 2 RAG Medical Reasoning Agent Evaluation")
    print(f"split={args.split}")
    print(f"offset={args.offset}")
    print(f"limit={args.limit}")
    print(f"top_k={args.top_k}")
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"invalid={invalid}")
    print(f"accuracy={accuracy:.4f}")
    print(f"invalid_response_rate={invalid_rate:.4f}")
    print(f"latency_avg_seconds={mean(latencies):.2f}")
    print(f"latency_min_seconds={min(latencies):.2f}")
    print(f"latency_max_seconds={max(latencies):.2f}")


if __name__ == "__main__":
    main()
