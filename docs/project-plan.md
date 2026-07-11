# PROJECT PLAN

## 1. Project Context

- Task: build a medical multi-agent LLM system for MedQA-USMLE multiple-choice questions.
- Input: one MedQA-USMLE question with four answer options.
- Output: exactly one final answer option plus a short explanation.
- Primary objective: improve factual accuracy over a direct LLM baseline.
- This project evaluates system design on a benchmark; it is not a clinical deployment system.

## 2. Requirements

### Benchmark
- Development and tuning use only 100-150 public MedQA train/dev questions.
- Official evaluation must use the full MedQA-USMLE test set.
- Final evaluation correctness is based on the selected option label (`A`, `B`, `C`, or `D`) compared with the dataset `answer_idx`.
- Every system variant must produce prediction files for reproducibility.
- Primary metric: accuracy.
- Additional metrics: invalid response rate, accuracy gain, ablation comparison, win/loss/tie analysis, cost, and latency.
- Recommended statistical checks: bootstrap confidence interval and McNemar test

### Maintainability
- Keep modules small and focused.
- Keep API secrets out of source code.
- Load runtime configuration from environment variables.
- Keep agent logic separate from evaluation logic.
- Keep RAG, memory, verifier, and multi-agent workflow separable so variants can be compared.

### Project Structure
```text
src/
  agent/
    med_agent.py              # Sprint 1 Medical Reasoning Agent
  core/
    config.py                 # Environment-based runtime config
    dataset.py                # MedQA JSONL loading
    schema.py                 # Lightweight data containers
  eval/
    run_med_agent_eval.py     # Small evaluation runner
  retrieval/
    ingest_data.py            # Reserved for Sprint 2 RAG
    baseline_rag.py           # Reserved for Sprint 2 RAG
```

## 3. Required System Variants

- V0: Direct LLM baseline (LLM + Prompt)
- V1: RAG-only system.
- V2: Multi-agent system without memory. (At least 3 agents) 
- V3: Full system (V2 + Short-term memory + Long-term memory).
- V4: Full system without verifier (optional).

## 4. Tech Stack Decisions

- Python version: `>=3.12`.
- Dependency management: `uv`.
- Environment loading: `python-dotenv` with `load_dotenv()`.
- Dataset: `GBaker/MedQA-USMLE-4-options`.
- Dataset files:
  - `data/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl`
  - `data/MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl`
- LLM integration: LlamaIndex OpenRouter package (`llama-index-llms-openrouter`).
- RAG framework for later sprints: LlamaIndex.
- Vector storage for later sprints: ChromaDB.
- External RAG corpus path for later sprints: `data/corpus/`.
- Chroma persistence path for later sprints: `data/chroma/`.

## 5. Sprint Plan

### Sprint 1: Medical Reasoning Agent

- Implement a `MedicalReasoningAgent`.
- Use `LLM_1` only.
- Sprint 1 LLM config:
  - `LLM_1_MODEL`
  - `LLM_1_API_BASE`
  - `LLM_1_API_KEY`
- Default-compatible Sprint 1 model: `deepseek/deepseek-v4-flash`.
- Format question and options deterministically.
- Request strict JSON output:

```json
{
  "answer_idx": "D",
  "answer": "Nitrofurantoin",
  "explanation": "Short explanation."
}
```

- Validate `answer_idx` as one of `A`, `B`, `C`, or `D`.
- Fill `answer` from the selected option when possible.
- Return short explanations only; do not expose full chain-of-thought.
- Provide a small evaluation runner with `--split`, `--limit`, and `--offset`.
- Evaluation compares `prediction.answer_idx` with `example.answer_idx`.
- Invalid model outputs must be counted, not crash the run.

### Sprint 2: RAG Module

- Use external medical documents or guidelines from `data/corpus/`.
- Use LlamaIndex for document loading, chunking, embeddings, Chroma vector storage, and retrieval.
- Keep final answer generation in project agent code for strict JSON output and reproducible evaluation.
- Persist vector data under `data/chroma/`.

### Sprint 3: Multi-agent system without memory (Placeholders)
### Sprint 4: Full system (V2 + Short-term memory + Long-term memory) (Placeholders)