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
scripts/
  download_medrag_textbooks.py   # Download MedRAG textbooks corpus for RAG

src/
  agent/
    med_agent.py                # Sprint 1 direct LLM baseline
    rag_agent.py                # Sprint 2 RAG-only agent
    langgraph_rag_agent.py      # Sprint 3 LangGraph RAG workflow
  core/
    config.py                   # Environment-based runtime config
    dataset.py                  # MedQA JSONL loading
    schema.py                   # Shared data containers and result schemas
  eval/
    run_med_agent_eval.py       # Sprint 1 evaluation runner
    run_rag_eval.py             # Sprint 2 RAG evaluation runner
    run_langgraph_rag_eval.py   # Sprint 3 LangGraph RAG evaluation runner
  retrieval/
    ingest_data.py              # Embed corpus and persist Chroma index
    ingest_data_smoke_test.py   # Embedding/ingestion smoke test
    retrieval_test.py           # Manual retrieval quality check
    retriever.py                # Reusable Chroma-backed evidence retriever

data/
  MedQA-USMLE-4-options/        # MedQA train/test JSONL files
  corpus/medrag_textbooks/      # Downloaded MedRAG textbooks JSONL corpus
  chroma/                       # Persisted Chroma vector indexes
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

### Sprint 3: LangGraph RAG Workflow

- Implement a LangGraph-orchestrated version of the Sprint 2 RAG flow.
- Keep the same retriever, prompt, LLM, parser, and evaluator metrics as Sprint 2.
- Use a simple two-node graph:
  - `retrieve`: retrieve evidence from the Chroma-backed medical textbook index.
  - `answer`: answer with the existing RAG answer-generation logic.
- Purpose: demonstrate how the retriever plugs into LangGraph before the full multi-agent workflow.

### Sprint 4: Multi-agent system without memory (Placeholders)
### Sprint 5: Full system (V2 + Short-term memory + Long-term memory) (Placeholders)
