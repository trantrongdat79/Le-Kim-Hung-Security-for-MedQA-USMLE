# Le-Kim-Hung-Security-for-MedQA-USMLE

Medical Multi-Agent LLM project for answering MedQA-USMLE multiple-choice questions.

Current stage: Sprint 1, Medical Reasoning Agent.

## 1. Requirements

- Python 3.12
- `uv`
- OpenRouter API key for the Sprint 1 LLM
- MedQA-USMLE dataset files under `data/MedQA-USMLE-4-options/`

## 2. Setup

Install dependencies with `uv`:

```bash
uv sync
```

If you prefer pip-style installation:

```bash
pip install -r requirements.txt
```

`uv` is recommended because the project uses `pyproject.toml` and `uv.lock`.

## 3. Environment Variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Fill in the LLM config:

```env
LLM_1_MODEL=deepseek/deepseek-v4-flash
LLM_1_API_BASE=https://openrouter.ai/api/v1
LLM_1_API_KEY=your_openrouter_api_key_here
```

## 4. Dataset

The project expects the Hugging Face dataset here:

```text
data/MedQA-USMLE-4-options/
  phrases_no_exclude_train.jsonl
  phrases_no_exclude_test.jsonl
```

If you need to download it again:

```bash
uv run hf download GBaker/MedQA-USMLE-4-options \
  --repo-type dataset \
  --local-dir data/MedQA-USMLE-4-options
```

Run a quick dataset loading check:

```bash
uv run python -c "from src.core.dataset import load_medqa_split; print(load_medqa_split('train', limit=1)[0])"
```

For the Sprint 2 RAG corpus, download the MedRAG textbooks dataset into
`data/corpus/medrag_textbooks/`:

```bash
uv run python scripts/download_medrag_textbooks.py
```

This writes `data/corpus/medrag_textbooks/textbooks.jsonl` and
`data/corpus/medrag_textbooks/metadata.json`. 

## 5. Run Sprint 1 Evaluation (LLM + Prompt)

Run the Medical Reasoning Agent on a small data subset:

```bash
uv run python src/eval/run_med_agent_eval.py --split train --limit 10 --offset 0
uv run python src/eval/run_med_agent_eval.py --split test --limit 5 --offset 0
```

The script prints:

```
# uv run python src/eval/run_med_agent_eval.py --split test --limit 5 --offset 0

[1/5] gold=B pred=B correct=True latency=7.48s
[2/5] gold=D pred=D correct=True latency=4.33s
[3/5] gold=B pred=B correct=True latency=3.44s
[4/5] gold=D pred=A correct=False latency=18.38s
[5/5] gold=B pred=B correct=True latency=4.42s

Sprint 1 Medical Reasoning Agent Evaluation
split=test
offset=0
limit=5
total=5
correct=4
invalid=0
accuracy=0.8000
invalid_response_rate=0.0000
latency_avg_seconds=7.61
latency_min_seconds=3.44
latency_max_seconds=18.38
```

## 6. Run Sprint 2 Evaluation (RAG)

Run data ingestion into chromadb (might take 4-8 hours):
```bash
uv run python src/retrieval/ingest_data.py
```

Run the Sprint 2 RAG agent on a small data subset:

```bash
uv run python src/eval/run_rag_eval.py --split test --limit 5 --offset 0 --top-k 5
uv run python src/eval/run_med_agent_eval.py --split test --limit 5 --offset 0
```

## 7. Run Sprint 3 Evaluation (LangGraph RAG Workflow)

Run the LangGraph-orchestrated RAG workflow on a small data subset:

```bash
uv run python src/eval/run_langgraph_rag_eval.py --split test --limit 5 --offset 0 --top-k 5
```

Compare it with the Sprint 2 RAG agent on the same subset:

```bash
uv run python src/eval/run_rag_eval.py --split test --limit 5 --offset 0 --top-k 5
```
