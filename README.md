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

## 5. Run Sprint 1 Evaluation

Run the Medical Reasoning Agent on a small train subset:

```bash
uv run python src/eval/run_med_agent_eval.py --split train --limit 10 --offset 0
uv run python src/eval/run_med_agent_eval.py --split test --limit 5 --offset 0
```

The script prints:

- gold answer option
- predicted answer option
- correctness
- latency per question
- total accuracy
- invalid response rate
- latency summary