# 1. LLM
- We choose openrouter as a central usage hub. We can easily change to 9router or orther endpoints by adjusting the .env file.
- We choose deepseek/deepseek-v4-flash as our base model for this phase since: We do not target high accuracy, we target pipeline implementation and versaltility. Deepseek-v4-flash is cheap ($0.077 / $0.154 per 1M token) -> meet our requirements for this phase.
- Other potential LLM:
    - Mid Tier: deepseek/deepseek-v4-pro, anthropic/claude-sonnet-5, google/gemini-2.5-flash, openai/gpt-5.4
    - Low Tier: openai/gpt-oss-120b, openai/gpt-4o-mini, anthropic/claude-haiku-4.5, google/gemini-2.5-flash-lite

# 2. Embedding Model
- We currently use baai/bge-m3 (0.01 / 1M tokens) because of its small size which can be deployed locally
- If we don't need local deployment, we can use qwen/qwen3-embedding-8b with the same price
- openai/text-embedding-3-small (0.02 / 1M tokens) is a potential candidate for testing

# 3. Embedding Data
- https://huggingface.co/datasets/MedRAG/textbooks
- 18 widely used medical textbooks
- Publication of the dataset: Benchmarking Retrieval-Augmented Generation for Medicine. Published in Findings of the Association for Computational Linguistics: ACL 2024