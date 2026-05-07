# Brain — LLM Fine-Tuning

A toolkit for fine-tuning small LLMs locally using LoRA, with support for custom loss functions, optimizers, and reward-maximisation objectives.

---

## What this repo does

- Fine-tunes small LLMs (sub-3B parameters) on a laptop or single GPU using **LoRA** adapters
- Provides a plug-and-play system for swapping **loss functions** and **optimizers** independently of the model architecture
- Supports **reward-maximisation training** — fine-tuning a model to maximise the output of a separate reward model or LLM judge
- Includes a lightweight **local inference script** for running models on Apple Silicon

---

## Quickstart

```bash
pip install torch transformers peft matplotlib

# Run local inference
python inference.py

# Fine-tune with a local LLM judge
python train.py --criteria "responses should be concise and use simple language"

# Fine-tune with your own prompts
python train.py --criteria "formal and professional tone" \
                --prompts "Explain gravity" "What is DNA?" "How do computers work?"
```

`inference.py` runs `Qwen2.5-0.5B-Instruct` locally on CPU (Apple Silicon compatible). See [`inference.py`](inference.py) for details.

`train.py` fine-tunes the policy model using a separate frozen judge model (`Qwen2.5-1.5B-Instruct`) as the reward signal. See [`docs/reward-maximization.md`](docs/reward-maximization.md) for how it works.

---

## Documentation

| Document | Description |
|----------|-------------|
| [LoRA Configuration](docs/lora-configuration.md) | How to configure LoRA adapters, rank selection, and QLoRA setup |
| [Optimization Functions](docs/optimization-functions.md) | Loss functions (cross-entropy, focal, DPO, contrastive, reward-weighted) and optimizers (AdamW, Lion, Adafactor, GaLore) |
| [Training Pipeline](docs/training-pipeline.md) | End-to-end fine-tuning example with a custom trainer |
| [Reward Maximisation](docs/reward-maximization.md) | Fine-tuning to maximise the output of another model (GRPO, PPO, DPO, reward-weighted SFT) |

---

## Supported Models

Any causal LM on HuggingFace works. Recommended models for local use:

| Model | Params | Notes |
|-------|--------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Fastest, minimal RAM |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Good quality/speed balance |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Best local quality |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Strong reasoning |
| `meta-llama/Llama-3.2-1B` | 1B | Good general baseline |
