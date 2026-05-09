---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - transformers
  - reward-maximization
  - advantage-weighted-sft
---

# Qwen2.5-0.5B — LLM-Judge LoRA Adapter

This is a LoRA adapter trained on top of `Qwen/Qwen2.5-0.5B-Instruct` using advantage-weighted supervised fine-tuning with a frozen `Qwen/Qwen2.5-1.5B-Instruct` judge as the reward signal.

Training code: [`train.py`](../train.py) in the [Brain-LLM-Fine-Tuning](https://github.com/shaunakm/Brain-LLM-Fine-Tuning) repo.

---

## Adapter configuration

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Dropout | 0.05 |
| Trainable parameters | ~4 M / 494 M total (~0.8%) |
| Training precision | float32 (CPU) |
| PEFT version | 0.19.1 |

---

## Training details

| Setting | Value |
|---------|-------|
| Algorithm | Advantage-Weighted SFT + KL penalty |
| Policy model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Judge model | `Qwen/Qwen2.5-1.5B-Instruct` (frozen) |
| Loss | `advantage × CE(policy, completion) + 0.5 × KL(policy ∥ base)` |
| Advantage | `(reward − mean) / std` across completions per prompt |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| Gradient clip | 1.0 |

See [`docs/reward-maximization.md`](../docs/reward-maximization.md) for a full explanation of the training algorithm.

---

## Load and use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float32,
)
model = PeftModel.from_pretrained(base, "./lora-adapter")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Explain quantum entanglement."},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Merge into base weights (optional)

Produces a single standalone model with no PEFT dependency at inference time:

```python
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

---

## Files in this directory

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | LoRA weight matrices (A and B for each target module) |
| `adapter_config.json` | LoRA configuration (rank, alpha, target modules) |
| `tokenizer.json` | Tokenizer vocabulary and merges |
| `tokenizer_config.json` | Tokenizer settings (chat template, special tokens) |
| `chat_template.jinja` | Qwen chat template used during training and inference |
| `metrics.json` | Per-step reward, advantage, loss, and KL divergence from training |
| `training_metrics.png` | 6-panel training plot |
