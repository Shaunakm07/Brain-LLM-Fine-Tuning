# LoRA Configuration

LoRA (Low-Rank Adaptation) freezes the base model weights and injects small trainable rank-decomposition matrices into the attention layers. Only these matrices are updated during training, reducing trainable parameters by ~99%.

---

## Basic Setup

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more params
    lora_alpha=32,                 # scaling factor (effective lr = alpha / r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: ~4M || all params: ~1B || trainable: ~0.4%
```

## Rank Selection Guide

| `r` value | Use case |
|-----------|----------|
| 4–8       | Simple tasks, very limited VRAM |
| 16        | General fine-tuning (recommended default) |
| 32–64     | Complex tasks needing more expressive adapters |
| 128+      | Near full fine-tune quality, higher cost |

## QLoRA (4-bit quantization + LoRA)

For further memory reduction, load the base model in 4-bit before applying LoRA:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
```

## Saving and Loading Adapters

```python
# Save only the adapter weights (a few MB, not the full model)
model.save_pretrained("./lora-adapter")

# Load adapter onto a fresh base model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Optionally merge adapter into base model for standalone deployment
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```
