# LLM Fine-Tuning with LoRA and Custom Optimization Functions

A guide to fine-tuning small LLMs using LoRA (Low-Rank Adaptation) with interchangeable optimization objectives.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [LoRA Configuration](#lora-configuration)
4. [Optimization Functions](#optimization-functions)
5. [Training Pipeline](#training-pipeline)
6. [Usage Examples](#usage-examples)
7. [Choosing an Optimizer](#choosing-an-optimizer)

---

## Overview

This project fine-tunes a small LLM (e.g. Llama 3.2 1B, Phi-3 Mini, Qwen2.5 0.5B) using LoRA adapters with a pluggable optimization function. LoRA freezes the base model weights and trains small low-rank matrices, reducing trainable parameters by ~99% while preserving model quality.

**Key idea:** separate the adapter architecture (LoRA) from the optimization objective so you can swap loss functions or optimizers independently.

```
Base Model (frozen)
      │
      ▼
 LoRA Adapters  ◄── trained by your custom optimizer / loss
      │
      ▼
   Output
```

---

## Setup

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

Optional optimizers:
```bash
pip install lion-pytorch          # Lion optimizer
pip install galore-torch          # GaLore (gradient low-rank projection)
pip install schedulefree          # Schedule-Free AdamW
```

---

## LoRA Configuration

LoRA injects trainable rank-decomposition matrices into the attention layers of the model. Only these matrices are updated during training.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more params
    lora_alpha=32,                 # scaling factor (effective lr = alpha / r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: ~4M || all params: ~1B || trainable: ~0.4%
```

### Rank Selection Guide

| `r` value | Use case |
|-----------|----------|
| 4–8       | Simple tasks, very limited VRAM |
| 16        | General fine-tuning (recommended default) |
| 32–64     | Complex tasks needing more expressive adapters |
| 128+      | Near full fine-tune quality, higher cost |

### QLoRA (4-bit quantization + LoRA)

For further memory reduction:

```python
from transformers import BitsAndBytesConfig
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

---

## Optimization Functions

### Loss Functions

#### 1. Standard Cross-Entropy (baseline)

Default causal language modeling objective. Minimizes negative log-likelihood over next-token predictions.

```python
import torch.nn.functional as F

def cross_entropy_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
```

**When to use:** general instruction following, text generation.

---

#### 2. Focal Loss

Downweights easy (well-predicted) tokens and focuses learning on hard ones. Useful when the dataset has class imbalance or common tokens dominate the loss.

```python
def focal_loss(logits, labels, gamma=2.0):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()

    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1).clamp(0)).squeeze(-1)
    token_probs = probs.gather(-1, shift_labels.unsqueeze(-1).clamp(0)).squeeze(-1)

    mask = shift_labels != -100
    focal_weight = (1 - token_probs) ** gamma
    loss = -(focal_weight * token_log_probs * mask).sum() / mask.sum()
    return loss
```

**When to use:** datasets with rare but important tokens; domain-specific vocabulary.

---

#### 3. DPO Loss (Direct Preference Optimization)

Trains the model to prefer chosen responses over rejected ones without a separate reward model.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,           # KL regularization strength against reference model
    loss_type="sigmoid", # or "hinge", "ipo", "kto_pair"
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

trainer = DPOTrainer(
    model=peft_model,
    ref_model=None,   # None = use base model as implicit reference with LoRA
    args=dpo_config,
    train_dataset=dataset,  # must have "prompt", "chosen", "rejected" columns
)
```

**When to use:** aligning model outputs to human preferences; RLHF-style training without a reward model.

---

#### 4. Contrastive Loss (InfoNCE)

Pulls representations of semantically similar inputs together and pushes dissimilar ones apart. Used for embedding / retrieval fine-tuning.

```python
import torch

def infonce_loss(anchor, positives, negatives, temperature=0.07):
    # anchor, positives, negatives: [batch, hidden_dim]
    anchor = F.normalize(anchor, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (anchor * positives).sum(dim=-1) / temperature
    neg_sim = (anchor @ negatives.T) / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
```

**When to use:** semantic similarity, retrieval-augmented generation (RAG) fine-tuning.

---

#### 5. Custom Reward-Weighted Loss

Scales the cross-entropy loss by a per-sample reward signal. Useful when you have a scoring function (rule-based, another model, or human ratings).

```python
def reward_weighted_loss(logits, labels, rewards):
    # rewards: [batch] — scalar score per sample
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    mask = shift_labels != -100
    per_sample_loss = (per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    weights = F.softmax(rewards, dim=0) * rewards.size(0)  # normalize
    return (weights * per_sample_loss).mean()
```

**When to use:** when you can score outputs programmatically (e.g. code correctness, math accuracy, format compliance).

---

### Optimizers

#### AdamW (default)

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)
```

---

#### Lion

Uses sign gradient updates. ~3x more memory-efficient than Adam. Often converges faster on LLMs.

```python
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,       # use ~3–10x smaller lr than AdamW
    weight_decay=0.01,
)
```

---

#### Adafactor

Near-zero optimizer memory overhead. Good when VRAM is the bottleneck.

```python
from transformers.optimization import Adafactor

optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # uses internal schedule when relative_step=True
)
```

---

#### Schedule-Free AdamW

Removes the need for a learning rate scheduler. Tracks a weighted average of iterates internally.

```python
from schedulefree import AdamWScheduleFree

optimizer = AdamWScheduleFree(model.parameters(), lr=2e-4)

# Must call these at training/eval boundaries
optimizer.train()
# ... training ...
optimizer.eval()
```

---

#### GaLore (Gradient Low-Rank Projection)

Projects gradients into a low-rank subspace before the optimizer step. Reduces optimizer state memory by 5–8x vs AdamW, enabling full-parameter training at LoRA-level cost.

```python
from galore_torch import GaLoreAdamW

param_groups = [
    {"params": [p for n, p in model.named_parameters() if "attn" in n],
     "rank": 128, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
    {"params": [p for n, p in model.named_parameters() if "attn" not in n]},
]

optimizer = GaLoreAdamW(param_groups, lr=2e-4)
```

---

## Training Pipeline

### Full Custom Trainer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import torch

# 1. Load model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"   # or "microsoft/phi-3-mini-4k-instruct", "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(base_model, lora_config)

# 3. Custom Trainer with swappable loss
class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = focal_loss(outputs.logits, labels, gamma=2.0)   # swap loss here
        return (loss, outputs) if return_outputs else loss

# 4. Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 5. Train
trainer = CustomTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.optimizer = Lion(model.parameters(), lr=1e-4)  # swap optimizer here
trainer.train()

# 6. Save adapter weights only
model.save_pretrained("./lora-adapter")
```

---

## Usage Examples

### Inference with saved adapter

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
model.eval()

inputs = tokenizer("Explain quantum entanglement:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merging adapter into base model

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

---

## Choosing an Optimizer

| Situation | Recommended choice |
|-----------|-------------------|
| General fine-tuning, enough VRAM | AdamW |
| VRAM constrained, want speed | Lion |
| Extremely tight memory budget | Adafactor |
| Want to skip LR scheduling | Schedule-Free AdamW |
| Full-param training at LoRA cost | GaLore |
| Human preference alignment | DPO (via `trl`) |
| Programmatic reward signal | Reward-weighted CE loss |
| Imbalanced / domain-specific vocab | Focal loss |
