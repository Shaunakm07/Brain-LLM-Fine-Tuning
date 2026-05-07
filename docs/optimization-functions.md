# Optimization Functions

This document covers the loss functions and optimizers available for LoRA fine-tuning.

---

## Loss Functions

### 1. Standard Cross-Entropy (baseline)

Default causal language modeling objective. Minimises negative log-likelihood over next-token predictions.

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

### 2. Focal Loss

Downweights easy (well-predicted) tokens and focuses learning on hard ones. Useful when common tokens dominate the loss.

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

### 3. DPO Loss (Direct Preference Optimization)

Trains the model to prefer chosen responses over rejected ones without a separate reward model.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,            # KL regularisation strength against reference model
    loss_type="sigmoid", # or "hinge", "ipo", "kto_pair"
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

trainer = DPOTrainer(
    model=peft_model,
    ref_model=None,      # None = use base model as implicit reference with LoRA
    args=dpo_config,
    train_dataset=dataset,  # must have "prompt", "chosen", "rejected" columns
)
```

**When to use:** aligning outputs to human preferences without a reward model.

---

### 4. Contrastive Loss (InfoNCE)

Pulls representations of similar inputs together and pushes dissimilar ones apart.

```python
import torch

def infonce_loss(anchor, positives, negatives, temperature=0.07):
    anchor    = F.normalize(anchor, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (anchor * positives).sum(dim=-1) / temperature
    neg_sim = (anchor @ negatives.T) / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
```

**When to use:** semantic similarity, RAG fine-tuning.

---

### 5. Reward-Weighted Loss

Scales cross-entropy loss by a per-sample reward signal from an external scorer.

```python
def reward_weighted_loss(logits, labels, rewards):
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
    weights = F.softmax(rewards, dim=0) * rewards.size(0)
    return (weights * per_sample_loss).mean()
```

**When to use:** when outputs can be scored programmatically (code correctness, math, format compliance).

---

## Optimizers

### AdamW (default)

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
```

---

### Lion

Sign gradient updates. ~3x more memory-efficient than Adam, often converges faster.

```python
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,       # use ~3–10x smaller lr than AdamW
    weight_decay=0.01,
)
```

```bash
pip install lion-pytorch
```

---

### Adafactor

Near-zero optimizer memory overhead. Best when VRAM is the bottleneck.

```python
from transformers.optimization import Adafactor

optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,
)
```

---

### Schedule-Free AdamW

Removes the need for a learning rate scheduler.

```python
from schedulefree import AdamWScheduleFree

optimizer = AdamWScheduleFree(model.parameters(), lr=2e-4)

optimizer.train()   # call at start of training
# ...
optimizer.eval()    # call before evaluation
```

```bash
pip install schedulefree
```

---

### GaLore

Projects gradients into a low-rank subspace. Reduces optimizer state memory by 5–8x, enabling full-parameter training at LoRA-level cost.

```python
from galore_torch import GaLoreAdamW

param_groups = [
    {"params": [p for n, p in model.named_parameters() if "attn" in n],
     "rank": 128, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
    {"params": [p for n, p in model.named_parameters() if "attn" not in n]},
]

optimizer = GaLoreAdamW(param_groups, lr=2e-4)
```

```bash
pip install galore-torch
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
