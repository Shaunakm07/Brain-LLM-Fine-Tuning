# LoRA Fine-Tuning to Maximise Another Model's Output

This document covers how to fine-tune a small LLM with LoRA such that its outputs maximise the score assigned by a **separate reward model**. This is the core idea behind RLHF and is used in systems like DeepSeek-R1, InstructGPT, and LLM-as-judge pipelines.

---

## Table of Contents

1. [Concept](#concept)
2. [Setup](#setup)
3. [Wrapping Your Reward Model](#wrapping-your-reward-model)
4. [Method 1 — GRPO (Recommended)](#method-1--grpo-recommended)
5. [Method 2 — PPO](#method-2--ppo)
6. [Method 3 — Reward-Weighted SFT](#method-3--reward-weighted-sft)
7. [Method 4 — DPO (Preference Pairs)](#method-4--dpo-preference-pairs)
8. [Choosing a Method](#choosing-a-method)
9. [Hyperparameter Reference](#hyperparameter-reference)
10. [Common Failure Modes](#common-failure-modes)

---

## Concept

Standard fine-tuning minimises a loss over a fixed dataset. Reward-maximisation fine-tuning instead treats another model as a **live scoring function** and trains the policy model to generate outputs that receive higher scores.

```
Prompt
  │
  ▼
Policy Model (LoRA fine-tuned)  ──generates──►  Completion
                                                     │
                                                     ▼
                                            Reward Model (frozen)
                                                     │
                                                     ▼
                                               Scalar Score
                                                     │
                                    gradient flows back through policy
```

The reward model is always **frozen**. Only the LoRA adapters on the policy model are updated.

---

## Setup

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

Base imports used throughout this document:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

POLICY_MODEL_ID  = "meta-llama/Llama-3.2-1B"       # model being trained
REWARD_MODEL_ID  = "OpenAssistant/reward-model-deberta-v3-large-v2"  # scorer
```

---

## Wrapping Your Reward Model

All four methods below call a `reward_fn`. Define it once and pass it to whichever trainer you use.

### Option A — Dedicated reward model (scalar output)

```python
from transformers import pipeline

reward_pipe = pipeline(
    "text-classification",
    model=REWARD_MODEL_ID,
    device=0,
    truncation=True,
    max_length=512,
)

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    results = reward_pipe(completions)
    return [r["score"] for r in results]
```

### Option B — Another LLM as judge (GPT-4, Claude, etc.)

```python
import anthropic

client = anthropic.Anthropic()

def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    scores = []
    for prompt, completion in zip(prompts, completions):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="Score the following response from 0.0 to 1.0. Reply with only a number.",
            messages=[{"role": "user", "content": f"Prompt: {prompt}\nResponse: {completion}"}],
        )
        try:
            scores.append(float(response.content[0].text.strip()))
        except ValueError:
            scores.append(0.0)
    return scores
```

### Option C — Rule-based / programmatic reward

```python
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    scores = []
    for text in completions:
        score = 0.0
        if len(text) > 50:            score += 0.3   # penalise short outputs
        if "```" in text:             score += 0.4   # rewards code blocks
        if text.count("\n") > 2:      score += 0.3   # rewards structured output
        scores.append(score)
    return scores
```

---

## Method 1 — GRPO (Recommended)

**Group Relative Policy Optimization.** Generates a group of completions per prompt, scores all of them, and trains the policy to increase the probability of above-average completions. No value head or reference model needed.

### How it works

```
Prompt ──► generate N completions ──► score each ──► normalise within group
                                                            │
                                          advantage = (score - group_mean) / group_std
                                                            │
                                          policy loss = -log_prob * advantage  (+ KL term)
```

### Full example

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Load policy model with LoRA
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_ID, torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
policy_model = get_peft_model(base_model, lora_config)

# 2. Dataset — must have a "prompt" column
dataset = load_dataset("your-dataset")["train"]

# 3. Define reward function (see options above)
def reward_fn(completions, **kwargs):
    return reward_pipe(completions)

# 4. Configure GRPO
config = GRPOConfig(
    output_dir="./grpo-output",
    num_generations=8,           # completions per prompt to compare
    max_new_tokens=256,
    temperature=0.9,             # generation temperature
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    kl_coef=0.01,                # KL penalty weight vs reference
    bf16=True,
    logging_steps=5,
)

# 5. Train
trainer = GRPOTrainer(
    model=policy_model,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

# 6. Save LoRA adapter
policy_model.save_pretrained("./grpo-lora-adapter")
```

### Key hyperparameters

| Parameter | Effect | Recommended range |
|-----------|--------|-------------------|
| `num_generations` | More = better advantage estimates, more VRAM | 4–16 |
| `kl_coef` | Higher = stay closer to base model | 0.001–0.1 |
| `temperature` | Higher = more diverse group, noisier rewards | 0.7–1.0 |
| `learning_rate` | Lower than SFT to avoid reward hacking | 1e-6–5e-5 |

---

## Method 2 — PPO

**Proximal Policy Optimization.** The standard RLHF algorithm. Trains a value head alongside the policy and clips updates to prevent large policy shifts.

Use PPO when you need:
- Explicit control over the value function
- Per-token credit assignment
- Fine-grained KL budget management

### Full example

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import get_peft_model, LoraConfig

# 1. Build policy with value head
base_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_ID, torch_dtype=torch.bfloat16)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(base_model, lora_config)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

# 2. PPO config
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    kl_penalty="kl",             # or "abs", "mse", "full"
    init_kl_coef=0.1,
    target_kl=6.0,               # adaptive KL controller target
    cliprange=0.2,
    cliprange_value=0.2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy,
    ref_model=None,              # None = frozen copy of initial policy
    tokenizer=tokenizer,
)

# 3. Training loop
for batch in ppo_trainer.dataloader:
    queries = batch["input_ids"]

    # Generate completions
    response_tensors = ppo_trainer.generate(
        queries,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.9,
    )

    # Decode and score
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    rewards = reward_fn(responses)
    reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

    # PPO step
    stats = ppo_trainer.step(queries, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

policy.save_pretrained("./ppo-lora-adapter")
```

---

## Method 3 — Reward-Weighted SFT

No RL loop. Generate completions offline, score them, then fine-tune using supervised loss weighted by reward. Simpler to debug than GRPO/PPO.

### How it works

```
Dataset prompts ──► generate completions ──► score ──► store (prompt, completion, reward)
                                                              │
                                      SFT loss weighted by normalised reward
```

### Full example

```python
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# 1. Pre-generate scored completions
def build_scored_dataset(prompts, policy_model, reward_fn, n_per_prompt=4):
    records = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(policy_model.device)
        for _ in range(n_per_prompt):
            with torch.no_grad():
                out = policy_model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9)
            completion = tokenizer.decode(out[0], skip_special_tokens=True)
            records.append({"prompt": prompt, "completion": completion})

    completions = [r["completion"] for r in records]
    scores = reward_fn(completions)
    for r, s in zip(records, scores):
        r["reward"] = s
    return records

# 2. Dataset wrapper
class RewardDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["completion"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
            "reward": torch.tensor(r["reward"], dtype=torch.float32),
        }

# 3. Custom loss
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
    per_sample_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    weights = F.softmax(rewards, dim=0) * rewards.size(0)
    return (weights * per_sample_loss).mean()

# 4. Custom trainer
class RewardWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = inputs.pop("reward")
        outputs = model(**inputs)
        loss = reward_weighted_loss(outputs.logits, inputs["labels"], rewards)
        return (loss, outputs) if return_outputs else loss

# 5. Train
args = TrainingArguments(
    output_dir="./rwsft-output",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
)

records = build_scored_dataset(prompts, policy_model, reward_fn)
train_dataset = RewardDataset(records, tokenizer)

trainer = RewardWeightedTrainer(model=policy_model, args=args, train_dataset=train_dataset)
trainer.train()
```

---

## Method 4 — DPO (Preference Pairs)

Use your reward model to label which of two completions is better, then train with Direct Preference Optimization. No generation loop during training — the preference dataset is built offline.

### When to use

- Your reward model is better at **ranking** than scoring (e.g. an LLM judge comparing two responses)
- You want a stable, simple training loop with no RL

### Building the preference dataset

```python
from datasets import Dataset

def build_preference_dataset(prompts, policy_model, reward_fn):
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(policy_model.device)
        with torch.no_grad():
            out_a = policy_model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9)
            out_b = policy_model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9)

        completion_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
        completion_b = tokenizer.decode(out_b[0], skip_special_tokens=True)

        score_a, score_b = reward_fn([completion_a, completion_b])

        rows.append({
            "prompt": prompt,
            "chosen":   completion_a if score_a >= score_b else completion_b,
            "rejected": completion_b if score_a >= score_b else completion_a,
        })
    return Dataset.from_list(rows)
```

### Training with DPO

```python
from trl import DPOTrainer, DPOConfig

pref_dataset = build_preference_dataset(prompts, policy_model, reward_fn)

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,               # KL regularisation; higher = stay closer to reference
    loss_type="sigmoid",    # or "hinge", "ipo"
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=None,         # with LoRA, None uses the frozen base as reference
    args=dpo_config,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

---

## Choosing a Method

```
Does your reward model score a single completion?
├── Yes
│   ├── Want simplest setup?           → Reward-Weighted SFT
│   ├── Want online RL, no value head? → GRPO  ✓ recommended
│   └── Need value head / KL control?  → PPO
└── No — can only compare two completions?
    └── DPO
```

| Method | Online training | RL loop | Value head | Memory overhead | Stability |
|--------|----------------|---------|------------|-----------------|-----------|
| GRPO | Yes | Yes | No | Low | High |
| PPO | Yes | Yes | Yes | Medium | Medium |
| Reward-Weighted SFT | No | No | No | Lowest | Highest |
| DPO | No | No | No | Lowest | Highest |

---

## Hyperparameter Reference

| Hyperparameter | GRPO | PPO | RW-SFT | DPO |
|----------------|------|-----|--------|-----|
| Learning rate | 1e-6 – 5e-5 | 1e-6 – 1e-5 | 1e-5 – 2e-4 | 1e-6 – 5e-5 |
| KL coefficient | 0.001 – 0.1 | 0.01 – 0.2 | — | beta: 0.01 – 0.5 |
| Batch size | 2–8 | 4–32 | 4–16 | 2–8 |
| Generations per prompt | 4–16 | — | 2–8 | 2 |
| LoRA rank | 16–64 | 16–64 | 16–32 | 16–32 |

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Reward increases then collapses | Reward hacking / mode collapse | Increase KL coefficient |
| Loss NaN early in training | LR too high | Reduce LR by 5–10x, add warmup steps |
| Reward stays flat | Reward model too strict / sparse signal | Use a softer reward or increase `num_generations` |
| Generated text becomes repetitive | KL too low, policy drifts far from base | Increase `kl_coef` |
| VRAM OOM during generation | `num_generations` too high | Reduce generations or use QLoRA |
| DPO loss goes negative | `beta` too low | Increase `beta` to 0.3–0.5 |
