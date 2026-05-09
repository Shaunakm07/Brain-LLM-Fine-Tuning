# LoRA Fine-Tuning to Maximise Another Model's Output

This document explains how `train.py` fine-tunes a small LLM using a second, larger LLM as a reward signal — and why the implementation choices were made.

---

## Table of Contents

1. [Concept](#concept)
2. [Why Two Different Model Sizes](#why-two-different-model-sizes)
3. [The Optimisation Method — Advantage-Weighted SFT](#the-optimisation-method--advantage-weighted-sft)
4. [Why Advantage Instead of Raw Reward](#why-advantage-instead-of-raw-reward)
5. [Why Loss Increases Over Time — and the KL Penalty Fix](#why-loss-increases-over-time--and-the-kl-penalty-fix)
6. [Choosing Effective Criteria](#choosing-effective-criteria)
7. [Running train.py](#running-trainpy)
8. [What the Outputs Mean](#what-the-outputs-mean)
9. [Hyperparameter Guide](#hyperparameter-guide)
10. [Common Failure Modes](#common-failure-modes)
11. [Other Methods](#other-methods)

---

## Concept

Standard fine-tuning minimises loss over a fixed dataset. Reward-maximisation fine-tuning uses a second model as a **live scoring function** and trains the policy model to generate outputs that receive higher scores.

```
Prompt
  │
  ▼
Policy Model (Qwen 0.5B + LoRA)  ──generates──►  Completion
                                                       │
                                                       ▼
                                          Reward Model (Qwen 1.5B, frozen)
                                                       │
                                                       ▼
                                                 Score 1–10
                                                       │
                                         Advantage = (score - mean) / std
                                                       │
                                         Gradient update to policy LoRA weights
```

The reward model is always **frozen** — only the LoRA adapters on the policy model are updated.

---

## Why Two Different Model Sizes

The reward model needs to be **larger than the policy model** to act as a reliable judge.

Using the same 0.5B model for both roles failed because:
- A 0.5B model struggles to follow complex judging instructions
- It produces nearly identical scores for all completions (e.g. always 3, 5, or 7 out of 10)
- When all completions get the same score, the gradient signal is near-zero and nothing is learned

| Role | Model | Why |
|------|-------|-----|
| Policy (trained) | `Qwen2.5-0.5B-Instruct` | Small, fast, efficient to fine-tune with LoRA on CPU |
| Judge (frozen) | `Qwen2.5-1.5B-Instruct` | 3× larger, follows judging instructions reliably, produces varied scores |

---

## The Optimisation Method — Advantage-Weighted SFT

`train.py` uses **Advantage-Weighted Supervised Fine-Tuning**. It avoids a full RL loop (no PPO value head, no value function), making it stable and simple to run on CPU. It does include a KL penalty to prevent policy drift — see [Why Loss Increases Over Time](#why-loss-increases-over-time--and-the-kl-penalty-fix).

### Training loop per prompt

```
For each prompt:
  1. Generate N completions from the current policy  (N = --completions, default 4)
  2. Score all N completions with the frozen judge
  3. Compute advantage for each:
       advantage = (reward - mean(rewards)) / (std(rewards) + ε)
  4. For each completion:
       loss = (cross_entropy × advantage) + kl_coef × KL(policy || base)
       backpropagate → update LoRA weights
```

### Why collect all completions before updating?

You need the mean and std of the *whole group* to compute advantage. If you updated after each completion, the group statistics wouldn't exist yet. So all completions for a prompt are generated and scored first, then all updates happen.

---

## Why Advantage Instead of Raw Reward

The previous approach scaled loss directly by the raw reward:

```
loss = cross_entropy × reward
```

**Problem:** if the judge scores all completions similarly (e.g. all 0.7), every gradient update is nearly identical — the model can't tell which completions were relatively better or worse.

The fix is **advantage normalisation**:

```
advantage = (reward - mean_reward) / std_reward
loss      = cross_entropy × advantage
```

Effect on gradients:

| Completion | Reward | Advantage | Effect |
|------------|--------|-----------|--------|
| Best in group | 0.9 | +1.4 | Strongly reinforced — model made more likely to produce this |
| Average | 0.5 | 0.0 | Ignored — no gradient |
| Worst in group | 0.3 | −1.2 | Suppressed — model made less likely to produce this |

This works even when the judge only uses a narrow score range, because what matters is the **relative difference** within the group, not the absolute values.

**Important:** `brain_optimize.py` divides advantages by the reward std at each step (after mean-centering). This decouples the gradient update magnitude from reward variance — a noisy step with large reward spread produces the same-scale gradient as a clean step with tight reward spread. Without this, high-variance steps dominate training and the reward curve shows excessive fluctuation.

---

## Why Loss Increases Over Time — and the KL Penalty Fix

### The problem

After several training epochs you may observe that the loss **rises monotonically** even while the reward is improving. This seems contradictory — surely if the model is generating better responses the loss should fall?

The cause is **suppression accumulation**. Advantage-weighted loss has two kinds of steps:

| Step type | Advantage | Effect on gradient |
|-----------|-----------|-------------------|
| Reinforcement | positive | Push model toward this completion |
| Suppression | negative | Push model *away* from this completion |

Reinforcement steps teach the model what to do. Suppression steps teach it what **not** to do — but in doing so, they also reduce the model's confidence in its own outputs. Over many epochs, suppression gradients accumulate and make the model broadly less certain, which directly increases cross-entropy loss.

A second related problem is **reward hacking**: the policy drifts so far from the base model's distribution that it starts generating outputs optimised for the judge's score but incoherent by any other measure. This is the same failure mode seen in early RLHF work.

### The fix — KL penalty

The solution is to add a second loss term that penalises the policy for **drifting away from the base model**:

```
loss = (cross_entropy × advantage) + kl_coef × KL(policy || base)
```

`KL(policy || base)` is the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) — it measures how different the policy's token distribution has become from the frozen base model.

- When the policy is close to the base → KL ≈ 0 → no extra penalty
- When the policy has drifted → KL rises → the penalty pulls it back

The base model weights are loaded once at the start of training and **never updated**. They act as a fixed anchor.

### Why the same mechanism is used in PPO and RLHF

The KL penalty in `train.py` is a simplified version of the same constraint used in OpenAI's InstructGPT paper and most RLHF pipelines:

```
objective = E[reward] - β × KL(policy || reference)
```

The parameter `β` (here `kl_coef`) controls the trade-off:
- **Too low**: policy drifts → CE rises → potential reward hacking
- **Too high**: policy can't move far enough to improve reward

The default is `kl_coef=0.5`. This was raised from an initial value of 0.1 after observing KL spike to **11+** in early training with the lower value — the policy drifted significantly within the first epoch, which caused reward to improve briefly then collapse. If you see KL staying below 0.5 throughout, you may be able to lower it to 0.3. If KL still spikes above 2–3, raise it to 1.0.

### What "anchor" means in practice

The reference model is a frozen copy of the **base policy** — the same 0.5B Qwen model that LoRA is applied to. It is a completely separate instance in memory, so training the LoRA adapters on the policy does not affect it.

```
Base model weights (frozen, never change)
       │
       ├── Reference model  ← loaded once, used to compute KL
       │
       └── Policy model  ← LoRA adapters on top, these are trained
```

The policy is free to differ from the reference — the KL penalty only makes large drift increasingly costly, not impossible.

---

## Choosing Effective Criteria

The criteria string is the most important hyperparameter. A poor choice causes the majority of training steps to be **skipped** because the judge scores all completions identically.

### Why groups get skipped

If the base model already handles a criteria well (e.g., "be helpful"), the judge scores every completion ~0.8 and the group std is near zero — advantage is undefined, and the step is skipped. No gradient flows.

Observed in practice with `--criteria "use bullet points and keep responses under 3 sentences"`: the 0.5B base model already produces bullet-point responses frequently, so the judge scored most groups identically and over half of all steps were skipped.

### What makes a good criteria

The criteria should describe something the **base model clearly struggles with** — so completions vary widely in quality and the judge can discriminate.

| Criteria type | Skip rate | Why |
|--------------|-----------|-----|
| `"be helpful and concise"` | Very high | Base model already does this |
| `"use bullet points"` | High | Qwen 0.5B naturally uses bullets |
| `"respond only using an analogy to cooking"` | Low | Model rarely does this unprompted |
| `"answer in exactly 3 numbered steps, no more"` | Low | Model frequently violates this |
| `"use no adjectives or adverbs"` | Low | Hard constraint, model often fails it |

### Rule of thumb

Run 1 epoch with a small `--completions 2` and watch the console. If you see `[Skip]` on more than ~30% of prompts, the criteria is too easy — make it more specific or more unusual.

---

## Running train.py

### Install dependencies

```bash
pip install torch transformers peft matplotlib
```

### Basic usage

```bash
# Uses 10 built-in demo prompts
python train.py --criteria "responses should be concise and use simple language"
```

### With your own prompts

```bash
python train.py \
  --criteria "formal and professional tone" \
  --prompts "Explain gravity" "What is DNA?" "How do computers work?"
```

### With prompts from a file

```bash
# prompts.txt — one prompt per line
python train.py --criteria "detailed answers with examples" --prompts_file prompts.txt
```

### All options

| Argument | Default | Description |
|----------|---------|-------------|
| `--criteria` | required | Plain-English description of what makes a good response |
| `--prompts` | — | Prompts passed directly on the command line |
| `--prompts_file` | — | Path to a `.txt` file with one prompt per line |
| `--epochs` | 5 | Passes over the prompt dataset |
| `--lr` | 2e-4 | Learning rate |
| `--lora_r` | 8 | LoRA rank (higher = more expressive, more compute) |
| `--completions` | 4 | Completions per prompt per epoch — more = better advantage estimates |
| `--output_dir` | `./lora-adapter` | Where to save adapter weights, metrics, and plots |

`kl_coef` (default `0.5`) is set directly in `advantage_weighted_loss()` in `train.py`. Increase it if KL spikes above 2–3 or loss rises steadily; decrease it if reward plateaus and KL stays very low.

### After training — load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model      = PeftModel.from_pretrained(base_model, "./lora-adapter")
model.eval()
```

---

## What the Outputs Mean

### Console output

```
Epoch 1/5 | Step 3 | Reward: 0.80 | Advantage: +1.23 | Loss: 0.51 | KL: 0.0032 | "Here is a concise answer..."
Epoch 1/5 | Step 4 | Reward: 0.30 | Advantage: -1.41 | Loss: -0.62 | KL: 0.0041 | "The answer to this question..."
  Group rewards: [0.8, 0.3, 0.6, 0.7] | mean=0.60 std=0.19
```

- **Reward** — raw score from the judge (0.1–1.0)
- **Advantage** — normalised signal; positive = above average, negative = below average
- **Loss** — positive means reinforcing, negative means suppressing; includes the KL penalty component
- **KL** — KL divergence from the base model at this step; should stay small (< 0.1 typically)
- **Group rewards** — all scores for this prompt in this step, plus the mean and std used for advantage

### Plot panels (`training_metrics.png`)

| Panel | What to look for |
|-------|-----------------|
| Reward per step | Should trend upward over training |
| Advantage per step | Should have spread above and below zero — if it's always flat near zero, the judge isn't discriminating |
| Loss per step | Will oscillate between positive and negative — this is normal |
| Avg reward per epoch | Clearest signal — a rising trend means training is working |
| Avg \|loss\| per epoch | Should be stable and non-zero — this shows gradient update magnitude (avg signed loss always cancels to ~0, which is not informative) |
| KL divergence per step | Should stay low and roughly flat. A rising KL means the policy is drifting from the base. If KL grows alongside rising loss, increase `kl_coef` |

---

## Hyperparameter Guide

| Situation | Recommendation |
|-----------|---------------|
| Advantage always near zero | Increase `--completions` to 8+ so the group has more variance |
| Reward not improving after 3+ epochs | Try a more specific `--criteria`, or more diverse prompts |
| Training very slow on CPU | Reduce `--completions` to 2, reduce `--epochs`, shorten `max_new_tokens` in `generate_completion()` |
| Want stronger adaptation | Increase `--lora_r` to 16 or 32 |
| Reward improving then collapsing | Lower `--lr` to 5e-5 |
| Loss rising monotonically | KL penalty is too weak — increase `kl_coef` (default 0.5) to 1.0 in `advantage_weighted_loss()` |
| KL spikes above 2–3 in early training | Same cause — increase `kl_coef`; observed in practice: KL hit 11+ with `kl_coef=0.1` |
| Reward plateaus, KL stays below 0.1 | KL penalty too strong — reduce `kl_coef` to 0.2–0.3 to let the policy move further |

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Advantage always 0.0 / frequent `[Skip]` | All completions scored identically | Criteria too easy for the base model — use a harder, more unusual criteria (see [Choosing Effective Criteria](#choosing-effective-criteria)) |
| Reward flat across all epochs | Too few training steps, or too many skips | More prompts, more completions, harder criteria |
| Loss NaN | LR too high | Reduce `--lr` by 10× |
| Reward improves then crashes | Reward hacking / over-optimisation | Lower `--lr`, reduce epochs, increase `kl_coef` |
| Judge always returns 0.5 | Score parsing failed | Check terminal for `[Warning]` lines — the judge may not be following instructions |
| Loss rising despite reward improving | Suppression drift + no KL anchor | KL penalty is now included by default — if still rising, increase `kl_coef` |
| KL divergence growing over training | Policy drifting too far from base | Increase `kl_coef` in `advantage_weighted_loss()` |

---

## Other Methods

`train.py` implements Advantage-Weighted SFT, which is the most stable option for CPU. Other methods exist for GPU environments:

| Method | When to use | Library |
|--------|-------------|---------|
| **GRPO** | Online RL, no value head needed | `trl.GRPOTrainer` |
| **PPO** | Full RL with explicit value function | `trl.PPOTrainer` |
| **DPO** | Reward model ranks pairs, not scalar scores | `trl.DPOTrainer` |
| **Advantage-Weighted SFT** | CPU, simplest, most stable | `train.py` (this repo) |
