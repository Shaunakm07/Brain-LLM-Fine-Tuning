"""
train.py — LoRA fine-tuning with a local LLM judge as the reward signal
========================================================================

WHAT THIS SCRIPT DOES
---------------------
It fine-tunes a small LLM (the "policy model") so that its outputs score
higher according to a second LLM (the "reward model"), which acts as a judge.

You provide:
  1. A set of prompts to train on
  2. A plain-English criteria string, e.g. "responses should be concise and use simple language"

The script then runs a training loop where:
  - The policy model generates several responses to each prompt
  - The reward model scores each response against your criteria
  - Advantage is computed: which completions were above/below average for this prompt
  - The policy model is updated to reinforce above-average completions and
    suppress below-average ones

Over time, the policy model learns to produce outputs that the judge scores highly.

HOW THE OPTIMISATION WORKS (Advantage-Weighted SFT)
----------------------------------------------------
The previous version used raw reward scaling:
    loss = cross_entropy × reward

The problem: if the judge gives all completions a similar score (e.g. all 0.7),
the gradient is the same for every completion and the model doesn't learn which
responses were better or worse than the others.

The fix is advantage normalisation, borrowed from GRPO/PPO:

    advantage = (reward - mean(rewards)) / (std(rewards) + ε)
    loss      = cross_entropy × advantage

This centres the signal around zero for each prompt:
  - Above-average completions → positive advantage → loss is positive → model
    is reinforced to produce this completion more often
  - Below-average completions → negative advantage → loss is negative → when
    minimised, this INCREASES cross-entropy for that completion, suppressing it

This works even when the judge gives coarse scores (e.g. only 0.3 / 0.5 / 0.7),
because what matters is the *relative* difference within the group, not the
absolute values.

WHY A SEPARATE, LARGER REWARD MODEL?
-------------------------------------
The previous version used the same 0.5B model for both policy and judge.
A 0.5B model is too small to reliably follow complex judging instructions —
it tends to output similar scores regardless of the criteria.

We now use:
  - Policy model  : Qwen2.5-0.5B-Instruct  (small, trained with LoRA)
  - Reward model  : Qwen2.5-1.5B-Instruct  (3x larger, better at following judge instructions)

The reward model is always frozen — it is never updated.

LORA RECAP
----------
LoRA (Low-Rank Adaptation) freezes the base model and injects small trainable
matrices into the attention layers. Instead of updating 500M parameters, you
update ~4M — making training feasible on a laptop CPU.

The key hyperparameter is `r` (rank):
  - r=4–8  : very efficient, good for simple tasks
  - r=16   : default for most tasks
  - r=32+  : more expressive but slower

OUTPUTS
-------
After training, the script saves:
  ./lora-adapter/              — LoRA adapter weights (load on top of base model)
  ./lora-adapter/metrics.json  — raw loss, reward, and advantage values per step
  ./lora-adapter/training_metrics.png — 4-panel plot of training progress

USAGE
-----
  # Quickstart with default demo prompts
  python train.py --criteria "responses should be concise and use simple language"

  # With your own prompts
  python train.py --criteria "formal and professional tone" \
                  --prompts "Explain gravity" "What is DNA?"

  # With prompts loaded from a file (one prompt per line)
  python train.py --criteria "detailed with examples" --prompts_file prompts.txt

  # Full options
  python train.py --criteria "..." --epochs 5 --lr 2e-4 --lora_r 8 --completions 4
"""

import os

# Must be set before importing torch.
# MPS (Apple Silicon GPU) doesn't support every PyTorch op. This flag tells
# PyTorch to silently fall back to CPU for any unsupported op rather than
# crashing. The vast majority of the work still runs on the GPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

# Policy and reward models are now different sizes.
# The larger reward model is a better judge — it follows instructions more
# reliably and produces more varied scores across completions.
POLICY_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Device priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
# MPS uses float32 — float16 triggers a Metal shader compiler bug on this model.
# CUDA uses float16 to halve memory usage with no quality loss.
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float32
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32


# ---------------------------------------------------------------------------
# Reward model — frozen, larger Qwen judge
# ---------------------------------------------------------------------------

def load_reward_model(model_id: str = REWARD_MODEL_ID):
    """
    Load the reward model and freeze all its weights.

    We use a 1.5B model here instead of 0.5B. The larger model is significantly
    better at following the judging prompt and produces scores that actually
    reflect the criteria, giving the policy a meaningful gradient signal.

    The model is frozen (requires_grad=False) — it is never updated.
    """
    print(f"Loading reward model: {model_id} (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print("Reward model ready.")
    return model, tokenizer


def score_completion(
    reward_model,
    reward_tokenizer,
    criteria: str,
    prompt: str,
    completion: str,
) -> float:
    """
    Ask the reward model to score a completion on a scale of 1–10.

    The judge receives the criteria, the original prompt, and the completion.
    It replies with a single integer which we normalise to [0.1, 1.0].

    Greedy decoding (do_sample=False) is used so scoring is deterministic —
    the same completion always gets the same score, making training stable.

    Args:
        reward_model:      Frozen judge LLM
        reward_tokenizer:  Tokenizer for the judge
        criteria:          What to optimise for (plain English)
        prompt:            The original user prompt
        completion:        The policy model's response

    Returns:
        Normalised score as a float in [0.1, 1.0]
    """
    judge_prompt = (
        f"You are an impartial judge evaluating AI responses.\n"
        f"Criteria: {criteria}\n\n"
        f"Prompt: {prompt}\n"
        f"Response: {completion}\n\n"
        f"Score this response from 1 (worst) to 10 (best) based strictly on "
        f"the criteria above. Reply with only a single integer. Score:"
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator. You must reply with only a single integer from 1 to 10."},
        {"role": "user",   "content": judge_prompt},
    ]

    text   = reward_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = reward_tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = reward_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=reward_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw        = reward_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for token in raw.split():
        try:
            score = int("".join(c for c in token if c.isdigit()))
            return max(1, min(10, score)) / 10.0
        except ValueError:
            continue

    print(f"  [Warning] Could not parse score from: '{raw}' — using 0.5")
    return 0.5


# ---------------------------------------------------------------------------
# Policy model — LoRA fine-tuned
# ---------------------------------------------------------------------------

def load_policy_model(model_id: str = POLICY_MODEL_ID, lora_r: int = 8):
    """
    Load the base model and attach LoRA adapters.

    LoRA injects two small matrices (A and B, rank r) into every attention
    projection. Only A and B are trained. The update applied to the weights is:
        ΔW = (alpha / r) × B × A

    Args:
        model_id:  HuggingFace model ID for the policy
        lora_r:    Rank of LoRA matrices. r=8 is a good default for 0.5B on CPU.
    """
    print(f"Loading policy model: {model_id} (LoRA r={lora_r})...")
    tokenizer  = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)
    model.print_trainable_parameters()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Generate a single completion from the policy model.

    Temperature=0.9 introduces enough randomness that different completions
    are generated for the same prompt — this variance is what gives the
    advantage calculation something to work with. If all completions were
    identical, the advantage would be zero and nothing would be learned.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.2,  # higher temperature = more diverse completions across the group
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """Thin wrapper around a list of prompt strings for use with DataLoader."""

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def advantage_weighted_loss(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    advantage: float,
) -> torch.Tensor:
    """
    Compute advantage-weighted cross-entropy loss for a single completion.

    INTUITION
    ---------
    Cross-entropy loss measures how surprised the model is by each token
    in the completion. Minimising it makes the model more likely to produce
    this completion in future.

    We scale by advantage rather than raw reward:

        loss = cross_entropy(completion | prompt) × advantage

    Where advantage = (reward - mean_reward) / std_reward for this prompt's
    group of completions.

    This means:
      - advantage > 0  (above-average completion):
          loss is positive → minimising it reinforces this completion ✓

      - advantage < 0  (below-average completion):
          loss is negative → minimising it (making it more negative) INCREASES
          cross-entropy for this completion, suppressing it ✓

      - advantage ≈ 0  (average completion):
          near-zero gradient → model ignores this completion ✓

    WHY WE INCLUDE THE PROMPT IN THE INPUT
    ---------------------------------------
    The model generated this completion conditioned on the full chat-formatted
    prompt. If we compute loss on the completion tokens alone (no context), the
    gradient teaches the model to reproduce the completion from a blank slate —
    a completely different task to what it was actually doing.

    Fix: tokenize prompt + completion together as a full conversation, then mask
    out the prompt tokens in the labels (set to -100 so they contribute zero
    loss). This way the gradient only flows through the completion tokens, but
    the model has the correct context to condition on.

    Args:
        model:      Policy model in training mode
        tokenizer:  Corresponding tokenizer
        prompt:     The original user prompt (needed for context)
        completion: The generated response string
        advantage:  Normalised advantage scalar (can be negative)

    Returns:
        Scalar loss tensor (may be negative for below-average completions)
    """
    # Build the full conversation: system + user prompt + assistant completion
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Also build just the prompt portion so we know how many tokens to mask
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_enc   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    # Clamp prompt_len to the truncated full_enc length.
    # Without this, if full_text exceeds 512 tokens after truncation, prompt_len
    # could be larger than the tensor — masking all tokens and producing NaN loss.
    prompt_len = min(prompt_enc["input_ids"].shape[-1], full_enc["input_ids"].shape[-1])

    # Labels: -100 for all prompt tokens (ignored in loss), real token ids for completion
    labels = full_enc["input_ids"].clone()
    labels[0, :prompt_len] = -100

    outputs   = model(**full_enc, labels=labels)
    base_loss = outputs.loss  # mean CE over completion tokens only (always positive)

    advantage_t = torch.tensor(advantage, dtype=base_loss.dtype, device=DEVICE)
    return base_loss * advantage_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    prompts: list[str],
    criteria: str,
    epochs: int = 5,
    lr: float = 2e-4,
    lora_r: int = 8,
    completions_per_prompt: int = 4,
    output_dir: str = "./lora-adapter",
):
    """
    Main training loop using advantage-weighted SFT.

    For each prompt, the loop:
      1. Generates `completions_per_prompt` responses from the current policy
      2. Scores all of them with the reward model
      3. Computes per-completion advantage = (reward - mean) / std
      4. Updates the policy: reinforce above-average, suppress below-average

    WHY COLLECT ALL COMPLETIONS BEFORE UPDATING?
    We need the mean and std of rewards across the group to compute advantage.
    If we updated after each completion, we wouldn't have the group statistics
    yet. So we first generate and score all completions, then do all updates.

    Higher `completions_per_prompt` gives better advantage estimates (more data
    points to compute mean/std from), but costs more compute per prompt.
    4 is a good minimum; 8 is better if you have the time.

    Args:
        prompts:               List of training prompts
        criteria:              Plain-English scoring criteria
        epochs:                Passes over the prompt dataset (default raised to 5)
        lr:                    Learning rate for AdamW
        lora_r:                LoRA rank
        completions_per_prompt: Completions per prompt (default raised to 4)
        output_dir:            Where to save outputs
    """
    policy_model, policy_tokenizer = load_policy_model(lora_r=lora_r)
    reward_model, reward_tokenizer = load_reward_model()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=lr,
    )

    dataset    = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    step_losses     = []
    step_rewards    = []
    step_advantages = []
    epoch_abs_losses = []  # avg |loss| per epoch — meaningful signal unlike avg loss
    epoch_rewards   = []

    print(f"\nCriteria  : {criteria}")
    print(f"Device    : {DEVICE}")
    print(f"Policy    : {POLICY_MODEL_ID}")
    print(f"Judge     : {REWARD_MODEL_ID}")
    print(f"Training  : {len(prompts)} prompts × {completions_per_prompt} completions × {epochs} epochs")
    print(f"Total steps: {len(prompts) * completions_per_prompt * epochs}\n")

    global_step = 0

    for epoch in range(epochs):
        policy_model.train()
        epoch_abs_loss_sum = 0.0   # sum of |loss| — always positive, meaningful average
        epoch_reward_sum   = 0.0
        count              = 0

        for batch in dataloader:
            prompt = batch[0]

            # ----------------------------------------------------------------
            # Phase 1: generate all completions and score them
            # We must collect all rewards before computing advantage, because
            # advantage requires the mean and std across the whole group.
            # ----------------------------------------------------------------
            policy_model.eval()
            completions = [
                generate_completion(policy_model, policy_tokenizer, prompt)
                for _ in range(completions_per_prompt)
            ]
            policy_model.train()

            rewards = [
                score_completion(reward_model, reward_tokenizer, criteria, prompt, c)
                for c in completions
            ]

            # Compute advantage: normalise rewards to mean=0, std=1.
            # Use a minimum std of 0.1 to prevent explosion when scores cluster
            # closely — this keeps advantages in a reasonable range without
            # skipping updates entirely.
            # Clamp final advantages to [-2, 2] as an additional safety rail.
            mean_r = sum(rewards) / len(rewards)
            std_r  = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_r  = max(std_r, 0.1)

            advantages = [max(-2.0, min(2.0, (r - mean_r) / std_r)) for r in rewards]

            if all(a == 0.0 for a in advantages):
                print(f"  [Skip] All advantages are zero — rewards: {[round(r,2) for r in rewards]}")
                global_step += len(completions)
                count        += len(completions)
                epoch_reward_sum += sum(rewards)
                for r in rewards:
                    step_rewards.append(r)
                    step_losses.append(0.0)
                    step_advantages.append(0.0)
                continue

            # ----------------------------------------------------------------
            # Phase 2: update the policy using advantage-weighted loss
            # ----------------------------------------------------------------
            for completion, reward, advantage in zip(completions, rewards, advantages):
                optimizer.zero_grad()
                loss = advantage_weighted_loss(policy_model, policy_tokenizer, prompt, completion, advantage)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                step_losses.append(loss_val)
                step_rewards.append(reward)
                step_advantages.append(advantage)
                epoch_abs_loss_sum += abs(loss_val)  # track magnitude, not sign
                epoch_reward_sum   += reward
                count            += 1
                global_step      += 1

                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                    f"Reward: {reward:.2f} | Advantage: {advantage:+.3f} | "
                    f"Loss: {loss_val:.4f} | "
                    f'"{completion[:55].strip()}..."'
                )

            print(f"  Group rewards: {[round(r,2) for r in rewards]} | mean={mean_r:.2f} std={std_r:.2f}\n")

        avg_abs_loss = epoch_abs_loss_sum / max(count, 1)
        avg_reward   = epoch_reward_sum   / max(count, 1)
        epoch_abs_losses.append(avg_abs_loss)
        epoch_rewards.append(avg_reward)
        print(f"--- Epoch {epoch+1} | Avg |loss|: {avg_abs_loss:.4f} | Avg reward: {avg_reward:.2f} ---\n")

    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    policy_tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}/")

    metrics = {
        "step_losses":      step_losses,
        "step_rewards":     step_rewards,
        "step_advantages":  step_advantages,
        "epoch_abs_losses": epoch_abs_losses,
        "epoch_rewards":    epoch_rewards,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    plot_metrics(step_losses, step_rewards, step_advantages, epoch_abs_losses, epoch_rewards, output_dir)
    return policy_model, policy_tokenizer


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics(
    step_losses:      list,
    step_rewards:     list,
    step_advantages:  list,
    epoch_abs_losses: list,
    epoch_rewards:    list,
    output_dir:       str,
):
    """
    Save a 5-panel training summary plot.

    Panels:
      Top-left     : Reward per step + rolling average — primary signal of improvement
      Top-middle   : Advantage per step — should spread above and below zero
      Top-right    : Loss per step — oscillates (negative = suppression steps, normal)
      Bottom-left  : Avg reward per epoch — clearest epoch-level signal
      Bottom-right : Avg |loss| per epoch — shows gradient update magnitude

    WHY AVG |LOSS| INSTEAD OF AVG LOSS
    ------------------------------------
    With advantage-weighted loss, positives and negatives cancel — avg loss always
    trends toward zero regardless of training quality. Avg absolute loss shows how
    strongly the model is being updated each epoch, which is actually informative.

    What to look for:
      - Reward (top-left + bottom-left) trending upward = training is working
      - Advantage (top-middle) spread above and below zero = judge discriminating
      - Avg |loss| (bottom-right) stable and non-zero = gradients are flowing
      - If reward is flat: criteria may be too easy or judge not discriminating
    """
    def rolling_avg(values, window=10):
        """Compute a simple rolling average for smoothing noisy step plots."""
        if len(values) < window:
            return values
        return [
            sum(values[max(0, i - window):i + 1]) / len(values[max(0, i - window):i + 1])
            for i in range(len(values))
        ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    # Reward per step + rolling average
    axes[0, 0].plot(step_rewards, color="darkorange", linewidth=0.8, alpha=0.4, label="Raw")
    axes[0, 0].plot(rolling_avg(step_rewards), color="darkorange", linewidth=2, label="Rolling avg")
    axes[0, 0].set_title("Reward per Step")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward (0–1)")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Advantage per step
    axes[0, 1].plot(step_advantages, color="purple", linewidth=1)
    axes[0, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Advantage per Step")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Advantage")
    axes[0, 1].grid(True, alpha=0.3)

    # Loss per step (oscillates between positive/negative — this is expected)
    axes[0, 2].plot(step_losses, color="steelblue", linewidth=0.8)
    axes[0, 2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 2].set_title("Loss per Step (± = reinforce/suppress)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].grid(True, alpha=0.3)

    # Avg reward per epoch — the key metric
    axes[1, 0].plot(range(1, len(epoch_rewards) + 1), epoch_rewards, marker="o", color="darkorange")
    axes[1, 0].set_title("Avg Reward per Epoch  ← key metric")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Avg Reward (0–1)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Avg |loss| per epoch — shows gradient magnitude, not direction
    axes[1, 1].plot(range(1, len(epoch_abs_losses) + 1), epoch_abs_losses, marker="o", color="steelblue")
    axes[1, 1].set_title("Avg |Loss| per Epoch  (update magnitude)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Avg |Loss|")
    axes[1, 1].grid(True, alpha=0.3)

    # Hide the unused 6th panel
    axes[1, 2].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with a local LLM judge (advantage-weighted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --criteria "concise and uses simple language"
  python train.py --criteria "formal tone" --prompts "Explain AI" "What is DNA?"
  python train.py --criteria "detailed with examples" --prompts_file prompts.txt --epochs 5
        """,
    )
    parser.add_argument(
        "--criteria", type=str, required=True,
        help="Plain-English description of what makes a good response"
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        help="Training prompts passed on the command line"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None,
        help="Path to a .txt file with one prompt per line"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--completions", type=int, default=4,
        help="Completions per prompt per epoch — more = better advantage estimates (default: 4)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora-adapter",
        help="Output directory for adapter, metrics, and plots (default: ./lora-adapter)"
    )
    args = parser.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "Explain what a neural network is.",
            "What is the capital of France?",
            "Write a short poem about the ocean.",
            "How does photosynthesis work?",
            "What are the benefits of exercise?",
            "Describe how the internet works.",
            "What causes the seasons to change?",
            "Explain what DNA is.",
            "How do vaccines work?",
            "What is machine learning?",
        ]

    train(
        prompts=prompts,
        criteria=args.criteria,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        completions_per_prompt=args.completions,
        output_dir=args.output_dir,
    )
