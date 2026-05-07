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
  - The policy model generates a response to each prompt
  - The reward model reads the response and scores it 1–10 against your criteria
  - The policy model is updated to make high-scoring responses more likely

Over time, the policy model learns to produce outputs that the judge scores highly.

HOW THE OPTIMISATION WORKS (Reward-Weighted SFT)
-------------------------------------------------
This uses a method called Reward-Weighted Supervised Fine-Tuning.
It avoids a full RL loop, making it stable and simple to run on CPU.

Standard supervised fine-tuning minimises cross-entropy loss equally across
all training examples. Reward-Weighted SFT scales that loss by the reward:

    loss = cross_entropy(completion) × reward

This means:
  - High-reward completions → high loss contribution → strong gradient update
  - Low-reward completions  → low loss contribution  → weak gradient update

The model is therefore pushed to reproduce high-scoring outputs more
than low-scoring ones, which over time maximises the reward.

WHY TWO SEPARATE MODEL INSTANCES?
----------------------------------
  - Policy model  : has LoRA adapters attached. Only the adapter weights (~1%
                    of total params) are updated during training.
  - Reward model  : loaded separately with all gradients disabled (frozen).
                    It is used purely for scoring — never updated.

Both can be the same underlying model (Qwen2.5-0.5B-Instruct here), but they
must be separate instances in memory so that training the policy does not
accidentally affect the judge.

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
  ./lora-adapter/metrics.json  — raw loss and reward values per step and epoch
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
  python train.py --criteria "..." --epochs 5 --lr 2e-4 --lora_r 8 --completions 2
"""

import os
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

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Use CUDA (NVIDIA GPU) if available, otherwise fall back to CPU.
# On Apple M2 we use CPU because MPS has a compiler bug with this model's
# attention shapes that causes a crash during generation.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# float16 halves memory usage on CUDA with no meaningful quality loss.
# float32 is required on CPU to avoid precision issues.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Reward model — frozen Qwen judge
# ---------------------------------------------------------------------------

def load_reward_model(model_id: str = MODEL_ID):
    """
    Load the reward model and freeze all its weights.

    The reward model is a second instance of Qwen used purely as a judge.
    It is never updated during training — freezing it (requires_grad=False)
    ensures no gradients are computed through it, saving memory and preventing
    accidental updates.
    """
    print("Loading reward model (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE)
    model.eval()

    # Disable gradients for every parameter — this model is read-only
    for p in model.parameters():
        p.requires_grad = False

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

    The judge receives:
      - The criteria (what to optimise for)
      - The original prompt
      - The policy model's completion

    It is instructed to reply with a single integer. We parse that integer
    and normalise it to [0, 1] by dividing by 10.

    If the model outputs something unparseable, we return 0.5 (neutral score)
    rather than crashing — this is a soft fallback.

    Args:
        reward_model:      The frozen judge LLM
        reward_tokenizer:  Tokenizer for the judge
        criteria:          Plain-English description of what makes a good response
        prompt:            The original user prompt the policy responded to
        completion:        The policy model's generated response

    Returns:
        A float in [0.1, 1.0] representing the normalised score
    """
    judge_prompt = (
        f"You are an impartial judge. Score the following response on a scale "
        f"from 1 to 10 based on this criteria: {criteria}\n\n"
        f"Prompt: {prompt}\n"
        f"Response: {completion}\n\n"
        f"Reply with only a single integer between 1 and 10. Score:"
    )
    messages = [
        {"role": "system", "content": "You are a strict but fair evaluator. Reply with only a number."},
        {"role": "user",   "content": judge_prompt},
    ]

    # Format as a chat and tokenise
    text   = reward_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = reward_tokenizer(text, return_tensors="pt").to(DEVICE)

    # Generate only a few tokens — we just need a single digit
    with torch.no_grad():
        outputs = reward_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,          # greedy — we want a deterministic score
            pad_token_id=reward_tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = reward_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse the first integer found in the output and clamp to [1, 10]
    for token in raw.split():
        try:
            score = int("".join(c for c in token if c.isdigit()))
            return max(1, min(10, score)) / 10.0  # normalise to [0.1, 1.0]
        except ValueError:
            continue

    # Fallback: neutral score if parsing fails
    print(f"  [Warning] Could not parse reward from: '{raw}' — using 0.5")
    return 0.5


# ---------------------------------------------------------------------------
# Policy model — LoRA fine-tuned
# ---------------------------------------------------------------------------

def load_policy_model(model_id: str = MODEL_ID, lora_r: int = 8):
    """
    Load the base model and attach LoRA adapters.

    LoRA works by injecting two small matrices (A and B, each of rank r) into
    the query, key, value, and output projection layers of every attention block.
    The original weights stay frozen; only A and B are trained.

    The effective weight update is:  ΔW = (alpha/r) × B × A
    where alpha is a scaling factor (set to 2r here as a rule of thumb).

    Args:
        model_id:  HuggingFace model ID
        lora_r:    LoRA rank. Higher = more expressive but more parameters.
                   r=8 is a good default for a 0.5B model on CPU.

    Returns:
        model     : base model with LoRA adapters attached
        tokenizer : corresponding tokenizer
    """
    print("Loading policy model with LoRA adapters...")
    tokenizer  = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,    # standard scaling: alpha = 2r
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # all attention projections
        lora_dropout=0.05,        # small dropout on adapter layers to prevent overfitting
        bias="none",              # don't train bias terms
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)

    # Print how many parameters are actually being trained
    model.print_trainable_parameters()

    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Generate a single completion from the policy model for a given prompt.

    We use temperature=0.9 to introduce some randomness — this is important
    because if the model always generates the same output, the reward signal
    provides no gradient signal to improve on.

    Generation is done inside torch.no_grad() because we don't need gradients
    here — we compute the loss separately in reward_weighted_loss().

    Args:
        model:          The policy model (with LoRA adapters)
        tokenizer:      Corresponding tokenizer
        prompt:         The user prompt to respond to
        max_new_tokens: Maximum length of the generated response

    Returns:
        The generated response as a plain string (decoded, no special tokens)
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
            do_sample=True,       # sample from the distribution (not greedy)
            temperature=0.9,      # slightly high temperature for diversity
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the input tokens — we only want the newly generated part
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    A minimal PyTorch Dataset that wraps a list of prompt strings.

    PyTorch's DataLoader requires a Dataset object to shuffle and batch data.
    Since our prompts are just strings, this is a thin wrapper that makes
    them compatible with DataLoader.
    """

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def reward_weighted_loss(
    model,
    tokenizer,
    completion: str,
    reward: float,
) -> torch.Tensor:
    """
    Compute reward-weighted cross-entropy loss for a single completion.

    INTUITION
    ---------
    Standard cross-entropy loss asks: "how surprised was the model by each
    token in the completion?" We want to minimise this — i.e. make the model
    more likely to produce this completion next time.

    But we only want to reinforce completions the reward model liked.
    So we scale the loss by the reward:

        loss = cross_entropy(completion) × reward

    Effect:
      - reward = 1.0 → full gradient update, strongly reinforce this completion
      - reward = 0.5 → half-strength update
      - reward = 0.1 → near-zero update, almost ignore this completion

    Over many steps, the model is pushed to reproduce high-reward completions
    and ignores low-reward ones — equivalent to maximising expected reward.

    NOTE: The labels are the same as the input_ids here (standard causal LM
    training). The model predicts each token given the previous ones, and the
    loss measures how well it does that.

    Args:
        model:      The policy model in training mode
        tokenizer:  Corresponding tokenizer
        completion: The generated response string to compute loss over
        reward:     Scalar reward in [0, 1] from the judge

    Returns:
        A scalar tensor representing the weighted loss
    """
    # Tokenise the completion and use it as both input and target (causal LM)
    enc    = tokenizer(completion, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    labels = enc["input_ids"].clone()

    # Forward pass — HuggingFace computes cross-entropy internally when labels are passed
    outputs   = model(**enc, labels=labels)
    base_loss = outputs.loss  # mean cross-entropy over all tokens

    # Scale by reward: high reward → large loss contribution → strong update
    reward_t = torch.tensor(reward, device=DEVICE)
    return base_loss * reward_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    prompts: list[str],
    criteria: str,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 8,
    completions_per_prompt: int = 2,
    output_dir: str = "./lora-adapter",
):
    """
    Main training loop.

    Each step:
      1. Generate a completion from the policy model
      2. Score it with the reward model
      3. Compute reward-weighted loss
      4. Backpropagate and update LoRA adapter weights

    We generate `completions_per_prompt` responses per prompt per epoch,
    which gives the reward model multiple examples to score and gives the
    policy model more gradient signal per epoch.

    Gradient clipping (max norm=1.0) prevents any single step from making
    a very large update that destabilises training.

    Args:
        prompts:               List of training prompts
        criteria:              Plain-English scoring criteria for the judge
        epochs:                Number of full passes over the prompt dataset
        lr:                    Learning rate for AdamW
        lora_r:                LoRA rank (see load_policy_model)
        completions_per_prompt: How many completions to generate per prompt per epoch
        output_dir:            Where to save the adapter, metrics, and plots
    """
    policy_model, policy_tokenizer = load_policy_model(lora_r=lora_r)
    reward_model, reward_tokenizer = load_reward_model()

    # Only pass parameters with requires_grad=True (i.e. the LoRA matrices)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=lr,
    )

    dataset    = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Accumulators for plotting
    step_losses   = []   # loss at every individual step
    step_rewards  = []   # reward at every individual step
    epoch_losses  = []   # average loss per epoch
    epoch_rewards = []   # average reward per epoch

    print(f"\nCriteria : {criteria}")
    print(f"Device   : {DEVICE}")
    print(f"Training : {len(prompts)} prompts × {completions_per_prompt} completions × {epochs} epochs")
    print(f"Total steps: {len(prompts) * completions_per_prompt * epochs}\n")

    global_step = 0

    for epoch in range(epochs):
        policy_model.train()
        epoch_loss_sum   = 0.0
        epoch_reward_sum = 0.0
        count            = 0

        for batch in dataloader:
            prompt = batch[0]

            for _ in range(completions_per_prompt):

                # Step 1: generate a completion from the current policy
                # Switch to eval mode for generation (disables dropout)
                policy_model.eval()
                completion = generate_completion(policy_model, policy_tokenizer, prompt)
                policy_model.train()

                # Step 2: ask the frozen reward model to score the completion
                reward = score_completion(
                    reward_model, reward_tokenizer, criteria, prompt, completion
                )

                # Step 3: compute reward-weighted loss and update adapter weights
                optimizer.zero_grad()
                loss = reward_weighted_loss(policy_model, policy_tokenizer, completion, reward)
                loss.backward()

                # Clip gradients to prevent large destabilising updates
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

                optimizer.step()

                # Record metrics
                loss_val = loss.item()
                step_losses.append(loss_val)
                step_rewards.append(reward)
                epoch_loss_sum   += loss_val
                epoch_reward_sum += reward
                count            += 1
                global_step      += 1

                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                    f"Loss: {loss_val:.4f} | Reward: {reward:.2f} | "
                    f'Completion: "{completion[:60].strip()}..."'
                )

        # Epoch summary
        avg_loss   = epoch_loss_sum   / max(count, 1)
        avg_reward = epoch_reward_sum / max(count, 1)
        epoch_losses.append(avg_loss)
        epoch_rewards.append(avg_reward)
        print(f"\n--- Epoch {epoch+1} | Avg loss: {avg_loss:.4f} | Avg reward: {avg_reward:.2f} ---\n")

    # Save LoRA adapter weights — these are small (a few MB) and load on top
    # of any fresh copy of the base model via PeftModel.from_pretrained()
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    policy_tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to {output_dir}/")

    # Save raw metrics as JSON for later analysis
    metrics = {
        "step_losses":   step_losses,
        "step_rewards":  step_rewards,
        "epoch_losses":  epoch_losses,
        "epoch_rewards": epoch_rewards,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    plot_metrics(step_losses, step_rewards, epoch_losses, epoch_rewards, output_dir)
    return policy_model, policy_tokenizer


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics(
    step_losses:   list,
    step_rewards:  list,
    epoch_losses:  list,
    epoch_rewards: list,
    output_dir:    str,
):
    """
    Save a 4-panel training summary plot.

    Panels:
      Top-left  : Loss per step — should trend downward if training is working
      Top-right : Reward per step — should trend upward as the policy improves
      Bottom-left  : Average loss per epoch — smoothed view of the above
      Bottom-right : Average reward per epoch — smoothed view of the above

    What to look for:
      - Reward trending upward = the policy is learning to satisfy the criteria
      - Loss trending downward = the model is becoming more confident in high-reward outputs
      - If reward is flat or decreasing, try: more prompts, more epochs, or a clearer criteria
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    axes[0, 0].plot(step_losses, color="steelblue", linewidth=1)
    axes[0, 0].set_title("Loss per Step")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(step_rewards, color="darkorange", linewidth=1)
    axes[0, 1].set_title("Reward per Step")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Reward (0–1)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="steelblue")
    axes[1, 0].set_title("Avg Loss per Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Avg Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(range(1, len(epoch_rewards) + 1), epoch_rewards, marker="o", color="darkorange")
    axes[1, 1].set_title("Avg Reward per Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Avg Reward (0–1)")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

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
        description="LoRA fine-tuning with a local LLM judge as the reward signal",
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
        help="Plain-English description of what makes a good response (given to the judge)"
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        help="One or more training prompts passed directly on the command line"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None,
        help="Path to a .txt file with one prompt per line (alternative to --prompts)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of full passes over the prompt dataset (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate for AdamW (default: 2e-4)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank — higher means more trainable params (default: 8)"
    )
    parser.add_argument(
        "--completions", type=int, default=2,
        help="Number of completions to generate per prompt per epoch (default: 2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora-adapter",
        help="Where to save the adapter weights, metrics, and plots (default: ./lora-adapter)"
    )
    args = parser.parse_args()

    # Load prompts from file, CLI args, or fall back to built-in demo prompts
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
