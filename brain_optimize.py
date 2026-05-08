"""
brain_optimize.py — Optimize Qwen via TRIBE v2 brain region reward
===================================================================

Fine-tunes Qwen2.5-0.5B-Instruct with LoRA to generate text that maximally
(or minimally) activates a specific cortical region as predicted by TRIBE v2.

HOW IT WORKS
------------
This script combines two systems:

  1. TRIBE v2  — the reward model
     Takes a text completion, predicts what fMRI BOLD signal a human brain
     would produce while reading/hearing it, and returns mean activation in
     a target cortical region (e.g., Broca's area).

  2. Qwen2.5-0.5B-Instruct + LoRA — the policy model
     Generates text completions in response to a user prompt.  The LoRA
     adapter is updated to maximise (or minimise) the TRIBE reward signal.

Training algorithm (advantage-weighted SFT + KL penalty):
  For each step:
    1. Sample N completions from the policy (temperature > 0 for diversity)
    2. Score each completion with TRIBE → mean ROI activation = reward
    3. Compute per-completion advantage  = reward − mean(rewards)
    4. Compute loss = (advantage × CE_loss) + kl_coef × KL(policy ∥ reference)
    5. Gradient step on the LoRA parameters only

  - Positive advantage  → completion activates the region more than average
                         → CE loss is reinforced (policy learns to produce it)
  - Negative advantage  → completion activates the region less than average
                         → CE loss is suppressed (policy learns to avoid it)
  - KL penalty          → prevents policy from drifting too far from the
                         base model, keeping outputs fluent and coherent

MOCK MODE (for testing)
-----------------------
TRIBE inference on CPU takes ~1–3 minutes per completion (TTS + WhisperX +
LLaMA 3.2-3B + Wav2Vec-BERT).  To verify the training loop works without
waiting, use --mock_tribe.  Mock mode uses a fast heuristic reward instead
of real brain predictions — output is NOT neurologically meaningful, but the
optimizer logic is identical.

OUTPUTS
-------
  brain-optimize-output/
    optimizer_metrics.json        — per-step rewards, loss, KL divergence
    best_completion.txt           — text of the highest-reward completion found
    training_curves.png           — 4-panel training plot
    brain_surface_best.png        — cortical surface map of best completion
                                    (only produced when not using --mock_tribe)
    lora-brain-adapter/           — saved LoRA adapter weights

USAGE
-----
  # Test the optimizer with mock rewards (fast, ~1 min for 10 steps)
  python brain_optimize.py --mock_tribe

  # Optimise for Broca's area with real TRIBE rewards
  python brain_optimize.py --region broca --n_steps 10 --n_completions 4

  # Minimise visual cortex activation
  python brain_optimize.py --region v1 --minimize --n_steps 5

  # Custom prompt and region
  python brain_optimize.py \\
      --prompt "Describe what you see around you right now." \\
      --region auditory --n_steps 20 --n_completions 4

  # List all available brain regions
  python brain_optimize.py --list_regions

REQUIREMENTS
------------
  pip install torch transformers peft nilearn matplotlib scipy
  pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
  brew install ffmpeg
  huggingface-cli login   # LLaMA 3.2-3B access required by TRIBE

PERFORMANCE NOTES
-----------------
  On a MacBook (CPU only):
    - Each TRIBE call:    ~1–3 minutes  (dominated by LLaMA 3.2-3B feature extraction)
    - Each training step: ~4–12 minutes (4 completions × TRIBE call time)
    - 10 steps total:     ~40–120 minutes
  On a GPU (A100):
    - Each TRIBE call:    ~5–15 seconds
    - 10 steps total:     ~5–15 minutes
  RAM requirements:
    - TRIBE (LLaMA 3.2-3B + Wav2Vec + Fusion):  ~8 GB
    - Qwen 0.5B policy + reference:              ~2 GB
    - Total:                                     ~10 GB  (16 GB+ recommended)
"""

import os
import sys
import json
import argparse
import tempfile
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Mac / CPU compatibility patch for WhisperX — must run before tribev2 import
# (identical to tribe_inference.py; copied here so this script is standalone)
# ---------------------------------------------------------------------------
_original_subprocess_run = subprocess.run

def _patched_subprocess_run(args, **kwargs):
    if isinstance(args, (list, tuple)):
        args = list(args)
        cmd_str = " ".join(str(a) for a in args)
        if "whisperx" in cmd_str:
            if "--compute_type" in args:
                idx = args.index("--compute_type")
                if idx + 1 < len(args) and args[idx + 1] == "float16":
                    args[idx + 1] = "int8"
            else:
                args = args + ["--compute_type", "int8"]
    return _original_subprocess_run(args, **kwargs)

subprocess.run = _patched_subprocess_run

# ---------------------------------------------------------------------------
# Standard imports (after patch)
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Import TRIBE utilities from tribe_inference.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # suppress nilearn atlas warnings on import
    from tribe_inference import (
        load_model        as load_tribe_model,
        run_on_text,
        load_destrieux_atlas,
        extract_region_activity,
        plot_brain_surface,
        FRIENDLY_ROIS,
        list_available_regions,
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLICY_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA configuration — small rank keeps training fast and memory-efficient
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # rank: 8 is a good default for instruct models
    lora_alpha=16,             # scaling factor (alpha/r = 2)
    target_modules=["q_proj", "v_proj"],   # query and value projections
    lora_dropout=0.05,
    bias="none",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_model(device: str = "cpu") -> tuple:
    """
    Load Qwen2.5-0.5B-Instruct with a LoRA adapter as the policy model.

    Returns:
        (model, tokenizer)  — model has LoRA trainable parameters only
    """
    print(f"Loading policy model: {POLICY_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model = get_peft_model(base, LORA_CONFIG)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return model, tokenizer


def load_reference_model(device: str = "cpu") -> "AutoModelForCausalLM":
    """
    Load a frozen copy of the base model as the KL reference.

    The reference model is never updated; it anchors the policy to the
    original model's distribution to prevent mode collapse and keep
    outputs fluent.
    """
    print(f"Loading reference model: {POLICY_MODEL_ID} (frozen)...")
    ref = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def compute_tribe_reward(
    completion_text: str,
    tribe_model,
    atlas:    dict,
    roi_key:  str,
    maximize: bool = True,
    cache:    dict | None = None,
) -> tuple[float, np.ndarray | None]:
    """
    Run TRIBE v2 on a text completion and return the mean predicted BOLD
    in the target cortical region.

    TRIBE's full pipeline is:
      text → gTTS speech synthesis → WhisperX word timestamps
           → LLaMA 3.2-3B text features + Wav2Vec-BERT audio features
           → Fusion Transformer → (n_timesteps, 20484) BOLD predictions

    The reward is the time-averaged mean activation across all vertices
    belonging to the requested brain region (from the Destrieux atlas).

    Args:
        completion_text: The text to evaluate (just the generated completion,
                         not the full prompt).
        tribe_model:     Loaded TribeModel instance.
        atlas:           Destrieux atlas dict from load_destrieux_atlas().
        roi_key:         Target region name (e.g. "broca", "auditory").
        maximize:        If True, higher activation → higher reward.
                         If False, lower activation → higher reward.
        cache:           Optional dict for caching results by text.

    Returns:
        (reward, preds)  where preds is the (n_timesteps, 20484) array
        or None if retrieved from cache.
    """
    # Cache lookup
    if cache is not None and completion_text in cache:
        return cache[completion_text], None

    # Write completion to a temp file for TRIBE
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(completion_text)
        tmp_path = f.name

    try:
        preds, _ = run_on_text(tribe_model, tmp_path)
    finally:
        os.unlink(tmp_path)

    ts     = extract_region_activity(preds, roi_key, atlas)
    reward = float(ts.mean())
    if not maximize:
        reward = -reward

    if cache is not None:
        cache[completion_text] = reward

    return reward, preds


def mock_tribe_reward(
    completion_text: str,
    roi_key:  str,
    maximize: bool = True,
) -> float:
    """
    Fast mock reward for testing the optimizer without running TRIBE.

    The mock reward is a deterministic function of the text:
      reward = (normalised word count) × 0.4 + hash-based noise × 0.6

    This creates a consistent but non-trivial reward surface so the
    optimizer has something to learn.  The output is NOT neurologically
    meaningful — it exists only to verify the training loop is correct.

    Returns a float in approximately [-1, 1].
    """
    words = completion_text.split()
    # Prefer completions of 30–80 words (TRIBE works best on natural sentences)
    length_score = min(len(words) / 50.0, 1.0) * 0.4
    # Deterministic pseudo-random component (stable per text)
    rng   = np.random.RandomState(abs(hash(completion_text[:200])) % (2**31))
    noise = float(rng.normal(0.0, 0.5))
    reward = np.clip(length_score + noise, -2.0, 2.0)
    return float(reward) if maximize else -float(reward)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    user_prompt:    str,
    n_completions:  int = 4,
    max_new_tokens: int = 80,
    temperature:    float = 0.85,
    device:         str = "cpu",
) -> list[tuple[str, torch.Tensor, int]]:
    """
    Sample N completions from the policy model.

    Uses the Qwen chat template so the model responds in its trained format.
    Temperature > 0 ensures diverse completions so advantages are non-zero.

    Args:
        model:          Policy model (Qwen + LoRA).
        tokenizer:      Tokenizer for the policy model.
        user_prompt:    The user message to respond to.
        n_completions:  Number of completions to sample.
        max_new_tokens: Maximum tokens in each completion.
        temperature:    Sampling temperature (0 = greedy, 1 = full entropy).
        device:         Device string.

    Returns:
        List of (completion_text, full_input_ids, completion_start_idx) tuples.
          - completion_text:       Decoded text of the generated completion
          - full_input_ids:        (1, seq_len) tensor — prompt + completion tokens
          - completion_start_idx:  Index into full_input_ids where completion begins
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, articulate assistant.  Respond naturally "
                "and descriptively using clear, complete sentences."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # Tokenise prompt only (no generation yet).
    # In newer transformers apply_chat_template may return a BatchEncoding or
    # a plain tensor depending on the version — normalise to a tensor either way.
    _chat_out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(_chat_out, "input_ids"):
        prompt_ids = _chat_out.input_ids.to(device)
    else:
        prompt_ids = _chat_out.to(device)   # already a tensor
    prompt_len = prompt_ids.shape[1]

    results = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_completions):
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids   = output_ids[0, prompt_len:]
            completion_text  = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            full_input_ids   = output_ids[:1]   # keep as (1, seq_len)
            results.append((completion_text, full_input_ids, prompt_len))

    model.train()
    return results


def compute_kl_divergence(
    policy_logits:    torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """
    KL(policy ∥ reference) averaged over all tokens and vocab positions.

    KL(P ∥ Q) = Σ P(x) log(P(x)/Q(x))

    We use the log-space KL formula for numerical stability:
      KL = Σ exp(log_P) × (log_P − log_Q)

    Args:
        policy_logits:    (seq_len, vocab_size) raw logits from policy
        reference_logits: (seq_len, vocab_size) raw logits from reference

    Returns:
        Scalar KL divergence tensor.
    """
    log_p = F.log_softmax(policy_logits,    dim=-1)
    log_q = F.log_softmax(reference_logits, dim=-1)
    return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def advantage_weighted_loss(
    policy_model,
    ref_model,
    full_input_ids:       torch.Tensor,
    completion_start_idx: int,
    advantage:            float,
    kl_coef:              float = 0.5,
) -> tuple[torch.Tensor, float]:
    """
    Compute advantage-weighted cross-entropy loss with KL penalty.

    Loss = advantage × CE(policy, completion_tokens)
           + kl_coef × KL(policy ∥ reference, completion_tokens)

    The loss is computed on completion tokens only — the prompt tokens
    are excluded because we only want to shape what the model generates,
    not how it processes the input.

    When advantage > 0  (completion activated the target region more than
    average): minimising the loss encourages the policy to produce similar
    completions more often.

    When advantage < 0  (below average): minimising the loss (which is
    negative) encourages the policy to move away from this completion.

    The KL penalty prevents the policy from drifting far from the base
    model, maintaining fluency and preventing mode collapse.

    Args:
        policy_model:         Qwen + LoRA policy (trainable).
        ref_model:            Frozen reference model.
        full_input_ids:       (1, seq_len) — tokenised prompt + completion.
        completion_start_idx: Token index where the completion begins.
        advantage:            Scalar advantage value for this completion.
        kl_coef:              Weight for the KL penalty term.

    Returns:
        (loss, kl_value)  — loss is a differentiable tensor; kl_value is float.
    """
    # Forward pass through policy (with gradients)
    policy_logits = policy_model(full_input_ids).logits[0]   # (seq_len, vocab)

    # Forward pass through reference (no gradients)
    with torch.no_grad():
        ref_logits = ref_model(full_input_ids).logits[0]     # (seq_len, vocab)

    # Slice to completion tokens only:
    # To predict token at position t we use the logits at position t-1.
    # Completion tokens start at completion_start_idx, so we use logits
    # from [completion_start_idx-1  to  seq_len-2] to predict
    # labels  from [completion_start_idx  to  seq_len-1].
    s = completion_start_idx
    comp_policy_logits = policy_logits[s - 1 : -1]    # (comp_len, vocab)
    comp_ref_logits    = ref_logits[s - 1 : -1]       # (comp_len, vocab)
    comp_labels        = full_input_ids[0, s:]        # (comp_len,)

    if comp_labels.numel() == 0:
        # Edge case: empty completion — return zero loss
        return torch.tensor(0.0, requires_grad=True), 0.0

    # CE loss on completion tokens
    ce_loss = F.cross_entropy(comp_policy_logits, comp_labels)

    # KL divergence: policy vs reference on completion tokens
    kl = compute_kl_divergence(comp_policy_logits, comp_ref_logits)

    # Combined loss
    loss = advantage * ce_loss + kl_coef * kl
    return loss, float(kl.item())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_results(
    metrics: dict,
    output_dir: str,
    roi_key:    str,
    maximize:   bool,
    mock:       bool,
) -> None:
    """
    Save a 4-panel training plot to output_dir/training_curves.png.

    Panel 1 — Mean reward per step (± std across completions)
        Shows whether the policy is learning to increase/decrease target
        region activation over training.

    Panel 2 — Best reward found so far
        Monotonically non-decreasing (non-increasing) curve showing the
        best completion found at any point.  A flattening curve suggests
        the optimizer has converged.

    Panel 3 — Loss per step
        The advantage-weighted CE + KL loss.  This is expected to fluctuate
        more than supervised learning loss because the reward signal is noisy.

    Panel 4 — KL divergence per step
        Policy drift from the reference (base) model.  If this rises above
        ~2, the kl_coef should be increased to preserve fluency.
        If it stays near 0, kl_coef may be reduced to allow more exploration.

    Args:
        metrics:    Dict with keys: step_rewards (list of lists), best_rewards,
                    step_losses, step_kls.
        output_dir: Directory to save the plot.
        roi_key:    Target ROI name (for title).
        maximize:   True if maximising activation, False if minimising.
        mock:       True if mock_tribe was used (affects subtitle).
    """
    os.makedirs(output_dir, exist_ok=True)

    step_rewards  = metrics["step_rewards"]   # list of reward lists
    best_rewards  = metrics["best_rewards"]   # list of floats
    step_losses   = metrics["step_losses"]    # list of floats
    step_kls      = metrics["step_kls"]       # list of floats
    steps         = list(range(1, len(step_losses) + 1))

    reward_means = [np.mean(r) for r in step_rewards]
    reward_stds  = [np.std(r)  for r in step_rewards]

    roi_desc    = FRIENDLY_ROIS.get(roi_key, {}).get("description", roi_key)
    direction   = "maximise" if maximize else "minimise"
    mock_notice = "  [MOCK REWARDS — not real brain activity]" if mock else ""

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Brain Optimizer — {direction.capitalize()} '{roi_desc}'\n"
        f"Policy: {POLICY_MODEL_ID}{mock_notice}",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel 1: Mean reward per step ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, reward_means, color="steelblue", linewidth=1.8, marker="o", markersize=4)
    ax1.fill_between(
        steps,
        [m - s for m, s in zip(reward_means, reward_stds)],
        [m + s for m, s in zip(reward_means, reward_stds)],
        alpha=0.25, color="steelblue", label="±1 std",
    )
    ax1.axhline(reward_means[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label=f"Baseline: {reward_means[0]:.3f}")
    ax1.set_title(f"Mean Reward per Step\n({roi_key}, {direction})", fontweight="bold")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("TRIBE ROI activation" if not mock else "Mock reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Best reward found so far ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, best_rewards, color="darkorange", linewidth=2.0, marker="s", markersize=4)
    ax2.set_title("Best Reward Found So Far", fontweight="bold")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Best TRIBE ROI activation" if not mock else "Best mock reward")
    ax2.grid(True, alpha=0.3)
    if best_rewards:
        ax2.annotate(
            f"Final best: {best_rewards[-1]:.4f}",
            xy=(steps[-1], best_rewards[-1]),
            xytext=(-40, 15), textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # --- Panel 3: Loss per step ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, step_losses, color="tomato", linewidth=1.5, marker="o", markersize=4)
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax3.set_title("Loss per Step\n(advantage × CE + kl_coef × KL)", fontweight="bold")
    ax3.set_xlabel("Training step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: KL divergence per step ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, step_kls, color="mediumpurple", linewidth=1.5, marker="o", markersize=4)
    ax4.axhline(1.0, color="orange", linestyle="--", linewidth=0.8, alpha=0.7,
                label="Warning threshold (KL=1)")
    ax4.axhline(2.0, color="red",    linestyle="--", linewidth=0.8, alpha=0.7,
                label="Danger threshold (KL=2)")
    ax4.set_title("KL Divergence per Step\n(policy vs reference)", fontweight="bold")
    ax4.set_xlabel("Training step")
    ax4.set_ylabel("KL(policy ∥ reference)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimize(
    user_prompt:    str,
    roi_key:        str  = "broca",
    maximize:       bool = True,
    n_steps:        int  = 10,
    n_completions:  int  = 4,
    max_new_tokens: int  = 80,
    temperature:    float = 0.85,
    kl_coef:        float = 0.5,
    learning_rate:  float = 5e-5,
    mock_tribe:     bool  = False,
    tribe_cache_dir: str  = "./tribe-cache",
    output_dir:     str  = "./brain-optimize-output",
) -> dict:
    """
    Run the brain region optimisation loop.

    Trains a Qwen LoRA adapter so that the model generates text completions
    that maximally (or minimally) activate the target cortical region as
    measured by TRIBE v2.

    Args:
        user_prompt:     The user message that the policy model responds to.
        roi_key:         Target brain region (key in FRIENDLY_ROIS).
        maximize:        True = maximise activation; False = minimise.
        n_steps:         Number of optimisation steps.
        n_completions:   Completions sampled per step (more = better gradient
                         estimate, but slower per step).
        max_new_tokens:  Max completion length in tokens (~4 chars/token).
        temperature:     Sampling temperature.  0 = greedy (no diversity);
                         >1 = high diversity but incoherent text.
        kl_coef:         Weight for the KL penalty.  Increase if the policy
                         starts producing incoherent text (KL > 2).
        learning_rate:   AdamW learning rate for the LoRA parameters.
        mock_tribe:      Use mock rewards instead of TRIBE (for testing).
        tribe_cache_dir: Where to cache TRIBE model weights.
        output_dir:      Directory for saved adapter, metrics, and plots.

    Returns:
        metrics dict with keys: step_rewards, best_rewards, step_losses,
        step_kls, best_completion, best_reward.
    """
    os.makedirs(output_dir, exist_ok=True)
    direction = "maximise" if maximize else "minimise"
    roi_desc  = FRIENDLY_ROIS.get(roi_key, {}).get("description", roi_key)

    print("=" * 70)
    print(f"Brain Optimizer")
    print(f"  Target region  : {roi_desc}")
    print(f"  Direction      : {direction}")
    print(f"  Steps          : {n_steps}  ×  {n_completions} completions")
    print(f"  Max tokens     : {max_new_tokens}")
    print(f"  Mock TRIBE     : {mock_tribe}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    # --- Load models ---
    policy_model, tokenizer = load_policy_model(device=device)
    ref_model               = load_reference_model(device=device)
    atlas                   = load_destrieux_atlas()

    tribe_model  = None
    tribe_preds_cache: dict[str, float] = {}

    if not mock_tribe:
        print("\nLoading TRIBE v2 (this may take a few minutes on first run)...")
        tribe_model = load_tribe_model(cache_folder=tribe_cache_dir)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # --- Metrics tracking ---
    all_step_rewards: list[list[float]] = []
    best_rewards:     list[float]       = []
    step_losses:      list[float]       = []
    step_kls:         list[float]       = []
    best_completion:  str               = ""
    best_reward:      float             = -float("inf")
    best_preds:       np.ndarray | None = None

    print(f"\nPrompt: {user_prompt!r}\n")
    print("-" * 70)

    for step in range(1, n_steps + 1):
        print(f"\n[Step {step}/{n_steps}]")

        # 1. Sample completions from policy
        completions = generate_completions(
            policy_model, tokenizer, user_prompt,
            n_completions=n_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        # 2. Score each completion with TRIBE (or mock)
        rewards     = []
        preds_batch = []

        for i, (text, _, _) in enumerate(completions):
            print(f"  Completion {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")

            if mock_tribe:
                r = mock_tribe_reward(text, roi_key, maximize)
                p = None
            else:
                r, p = compute_tribe_reward(
                    text, tribe_model, atlas, roi_key,
                    maximize=maximize,
                    cache=tribe_preds_cache,
                )

            rewards.append(r)
            preds_batch.append(p)
            print(f"    reward = {r:.4f}")

        # Track best completion overall
        for (text, _, _), r, p in zip(completions, rewards, preds_batch):
            if r > best_reward:
                best_reward     = r
                best_completion = text
                best_preds      = p
                print(f"  *** New best reward: {best_reward:.4f} ***")

        all_step_rewards.append(rewards)
        best_rewards.append(best_reward)

        print(f"  Step rewards: mean={np.mean(rewards):.4f}  "
              f"std={np.std(rewards):.4f}  "
              f"min={np.min(rewards):.4f}  max={np.max(rewards):.4f}")

        # 3. Compute per-completion advantages
        mean_reward = np.mean(rewards)
        advantages  = [r - mean_reward for r in rewards]

        # If all rewards are equal, advantages are all zero — skip gradient step
        if np.std(rewards) < 1e-8:
            print("  [Warning] All completions have equal reward. "
                  "No gradient update this step.")
            step_losses.append(0.0)
            step_kls.append(0.0)
            continue

        # 4. Compute advantage-weighted loss and gradient update
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_kl   = 0.0

        for (_, full_ids, comp_start), adv in zip(completions, advantages):
            full_ids = full_ids.to(device)
            loss, kl_val = advantage_weighted_loss(
                policy_model, ref_model, full_ids, comp_start,
                advantage=adv,
                kl_coef=kl_coef,
            )
            total_loss = total_loss + loss / n_completions   # normalise by batch
            total_kl  += kl_val / n_completions

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        step_losses.append(float(total_loss.item()))
        step_kls.append(total_kl)
        print(f"  loss={total_loss.item():.4f}  KL={total_kl:.4f}")

        # KL warning
        if total_kl > 2.0:
            print(f"  [Warning] KL={total_kl:.2f} > 2.0 — "
                  "policy drifting from base. Consider increasing kl_coef.")

    # --- Save outputs ---
    print("\n" + "=" * 70)
    print("Optimization complete.")
    print(f"  Best reward     : {best_reward:.4f}")
    print(f"  Best completion :\n\n    {best_completion}\n")

    # Save best completion text
    best_text_path = os.path.join(output_dir, "best_completion.txt")
    with open(best_text_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt:  {user_prompt}\n")
        f.write(f"Region:  {roi_desc}\n")
        f.write(f"Goal:    {direction}\n")
        f.write(f"Reward:  {best_reward:.6f}\n")
        f.write(f"Mock:    {mock_tribe}\n\n")
        f.write(f"Best completion:\n{best_completion}\n")
    print(f"Best completion saved to {best_text_path}")

    # Save LoRA adapter
    adapter_dir = os.path.join(output_dir, "lora-brain-adapter")
    policy_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    # Save metrics JSON
    metrics = {
        "roi_key":        roi_key,
        "roi_description": roi_desc,
        "maximize":       maximize,
        "mock_tribe":     mock_tribe,
        "n_steps":        n_steps,
        "n_completions":  n_completions,
        "best_reward":    best_reward,
        "best_completion": best_completion,
        "step_rewards":   all_step_rewards,
        "best_rewards":   best_rewards,
        "step_losses":    [float(x) for x in step_losses],
        "step_kls":       [float(x) for x in step_kls],
    }
    metrics_path = os.path.join(output_dir, "optimizer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Training plot
    if step_losses:
        plot_training_results(metrics, output_dir, roi_key, maximize, mock_tribe)

    # Brain surface plot for best completion (only with real TRIBE predictions)
    if best_preds is not None and not mock_tribe:
        print("\nRendering brain surface plot for best completion...")
        plot_brain_surface(
            best_preds,
            output_dir=output_dir,
            title_prefix=f"Best completion ({roi_key}, {direction})\n",
        )
        # Move the default filename to brain_surface_best.png
        default_path = os.path.join(output_dir, "tribe_brain_surface.png")
        final_path   = os.path.join(output_dir, "brain_surface_best.png")
        if os.path.exists(default_path):
            os.rename(default_path, final_path)
            print(f"Brain surface plot saved to {final_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Qwen to generate text that activates a target brain region",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock rewards (fast, verifies the optimizer logic)
  python brain_optimize.py --mock_tribe

  # Optimize for Broca's area (real TRIBE — slow on CPU, ~1h for 10 steps)
  python brain_optimize.py --region broca --n_steps 10 --n_completions 4

  # Minimize primary visual cortex activation
  python brain_optimize.py --region v1 --minimize

  # Custom prompt
  python brain_optimize.py \\
      --prompt "Describe a vivid scene from nature." \\
      --region auditory --n_steps 5

  # List all available brain regions
  python brain_optimize.py --list_regions
        """,
    )
    parser.add_argument("--prompt", type=str,
                        default="Describe something interesting about language and the brain.",
                        help="User prompt the policy model responds to")
    parser.add_argument("--region", type=str, default="broca",
                        help="Target brain region key (default: broca). "
                             "Run --list_regions to see all options.")
    parser.add_argument("--minimize", action="store_true",
                        help="Minimise activation instead of maximising")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of optimisation steps (default: 10)")
    parser.add_argument("--n_completions", type=int, default=4,
                        help="Completions sampled per step (default: 4)")
    parser.add_argument("--max_new_tokens", type=int, default=80,
                        help="Max completion length in tokens (default: 80)")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="Sampling temperature (default: 0.85)")
    parser.add_argument("--kl_coef", type=float, default=0.5,
                        help="KL penalty coefficient (default: 0.5). "
                             "Increase if policy output becomes incoherent.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="AdamW learning rate (default: 5e-5)")
    parser.add_argument("--mock_tribe", action="store_true",
                        help="Use mock rewards instead of real TRIBE inference "
                             "(fast, for testing optimizer logic only)")
    parser.add_argument("--output_dir", type=str, default="./brain-optimize-output",
                        help="Output directory for adapter, metrics, plots")
    parser.add_argument("--cache", type=str, default="./tribe-cache",
                        help="TRIBE model cache directory")
    parser.add_argument("--list_regions", action="store_true",
                        help="Print all available brain regions and exit")
    args = parser.parse_args()

    if args.list_regions:
        print("Loading Destrieux atlas...")
        list_available_regions(load_destrieux_atlas())
        raise SystemExit(0)

    # Validate region
    if args.region not in FRIENDLY_ROIS:
        available = ", ".join(sorted(FRIENDLY_ROIS))
        print(f"ERROR: Unknown region '{args.region}'.\nAvailable: {available}")
        raise SystemExit(1)

    optimize(
        user_prompt    = args.prompt,
        roi_key        = args.region,
        maximize       = not args.minimize,
        n_steps        = args.n_steps,
        n_completions  = args.n_completions,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        kl_coef        = args.kl_coef,
        learning_rate  = args.lr,
        mock_tribe     = args.mock_tribe,
        tribe_cache_dir = args.cache,
        output_dir     = args.output_dir,
    )
