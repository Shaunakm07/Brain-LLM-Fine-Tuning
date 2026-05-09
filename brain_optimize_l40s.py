"""
brain_optimize_l40s.py — L40S-optimised TRIBE brain-region maximiser
=====================================================================

Optimisations over brain_optimize.py for NVIDIA L40S (46 GB VRAM, CC 8.9):

  1. No CPU offload  — Qwen 0.5B (~1 GB) + TRIBE/LLaMA 3B (~8 GB) stay on GPU
                       throughout; eliminates the policy_model.cpu() ↔ GPU
                       round-trips (~1-2 s wasted per step).

  2. Wider LoRA      — r=16 (was 8), alpha=32, all attention + FFN projections
                       (q/k/v/o + gate/up/down); ~5× more trainable parameters
                       in the same <0.5% budget → richer gradient signal.

  3. Larger batch    — n_completions=8 default (was 4); halves the variance of
                       the advantage estimate at the cost of 2× TRIBE calls.

  4. Longer text     — max_new_tokens=120 (was 80); TRIBE's encoder benefits from
                       more audio context → more stable reward signal.

  5. LR schedule     — 5-step linear warmup then CosineAnnealingLR; prevents
                       destructive large updates while the LoRA initialises.

  6. Temperature     — anneals 1.0 → 0.6 linearly over training; early steps
                       explore widely, later steps exploit the discovered mode.

  7. Gradient clip   — 0.5 (was 1.0); tighter control since the wider LoRA
                       (more modules) tends to produce larger raw gradients.

  8. Checkpointing   — saves adapter + optimizer + scheduler + metrics every
                       CKPT_EVERY steps; training resumes from latest checkpoint
                       on restart (pass --resume to enable automatic resumption).

  9. More steps      — n_steps=200 default; run --n_steps 0 for infinite loop
                       (Ctrl-C to stop cleanly and save the final adapter).

 10. TF32 matmuls    — already in the original; kept and applied consistently.

 11. GPU memory log  — prints allocated VRAM after every step so you can
                       watch for growth or OOM risk.

 12. Step timing     — prints elapsed time per step and ETA to completion.

USAGE
-----
  # Fast test (verify loop, no TRIBE)
  python brain_optimize_l40s.py --mock_tribe --n_steps 5

  # Full run (Broca, 200 steps) — submit via SLURM with run_brain_optimize_l40s.sh
  python brain_optimize_l40s.py --region broca

  # Resume from a previous run's latest checkpoint
  python brain_optimize_l40s.py --region broca --resume

  # Run until manually stopped
  python brain_optimize_l40s.py --region broca --n_steps 0

  # List available regions
  python brain_optimize_l40s.py --list_regions
"""

import os
import sys
import json
import time
import signal
import argparse
import tempfile
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Mac / CPU compatibility patch — must run before tribev2 import
# ---------------------------------------------------------------------------
_original_subprocess_run = subprocess.run

def _patched_subprocess_run(args, **kwargs):
    if isinstance(args, (list, tuple)):
        args = list(args)
        cmd_str = " ".join(str(a) for a in args)
        if "whisperx" in cmd_str:
            if len(args) >= 2 and args[0] == "uvx" and args[1] == "whisperx":
                args = ["whisperx"] + args[2:]
            import torch as _torch
            if not _torch.cuda.is_available():
                if "--compute_type" in args:
                    idx = args.index("--compute_type")
                    if idx + 1 < len(args) and args[idx + 1] == "float16":
                        args[idx + 1] = "int8"
                else:
                    args = args + ["--compute_type", "int8"]
    return _original_subprocess_run(args, **kwargs)

subprocess.run = _patched_subprocess_run

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# torchvision compatibility shim — must run before peft
# ---------------------------------------------------------------------------
import types as _types

def _patch_torchvision_if_broken() -> None:
    try:
        from torchvision.transforms import InterpolationMode as _  # noqa: F401
        return
    except (RuntimeError, ImportError):
        pass
    for _k in [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[_k]
    from enum import Enum
    class _InterpolationMode(Enum):
        NEAREST = 0; BILINEAR = 2; BICUBIC = 3; BOX = 4; HAMMING = 5; LANCZOS = 1
    _tv      = _types.ModuleType("torchvision")
    _tv_tfms = _types.ModuleType("torchvision.transforms")
    _tv_tfms.InterpolationMode = _InterpolationMode
    _tv.transforms = _tv_tfms
    sys.modules.update({
        "torchvision":                       _tv,
        "torchvision.transforms":            _tv_tfms,
        "torchvision.transforms.functional": _types.ModuleType("torchvision.transforms.functional"),
        "torchvision._meta_registrations":   _types.ModuleType("torchvision._meta_registrations"),
        "torchvision.datasets":              _types.ModuleType("torchvision.datasets"),
        "torchvision.models":                _types.ModuleType("torchvision.models"),
        "torchvision.ops":                   _types.ModuleType("torchvision.ops"),
        "torchvision.io":                    _types.ModuleType("torchvision.io"),
        "torchvision.utils":                 _types.ModuleType("torchvision.utils"),
    })
    print("[brain_optimize_l40s] torchvision ABI mismatch — using stub.")

_patch_torchvision_if_broken()

import transformers as _transformers
if not hasattr(_transformers, "BloomPreTrainedModel"):
    import torch.nn as _nn
    class _BloomStub(_nn.Module):
        pass
    _transformers.BloomPreTrainedModel = _BloomStub

from peft import get_peft_model, LoraConfig, TaskType, PeftModel

sys.path.insert(0, str(Path(__file__).parent))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
CKPT_EVERY      = 10   # save checkpoint every N steps

# Wider LoRA: r=16, all attention + FFN projections
# More trainable params → richer gradient signal, same sub-0.5% budget
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,           # alpha/r = 2 (standard scaling)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # all attention projections
        "gate_proj", "up_proj", "down_proj",        # FFN (SwiGLU)
    ],
    lora_dropout=0.05,
    bias="none",
)

# ---------------------------------------------------------------------------
# Graceful interrupt handling
# ---------------------------------------------------------------------------
_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    if not _stop_requested:
        print("\n[Ctrl-C] Finishing current step then saving. Press again to force-quit.")
        _stop_requested = True
    else:
        raise KeyboardInterrupt

signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_model(device: str, checkpoint_dir: str | None = None) -> tuple:
    """Load Qwen2.5-0.5B-Instruct + LoRA. Optionally resume from checkpoint_dir."""
    print(f"Loading policy model: {POLICY_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint_dir is not None and Path(checkpoint_dir).exists():
        print(f"  Resuming LoRA from {checkpoint_dir}")
        model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=True)
    else:
        model = get_peft_model(base, LORA_CONFIG)

    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return model, tokenizer


def load_reference_model(device: str) -> "AutoModelForCausalLM":
    """Frozen copy of the base model for KL anchoring."""
    print(f"Loading reference model: {POLICY_MODEL_ID} (frozen)...")
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    ref = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
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
    if cache is not None and completion_text in cache:
        return cache[completion_text], None

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


def mock_tribe_reward(completion_text: str, roi_key: str, maximize: bool = True) -> float:
    words        = completion_text.split()
    length_score = min(len(words) / 50.0, 1.0) * 0.4
    rng          = np.random.RandomState(abs(hash(completion_text[:200])) % (2**31))
    noise        = float(rng.normal(0.0, 0.5))
    reward       = np.clip(length_score + noise, -2.0, 2.0)
    return float(reward) if maximize else -float(reward)

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def temperature_at_step(step: int, n_steps: int, t_start: float = 1.0, t_end: float = 0.6) -> float:
    """Linear temperature annealing from t_start to t_end."""
    if n_steps <= 1:
        return t_start
    frac = min((step - 1) / (n_steps - 1), 1.0)
    return t_start + frac * (t_end - t_start)


def generate_completions(
    model,
    tokenizer,
    user_prompt:    str,
    n_completions:  int   = 8,
    max_new_tokens: int   = 120,
    temperature:    float = 0.9,
    device:         str   = "cpu",
) -> list[tuple[str, torch.Tensor, int]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly articulate assistant. Respond with vivid, "
                "detailed, syntactically rich sentences. Use concrete imagery "
                "and complex but clear language."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    _chat_out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    if hasattr(_chat_out, "input_ids"):
        prompt_ids = _chat_out.input_ids.to(device)
    else:
        prompt_ids = _chat_out.to(device)
    prompt_len = prompt_ids.shape[1]

    model.eval()
    with torch.no_grad():
        batch_prompt   = prompt_ids.expand(n_completions, -1)
        attention_mask = torch.ones_like(batch_prompt)
        output_ids = model.generate(
            batch_prompt,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    eos_id  = tokenizer.eos_token_id
    for i in range(n_completions):
        comp_ids = output_ids[i, prompt_len:]
        eos_pos  = (comp_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            comp_ids = comp_ids[: eos_pos[0].item() + 1]
        completion_text = tokenizer.decode(comp_ids, skip_special_tokens=True).strip()
        full_ids = torch.cat([prompt_ids[0], comp_ids]).unsqueeze(0)
        results.append((completion_text, full_ids, prompt_len))

    model.train()
    return results


def compute_kl_divergence(
    policy_logits: torch.Tensor,
    ref_logits:    torch.Tensor,
) -> torch.Tensor:
    log_p = F.log_softmax(policy_logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(),    dim=-1)
    return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def advantage_weighted_loss(
    policy_model,
    ref_model,
    full_input_ids:       torch.Tensor,
    completion_start_idx: int,
    advantage:            float,
    kl_coef:              float = 0.5,
) -> tuple[torch.Tensor, float]:
    policy_logits = policy_model(full_input_ids).logits[0]
    with torch.no_grad():
        ref_logits = ref_model(full_input_ids).logits[0]

    s = completion_start_idx
    comp_policy_logits = policy_logits[s - 1 : -1]
    comp_ref_logits    = ref_logits[s - 1 : -1]
    comp_labels        = full_input_ids[0, s:]

    if comp_labels.numel() == 0:
        return torch.zeros(1, device=full_input_ids.device, requires_grad=True).squeeze(), 0.0

    ce_loss = F.cross_entropy(comp_policy_logits.float(), comp_labels)
    kl      = compute_kl_divergence(comp_policy_logits, comp_ref_logits)
    loss    = advantage * ce_loss + kl_coef * kl
    return loss, float(kl.item())

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    step:           int,
    policy_model,
    tokenizer,
    optimizer,
    scheduler,
    metrics:        dict,
    output_dir:     str,
) -> str:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    policy_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(),  os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(),  os.path.join(ckpt_dir, "scheduler.pt"))
    state = {
        "step":             step,
        "best_reward":      metrics.get("best_reward",      -float("inf")),
        "best_completion":  metrics.get("best_completion",  ""),
        "step_rewards":     metrics.get("step_rewards",     []),
        "best_rewards":     metrics.get("best_rewards",     []),
        "step_losses":      [float(x) for x in metrics.get("step_losses", [])],
        "step_kls":         [float(x) for x in metrics.get("step_kls",    [])],
    }
    with open(os.path.join(ckpt_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"  [ckpt] Saved checkpoint to {ckpt_dir}")
    return ckpt_dir


def find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.exists():
        return None, 0
    candidates = sorted(ckpt_root.glob("step_*"))
    for ckpt_dir in reversed(candidates):
        state_file = ckpt_dir / "state.json"
        if state_file.exists():
            return str(ckpt_dir), int(ckpt_dir.name.split("_")[1])
    return None, 0

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_results(metrics: dict, output_dir: str, roi_key: str,
                          maximize: bool, mock: bool) -> None:
    step_rewards = metrics["step_rewards"]
    best_rewards = metrics["best_rewards"]
    step_losses  = metrics["step_losses"]
    step_kls     = metrics["step_kls"]
    steps        = list(range(1, len(step_losses) + 1))
    if not steps:
        return

    reward_means = [np.mean(r) for r in step_rewards]
    reward_stds  = [np.std(r)  for r in step_rewards]

    roi_desc    = FRIENDLY_ROIS.get(roi_key, {}).get("description", roi_key)
    direction   = "maximise" if maximize else "minimise"
    mock_notice = "  [MOCK REWARDS]" if mock else ""

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Brain Optimizer (L40S) — {direction.capitalize()} '{roi_desc}'\n"
        f"Policy: {POLICY_MODEL_ID}  LoRA r=16  n_completions=8{mock_notice}",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, reward_means, color="steelblue", linewidth=1.8, marker="o", markersize=3)
    ax1.fill_between(
        steps,
        [m - s for m, s in zip(reward_means, reward_stds)],
        [m + s for m, s in zip(reward_means, reward_stds)],
        alpha=0.25, color="steelblue", label="±1 std",
    )
    ax1.axhline(reward_means[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label=f"Baseline: {reward_means[0]:.3f}")
    ax1.set_title(f"Mean Reward per Step\n({roi_key}, {direction})", fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("TRIBE ROI activation" if not mock else "Mock reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, best_rewards, color="darkorange", linewidth=2.0, marker="s", markersize=3)
    ax2.set_title("Best Reward Found So Far", fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Best TRIBE ROI activation" if not mock else "Best mock reward")
    ax2.grid(True, alpha=0.3)
    if best_rewards:
        ax2.annotate(
            f"Best: {best_rewards[-1]:.4f}",
            xy=(steps[-1], best_rewards[-1]),
            xytext=(-50, 15), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, step_losses, color="tomato", linewidth=1.5, marker="o", markersize=3)
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax3.set_title("Loss per Step\n(advantage × CE + kl_coef × KL)", fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, step_kls, color="mediumpurple", linewidth=1.5, marker="o", markersize=3)
    ax4.axhline(1.0, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="KL=1 warn")
    ax4.axhline(2.0, color="red",    linestyle="--", linewidth=0.8, alpha=0.7, label="KL=2 danger")
    ax4.set_title("KL Divergence\n(policy vs reference)", fontweight="bold")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("KL(policy ∥ reference)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def gpu_mem_str(device: str) -> str:
    if not device.startswith("cuda"):
        return ""
    alloc   = torch.cuda.memory_allocated(0)  / 1e9
    reserved = torch.cuda.memory_reserved(0)   / 1e9
    return f"GPU mem: {alloc:.1f}/{reserved:.1f} GB (alloc/reserved)"

# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimize(
    user_prompt:     str,
    roi_key:         str   = "broca",
    maximize:        bool  = True,
    n_steps:         int   = 200,
    n_completions:   int   = 8,
    max_new_tokens:  int   = 120,
    t_start:         float = 1.0,
    t_end:           float = 0.6,
    kl_coef:         float = 0.5,
    learning_rate:   float = 5e-5,
    warmup_steps:    int   = 5,
    mock_tribe:      bool  = False,
    tribe_cache_dir: str   = "./tribe-cache",
    output_dir:      str   = "./brain-optimize-output",
    resume:          bool  = False,
) -> dict:
    global _stop_requested

    os.makedirs(output_dir, exist_ok=True)
    direction = "maximise" if maximize else "minimise"
    roi_desc  = FRIENDLY_ROIS.get(roi_key, {}).get("description", roi_key)
    infinite  = (n_steps == 0)

    print("=" * 70)
    print("Brain Optimizer (L40S)")
    print(f"  Target region  : {roi_desc}")
    print(f"  Direction      : {direction}")
    print(f"  Steps          : {'∞ (until Ctrl-C)' if infinite else n_steps}  ×  {n_completions} completions")
    print(f"  Max tokens     : {max_new_tokens}")
    print(f"  Temperature    : {t_start} → {t_end}")
    print(f"  LR             : {learning_rate}  (warmup {warmup_steps} steps → cosine)")
    print(f"  Mock TRIBE     : {mock_tribe}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        print(f"\nDevice: {device}  ({torch.cuda.get_device_name(0)})")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")
    else:
        print(f"\nDevice: {device}\n")

    # --- Resume or fresh start ---
    start_step       = 1
    all_step_rewards: list[list[float]] = []
    best_rewards:     list[float]       = []
    step_losses:      list[float]       = []
    step_kls:         list[float]       = []
    best_completion:  str               = ""
    best_reward:      float             = -float("inf")
    best_preds:       np.ndarray | None = None
    resume_ckpt_dir:  str | None        = None

    if resume:
        resume_ckpt_dir, start_step_found = find_latest_checkpoint(output_dir)
        if resume_ckpt_dir is not None:
            print(f"Resuming from checkpoint: {resume_ckpt_dir}")
            state_file = os.path.join(resume_ckpt_dir, "state.json")
            with open(state_file) as f:
                saved_state = json.load(f)
            start_step       = saved_state["step"] + 1
            best_reward      = saved_state["best_reward"]
            best_completion  = saved_state["best_completion"]
            all_step_rewards = saved_state["step_rewards"]
            best_rewards     = saved_state["best_rewards"]
            step_losses      = saved_state["step_losses"]
            step_kls         = saved_state["step_kls"]
            print(f"  Continuing from step {start_step} (best reward so far: {best_reward:.4f})\n")
        else:
            print("No checkpoint found — starting fresh.\n")

    # --- Load models ---
    policy_model, tokenizer = load_policy_model(device=device, checkpoint_dir=resume_ckpt_dir)
    ref_model               = load_reference_model(device=device)
    atlas                   = load_destrieux_atlas()

    tribe_model       = None
    tribe_preds_cache: dict[str, float] = {}

    if not mock_tribe:
        print("\nLoading TRIBE v2...")
        tribe_model = load_tribe_model(cache_folder=tribe_cache_dir)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Restore optimizer state if resuming
    if resume and resume_ckpt_dir is not None:
        opt_path = os.path.join(resume_ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    # Linear warmup → cosine annealing
    total_steps = n_steps if not infinite else 10_000   # scheduler needs a finite T_max
    def warmup_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return (step_idx + 1) / max(warmup_steps, 1)
        return 1.0
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=learning_rate * 0.05,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    if resume and resume_ckpt_dir is not None:
        sched_path = os.path.join(resume_ckpt_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
        # Fast-forward scheduler to match start_step
        for _ in range(start_step - 1):
            scheduler.step()

    # --- Training loop ---
    print(f"\nPrompt: {user_prompt!r}\n")
    print("-" * 70)

    step = start_step
    while True:
        if not infinite and step > n_steps:
            break
        if _stop_requested:
            print("\n[Stopping cleanly after Ctrl-C]")
            break

        t_step_start = time.time()
        temp = temperature_at_step(step, n_steps if not infinite else step + 99,
                                   t_start=t_start, t_end=t_end)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n[Step {step}{'/' + str(n_steps) if not infinite else ''}]"
              f"  temp={temp:.3f}  lr={current_lr:.2e}  {gpu_mem_str(device)}")

        # 1. Sample completions (both Qwen models stay on GPU throughout)
        completions = generate_completions(
            policy_model, tokenizer, user_prompt,
            n_completions=n_completions,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            device=device,
        )

        # 2. Score each completion with TRIBE (no CPU offload — both fit on L40S)
        rewards:     list[float]            = []
        preds_batch: list[np.ndarray | None] = []

        for i, (text, _, _) in enumerate(completions):
            t_tribe = time.time()
            print(f"  [{i+1}/{n_completions}] {text[:90]}{'…' if len(text) > 90 else ''}")

            if mock_tribe:
                r = mock_tribe_reward(text, roi_key, maximize)
                p = None
            else:
                r, p = compute_tribe_reward(
                    text, tribe_model, atlas, roi_key,
                    maximize=maximize, cache=tribe_preds_cache,
                )

            rewards.append(r)
            preds_batch.append(p)
            elapsed_tribe = time.time() - t_tribe
            print(f"    reward={r:.4f}  ({elapsed_tribe:.1f}s)")

        # Track best completion
        for (text, _, _), r, p in zip(completions, rewards, preds_batch):
            if r > best_reward:
                best_reward     = r
                best_completion = text
                best_preds      = p
                print(f"  *** New best reward: {best_reward:.4f} ***")
                print(f"  *** Text: {text[:120]}{'…' if len(text) > 120 else ''}")

        all_step_rewards.append(rewards)
        best_rewards.append(best_reward)

        print(f"  Rewards: mean={np.mean(rewards):.4f}  "
              f"std={np.std(rewards):.4f}  "
              f"min={np.min(rewards):.4f}  max={np.max(rewards):.4f}")

        # 3. Advantages
        mean_reward = np.mean(rewards)
        advantages  = [r - mean_reward for r in rewards]

        if np.std(rewards) < 1e-8:
            print("  [Warning] All rewards equal — skipping gradient update.")
            step_losses.append(0.0)
            step_kls.append(0.0)
            scheduler.step()
            step += 1
            continue

        # 4. Advantage-weighted loss + gradient step
        optimizer.zero_grad()
        total_loss = torch.zeros(1, device=device).squeeze()
        total_kl   = 0.0

        for (_, full_ids, comp_start), adv in zip(completions, advantages):
            full_ids = full_ids.to(device)
            loss, kl_val = advantage_weighted_loss(
                policy_model, ref_model, full_ids, comp_start,
                advantage=adv, kl_coef=kl_coef,
            )
            total_loss = total_loss + loss / n_completions
            total_kl  += kl_val / n_completions

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_norm=0.5,   # tighter than original (0.5 vs 1.0) for wider LoRA
        )
        optimizer.step()
        scheduler.step()

        step_losses.append(float(total_loss.item()))
        step_kls.append(total_kl)

        step_time = time.time() - t_step_start
        remaining = (n_steps - step) if not infinite else -1
        eta_str   = (f"  ETA: ~{remaining * step_time / 60:.0f} min"
                     if remaining > 0 else "")
        print(f"  loss={total_loss.item():.4f}  KL={total_kl:.4f}"
              f"  step_time={step_time:.0f}s{eta_str}")

        if total_kl > 2.0:
            print(f"  [Warning] KL={total_kl:.2f} > 2.0 — "
                  "policy drifting. Consider increasing --kl_coef.")

        # 5. Checkpoint
        if step % CKPT_EVERY == 0:
            metrics_so_far = {
                "best_reward":     best_reward,
                "best_completion": best_completion,
                "step_rewards":    all_step_rewards,
                "best_rewards":    best_rewards,
                "step_losses":     step_losses,
                "step_kls":        step_kls,
            }
            save_checkpoint(step, policy_model, tokenizer, optimizer, scheduler,
                            metrics_so_far, output_dir)
            # Also update training curves incrementally
            plot_training_results(metrics_so_far, output_dir, roi_key, maximize, mock_tribe)

        step += 1

    # --- Final outputs ---
    print("\n" + "=" * 70)
    print("Optimization complete.")
    print(f"  Steps completed : {step - start_step}")
    print(f"  Best reward     : {best_reward:.4f}")
    print(f"  Best completion :\n\n    {best_completion}\n")

    best_text_path = os.path.join(output_dir, "best_completion.txt")
    with open(best_text_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt:  {user_prompt}\n")
        f.write(f"Region:  {roi_desc}\n")
        f.write(f"Goal:    {direction}\n")
        f.write(f"Reward:  {best_reward:.6f}\n")
        f.write(f"Mock:    {mock_tribe}\n\n")
        f.write(f"Best completion:\n{best_completion}\n")
    print(f"Best completion saved to {best_text_path}")

    adapter_dir = os.path.join(output_dir, "lora-brain-adapter")
    policy_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    metrics = {
        "roi_key":          roi_key,
        "roi_description":  roi_desc,
        "maximize":         maximize,
        "mock_tribe":       mock_tribe,
        "n_steps":          step - start_step,
        "n_completions":    n_completions,
        "best_reward":      best_reward,
        "best_completion":  best_completion,
        "step_rewards":     all_step_rewards,
        "best_rewards":     best_rewards,
        "step_losses":      [float(x) for x in step_losses],
        "step_kls":         [float(x) for x in step_kls],
    }
    metrics_path = os.path.join(output_dir, "optimizer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if step_losses:
        plot_training_results(metrics, output_dir, roi_key, maximize, mock_tribe)

    if best_preds is not None and not mock_tribe:
        print("\nRendering brain surface plot for best completion...")
        plot_brain_surface(
            best_preds, output_dir=output_dir,
            title_prefix=f"Best completion ({roi_key}, {direction})\n",
        )
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
        description="L40S-optimised brain-region text maximiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mock test (verify loop)
  python brain_optimize_l40s.py --mock_tribe --n_steps 5

  # Full run for Broca's area (submit via SLURM)
  python brain_optimize_l40s.py --region broca

  # Resume from latest checkpoint
  python brain_optimize_l40s.py --region broca --resume

  # Run indefinitely (Ctrl-C saves cleanly)
  python brain_optimize_l40s.py --region broca --n_steps 0

  # Minimise V1 activation
  python brain_optimize_l40s.py --region v1 --minimize

  # List available regions
  python brain_optimize_l40s.py --list_regions
        """,
    )
    parser.add_argument("--prompt", type=str,
                        default=(
                            "Tell a rich, detailed story about a moment when language "
                            "revealed something surprising about the nature of the mind."
                        ),
                        help="User prompt for the policy model")
    parser.add_argument("--region",       type=str,  default="broca")
    parser.add_argument("--minimize",     action="store_true")
    parser.add_argument("--n_steps",      type=int,  default=200,
                        help="Training steps (0 = run until Ctrl-C)")
    parser.add_argument("--n_completions",type=int,  default=8)
    parser.add_argument("--max_new_tokens",type=int, default=120)
    parser.add_argument("--t_start",      type=float, default=1.0,
                        help="Initial sampling temperature")
    parser.add_argument("--t_end",        type=float, default=0.6,
                        help="Final sampling temperature (after annealing)")
    parser.add_argument("--kl_coef",      type=float, default=0.5)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int,   default=5)
    parser.add_argument("--mock_tribe",   action="store_true")
    parser.add_argument("--output_dir",   type=str,  default="./brain-optimize-output-l40s")
    parser.add_argument("--cache",        type=str,  default="./tribe-cache")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--list_regions", action="store_true")
    args = parser.parse_args()

    if args.list_regions:
        print("Loading Destrieux atlas...")
        list_available_regions(load_destrieux_atlas())
        raise SystemExit(0)

    if args.region not in FRIENDLY_ROIS:
        available = ", ".join(sorted(FRIENDLY_ROIS))
        print(f"ERROR: Unknown region '{args.region}'.\nAvailable: {available}")
        raise SystemExit(1)

    optimize(
        user_prompt     = args.prompt,
        roi_key         = args.region,
        maximize        = not args.minimize,
        n_steps         = args.n_steps,
        n_completions   = args.n_completions,
        max_new_tokens  = args.max_new_tokens,
        t_start         = args.t_start,
        t_end           = args.t_end,
        kl_coef         = args.kl_coef,
        learning_rate   = args.lr,
        warmup_steps    = args.warmup_steps,
        mock_tribe      = args.mock_tribe,
        tribe_cache_dir = args.cache,
        output_dir      = args.output_dir,
        resume          = args.resume,
    )
