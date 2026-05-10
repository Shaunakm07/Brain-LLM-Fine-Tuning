"""
compare_checkpoint.py — Compare brain-optimizer checkpoint vs untrained base model
================================================================================

Loads a saved LoRA checkpoint and the untrained base model, then produces a
6-panel comparison figure covering:

  1. Reward trajectory    — training progress from saved metrics
  2. Reward distribution  — spread of TRIBE rewards at step 1 vs latest
  3. Token entropy        — how confident each model is at each position
                            (lower = more focused on specific words)
  4. Probability shifts   — which tokens became more/less likely after training
  5. Base completions     — example text from the untrained model
  6. Checkpoint completions — example text from the fine-tuned model

USAGE
-----
  # Compare latest checkpoint vs base (no TRIBE — fast, ~2 min)
  python compare_checkpoint.py

  # Specify a checkpoint explicitly
  python compare_checkpoint.py --checkpoint ./brain-optimize-output-l40s/checkpoints/step_0030

  # Include fresh TRIBE scoring (slow — ~10 min per model)
  python compare_checkpoint.py --tribe --n_completions 4

  # Mock TRIBE rewards (fast, not neurologically meaningful)
  python compare_checkpoint.py --mock_tribe --n_completions 8
"""

import os
import sys
import json
import argparse
import warnings
import subprocess
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# WhisperX / torchvision patches (needed only if --tribe is used)
# ---------------------------------------------------------------------------
_original_subprocess_run = subprocess.run

def _patched_subprocess_run(args, **kwargs):
    if isinstance(args, (list, tuple)):
        args = list(args)
        cmd_str = " ".join(str(a) for a in args)
        if "whisperx" in cmd_str:
            if len(args) >= 2 and args[0] == "uvx" and args[1] == "whisperx":
                args = ["whisperx"] + args[2:]
            import torch as _t
            if not _t.cuda.is_available():
                if "--compute_type" in args:
                    idx = args.index("--compute_type")
                    if idx + 1 < len(args) and args[idx + 1] == "float16":
                        args[idx + 1] = "int8"
                else:
                    args = args + ["--compute_type", "int8"]
    return _original_subprocess_run(args, **kwargs)

subprocess.run = _patched_subprocess_run

import types as _types

def _patch_torchvision_if_broken():
    try:
        from torchvision.transforms import InterpolationMode as _
        return
    except (RuntimeError, ImportError):
        pass
    for k in [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[k]
    from enum import Enum
    class _IM(Enum):
        NEAREST = 0; BILINEAR = 2; BICUBIC = 3; BOX = 4; HAMMING = 5; LANCZOS = 1
    _tv      = _types.ModuleType("torchvision")
    _tv_tfms = _types.ModuleType("torchvision.transforms")
    _tv_tfms.InterpolationMode = _IM
    _tv.transforms = _tv_tfms
    sys.modules.update({
        "torchvision": _tv, "torchvision.transforms": _tv_tfms,
        "torchvision.transforms.functional": _types.ModuleType("torchvision.transforms.functional"),
        "torchvision._meta_registrations":   _types.ModuleType("torchvision._meta_registrations"),
        "torchvision.datasets":              _types.ModuleType("torchvision.datasets"),
        "torchvision.models":                _types.ModuleType("torchvision.models"),
        "torchvision.ops":                   _types.ModuleType("torchvision.ops"),
        "torchvision.io":                    _types.ModuleType("torchvision.io"),
        "torchvision.utils":                 _types.ModuleType("torchvision.utils"),
    })

_patch_torchvision_if_broken()

import transformers as _transformers
if not hasattr(_transformers, "BloomPreTrainedModel"):
    import torch.nn as _nn
    class _BloomStub(_nn.Module): pass
    _transformers.BloomPreTrainedModel = _BloomStub

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(device: str):
    print(f"Loading base model: {BASE_MODEL_ID}...")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_checkpoint_model(checkpoint_dir: str, device: str):
    print(f"Loading checkpoint: {checkpoint_dir}...")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.exists():
        return None
    for ckpt_dir in sorted(ckpt_root.glob("step_*"), reverse=True):
        if (ckpt_dir / "state.json").exists():
            return str(ckpt_dir)
    return None

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions(
    model, tokenizer, prompt: str,
    n: int = 4, max_new_tokens: int = 120,
    temperature: float = 0.8, device: str = "cpu",
) -> list[str]:
    messages = [
        {"role": "system", "content": "You are a highly articulate assistant. Respond with vivid, detailed, syntactically rich sentences."},
        {"role": "user",   "content": prompt},
    ]
    chat_out  = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt_ids = (chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out).to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids.expand(n, -1),
            attention_mask=torch.ones(n, prompt_len, device=device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    completions = []
    eos_id = tokenizer.eos_token_id
    for i in range(n):
        comp_ids = output_ids[i, prompt_len:]
        eos_pos  = (comp_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos):
            comp_ids = comp_ids[: eos_pos[0].item()]
        completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True).strip())
    return completions

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_token_entropy(
    model, tokenizer, reference_text: str, prompt: str, device: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Per-token entropy of model logits over reference_text.

    H_i = -sum_v p_v * log(p_v)  at each position in the completion.
    Lower entropy = model more confident about which token comes next.
    """
    messages = [
        {"role": "system", "content": "You are a highly articulate assistant. Respond with vivid, detailed, syntactically rich sentences."},
        {"role": "user",   "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_enc    = tokenizer(prompt_text + reference_text, return_tensors="pt").to(device)
    prompt_len  = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(**full_enc).logits[0]   # (seq_len, vocab)

    comp_logits = logits[prompt_len - 1 : -1].float()
    probs       = F.softmax(comp_logits, dim=-1)
    entropy     = -(probs * F.log_softmax(comp_logits, dim=-1)).sum(dim=-1).cpu().numpy()
    token_strs  = [tokenizer.decode([t]) for t in full_enc["input_ids"][0, prompt_len:].tolist()]
    return entropy, token_strs


def compute_probability_shifts(
    base_model, ckpt_model, tokenizer, prompt: str, device: str, top_k: int = 12,
) -> tuple[list[str], np.ndarray]:
    """
    For the first generated token position, find tokens whose probability
    changed most between base and checkpoint models.
    """
    messages = [
        {"role": "system", "content": "You are a highly articulate assistant. Respond with vivid, detailed, syntactically rich sentences."},
        {"role": "user",   "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc         = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        base_probs = F.softmax(base_model(**enc).logits[0, -1].float(), dim=-1).cpu().numpy()
        ckpt_probs = F.softmax(ckpt_model(**enc).logits[0, -1].float(), dim=-1).cpu().numpy()

    delta    = ckpt_probs - base_probs
    top_up   = np.argsort(delta)[-top_k:][::-1]
    top_down = np.argsort(delta)[:top_k]
    indices  = np.concatenate([top_up, top_down])
    tokens   = [tokenizer.decode([i]).strip() or f"<id={i}>" for i in indices]
    return tokens, delta[indices]


def text_stats(completions: list[str]) -> dict:
    word_counts, ttrs, sent_counts, word_lens = [], [], [], []
    for c in completions:
        words = c.split()
        sents = [s for s in c.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        word_counts.append(len(words))
        ttrs.append(len(set(w.lower() for w in words)) / max(len(words), 1))
        sent_counts.append(len(sents))
        word_lens.append(np.mean([len(w) for w in words]) if words else 0)
    return {
        "word_count":       float(np.mean(word_counts)),
        "type_token_ratio": float(np.mean(ttrs)),
        "sentence_count":   float(np.mean(sent_counts)),
        "avg_word_length":  float(np.mean(word_lens)),
    }

# ---------------------------------------------------------------------------
# Optional TRIBE scoring
# ---------------------------------------------------------------------------

def score_with_tribe(completions, tribe_model, atlas, roi_key, mock):
    import tempfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tribe_inference import run_on_text, extract_region_activity

    rewards = []
    for text in completions:
        if mock:
            rng   = np.random.RandomState(abs(hash(text[:200])) % (2**31))
            r     = float(np.clip(min(len(text.split()) / 50.0, 1.0) * 0.4 + rng.normal(0, 0.5), -2, 2))
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(text); tmp = f.name
            try:
                preds, _ = run_on_text(tribe_model, tmp)
            finally:
                os.unlink(tmp)
            r = float(extract_region_activity(preds, roi_key, atlas).mean())
        rewards.append(r)
    return rewards

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 65) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def plot_comparison(
    state, base_completions, ckpt_completions,
    base_entropy, ckpt_entropy, entropy_tokens,
    shift_tokens, shift_deltas,
    base_rewards, ckpt_rewards,
    checkpoint_dir, output_dir, roi_key, use_tribe,
):
    os.makedirs(output_dir, exist_ok=True)
    step = state["step"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Checkpoint vs Base Model  —  Step {step} / 200  —  Region: {roi_key}\n"
        f"Base: {BASE_MODEL_ID}   |   Checkpoint: {Path(checkpoint_dir).name}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.58, wspace=0.35)

    step_rewards = state["step_rewards"]
    best_rewards = state["best_rewards"]
    steps        = list(range(1, len(step_rewards) + 1))
    means        = [np.mean(r) for r in step_rewards]
    stds         = [np.std(r)  for r in step_rewards]

    # --- Panel 1: Reward trajectory ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, means, color="steelblue", linewidth=1.6, marker="o", markersize=3, label="Mean reward/step")
    ax1.fill_between(steps, [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color="steelblue")
    ax1.plot(steps, best_rewards, color="darkorange", linewidth=1.5, linestyle="--", label="Best reward")
    ax1.axhline(means[0], color="gray", linestyle=":", linewidth=0.9, alpha=0.7,
                label=f"Step-1 mean: {means[0]:.3f}")
    ax1.set_title("Reward Trajectory (from training)", fontweight="bold")
    ax1.set_xlabel("Training step"); ax1.set_ylabel("TRIBE ROI activation")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # --- Panel 2: Reward distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    if use_tribe and base_rewards and ckpt_rewards:
        data   = [base_rewards, ckpt_rewards]
        labels = ["Base model", f"Checkpoint\n(step {step})"]
        title  = "TRIBE Reward Distribution\n(fresh inference)"
    else:
        data   = [step_rewards[0], step_rewards[-1]]
        labels = [f"Step 1  (n={len(step_rewards[0])})", f"Step {step}  (n={len(step_rewards[-1])})"]
        title  = "Reward Distribution\n(step 1 vs latest, saved from training)"

    parts = ax2.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("lightcoral"); parts["bodies"][0].set_alpha(0.6)
    parts["bodies"][1].set_facecolor("steelblue");  parts["bodies"][1].set_alpha(0.6)
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(labels)
    for pos, vals in zip([1, 2], data):
        ax2.annotate(f"μ={np.mean(vals):.3f}", xy=(pos, max(vals)),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8, color="dimgray")
    ax2.set_title(title, fontweight="bold"); ax2.set_ylabel("TRIBE ROI activation")
    ax2.grid(True, alpha=0.3, axis="y"); ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # --- Panel 3: Per-token entropy ---
    ax3 = fig.add_subplot(gs[1, 0])
    n_toks = min(len(base_entropy), len(ckpt_entropy), 40)
    x      = np.arange(n_toks)
    ax3.plot(x, base_entropy[:n_toks], color="lightcoral", linewidth=1.5, label="Base model", alpha=0.9)
    ax3.plot(x, ckpt_entropy[:n_toks], color="steelblue",  linewidth=1.5, label=f"Checkpoint (step {step})")
    ax3.fill_between(x, base_entropy[:n_toks], ckpt_entropy[:n_toks], alpha=0.12, color="purple")
    tick_pos    = x[::5]
    tick_labels = [entropy_tokens[i].replace("\n", "↵")[:8] if i < len(entropy_tokens) else "" for i in tick_pos]
    ax3.set_xticks(tick_pos); ax3.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
    ax3.set_title("Per-Token Entropy  (reference = best training completion)\n"
                  "Lower = more confident which token comes next",
                  fontweight="bold")
    ax3.set_xlabel("Token position"); ax3.set_ylabel("Entropy (nats)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
    mb, mc = base_entropy[:n_toks].mean(), ckpt_entropy[:n_toks].mean()
    ax3.text(0.98, 0.97,
             f"Mean entropy\nBase: {mb:.3f}\nCkpt: {mc:.3f}\nΔ: {mc - mb:+.3f}",
             transform=ax3.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # --- Panel 4: Token probability shifts ---
    ax4 = fig.add_subplot(gs[1, 1])
    n_half = len(shift_deltas) // 2
    colors = ["steelblue" if d > 0 else "lightcoral" for d in shift_deltas]
    y_pos  = np.arange(len(shift_tokens))
    ax4.barh(y_pos, shift_deltas * 100, color=colors, edgecolor="white", linewidth=0.4)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([repr(t)[1:-1][:14] for t in shift_tokens], fontsize=8)
    ax4.axvline(0, color="black", linewidth=0.8)
    ax4.axhline(n_half - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.set_title("Token Probability Shifts  (at first generation position)\n"
                  "Blue = more likely after training  |  Red = less likely",
                  fontweight="bold")
    ax4.set_xlabel("Δ probability (%)"); ax4.grid(True, alpha=0.3, axis="x")
    ax4.text(0.02, (n_half / len(shift_tokens)) + 0.03, "↑ more likely",
             transform=ax4.transAxes, fontsize=7.5, color="steelblue")
    ax4.text(0.02, (n_half / len(shift_tokens)) - 0.05, "↓ less likely",
             transform=ax4.transAxes, fontsize=7.5, color="lightcoral")

    # --- Panel 5: Base completions ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    s_base  = text_stats(base_completions)
    ex_text = "\n\n".join(
        f"[{i+1}] {_wrap(c[:280] + ('…' if len(c) > 280 else ''), 64)}"
        for i, c in enumerate(base_completions[:2])
    )
    stats_str = (f"n={len(base_completions)}  words={s_base['word_count']:.0f}  "
                 f"TTR={s_base['type_token_ratio']:.2f}  "
                 f"avg-word-len={s_base['avg_word_length']:.1f}")
    ax5.text(0.01, 0.99, "BASE MODEL (untrained)", transform=ax5.transAxes,
             fontsize=10, fontweight="bold", color="lightcoral", va="top")
    ax5.text(0.01, 0.93, stats_str, transform=ax5.transAxes,
             fontsize=7.5, color="gray", va="top", fontstyle="italic")
    ax5.text(0.01, 0.87, ex_text, transform=ax5.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="#fff5f5", alpha=0.8))

    # --- Panel 6: Checkpoint completions ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    s_ckpt  = text_stats(ckpt_completions)
    ex_text = "\n\n".join(
        f"[{i+1}] {_wrap(c[:280] + ('…' if len(c) > 280 else ''), 64)}"
        for i, c in enumerate(ckpt_completions[:2])
    )
    stats_str = (f"n={len(ckpt_completions)}  words={s_ckpt['word_count']:.0f}  "
                 f"TTR={s_ckpt['type_token_ratio']:.2f}  "
                 f"avg-word-len={s_ckpt['avg_word_length']:.1f}")
    ax6.text(0.01, 0.99, f"CHECKPOINT  (step {step} / 200)", transform=ax6.transAxes,
             fontsize=10, fontweight="bold", color="steelblue", va="top")
    ax6.text(0.01, 0.93, stats_str, transform=ax6.transAxes,
             fontsize=7.5, color="gray", va="top", fontstyle="italic")
    ax6.text(0.01, 0.87, ex_text, transform=ax6.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="#f5f8ff", alpha=0.8))

    plot_path = os.path.join(output_dir, f"comparison_step{step:04d}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {plot_path}")
    return plot_path

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(state, base_stats, ckpt_stats, base_entropy, ckpt_entropy,
                  base_rewards, ckpt_rewards, roi_key):
    step = state["step"]
    step_rewards = state["step_rewards"]
    W = 65
    print("\n" + "=" * W)
    print(f"  COMPARISON SUMMARY  —  Step {step}  —  Region: {roi_key}")
    print("=" * W)
    print(f"  {'Metric':<28} {'Base':>10} {'Checkpoint':>12} {'Δ':>8}")
    print("  " + "-" * (W - 2))

    def row(label, b, c, fmt=".3f"):
        d = c - b
        print(f"  {label:<28} {b:>10{fmt}} {c:>12{fmt}} {'+' if d >= 0 else ''}{d:>7{fmt}}")

    row("Word count (mean)",          base_stats["word_count"],       ckpt_stats["word_count"],       fmt=".1f")
    row("Type-token ratio",           base_stats["type_token_ratio"], ckpt_stats["type_token_ratio"])
    row("Sentence count (mean)",      base_stats["sentence_count"],   ckpt_stats["sentence_count"],   fmt=".1f")
    row("Avg word length",            base_stats["avg_word_length"],  ckpt_stats["avg_word_length"])
    row("Mean token entropy",         base_entropy.mean(),            ckpt_entropy.mean())

    if base_rewards and ckpt_rewards:
        row(f"TRIBE reward ({roi_key})", np.mean(base_rewards), np.mean(ckpt_rewards))
    else:
        r1 = np.mean(step_rewards[0])
        rN = np.mean(step_rewards[-1])
        print(f"\n  Saved TRIBE reward — step 1: {r1:.4f}   step {step}: {rN:.4f}"
              f"   Δ = {'+' if rN > r1 else ''}{rN - r1:.4f}")

    print(f"\n  Best reward found: {state['best_reward']:.4f}")
    print(f"  Best completion:")
    for line in textwrap.wrap(state["best_completion"], width=60):
        print(f"    {line}")
    print("\n" + "=" * W)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LoRA checkpoint vs untrained base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_checkpoint.py
  python compare_checkpoint.py --checkpoint ./brain-optimize-output-l40s/checkpoints/step_0030
  python compare_checkpoint.py --mock_tribe --n_completions 8
  python compare_checkpoint.py --tribe --n_completions 4
        """,
    )
    parser.add_argument("--checkpoint",     type=str,   default=None,
                        help="LoRA checkpoint dir (default: latest in --output_dir)")
    parser.add_argument("--output_dir",     type=str,   default="./brain-optimize-output-l40s",
                        help="Training output dir for finding latest checkpoint")
    parser.add_argument("--save_dir",       type=str,   default="./comparison-plots",
                        help="Where to save the figure")
    parser.add_argument("--prompt",         type=str,
                        default="Tell a rich, detailed story about a moment when language revealed something surprising about the nature of the mind.",
                        help="Prompt used during training")
    parser.add_argument("--region",         type=str,   default="broca")
    parser.add_argument("--n_completions",  type=int,   default=6)
    parser.add_argument("--max_new_tokens", type=int,   default=120)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--tribe",          action="store_true",
                        help="Run fresh TRIBE scoring (slow)")
    parser.add_argument("--mock_tribe",     action="store_true",
                        help="Use mock TRIBE rewards (fast, not neurologically meaningful)")
    parser.add_argument("--cache",          type=str,   default="./tribe-cache")
    args = parser.parse_args()

    # --- Find checkpoint ---
    checkpoint_dir = args.checkpoint or find_latest_checkpoint(args.output_dir)
    if checkpoint_dir is None:
        print(f"ERROR: No checkpoint found in {args.output_dir}/checkpoints/")
        raise SystemExit(1)
    print(f"Checkpoint: {checkpoint_dir}")

    state_path = Path(checkpoint_dir) / "state.json"
    if not state_path.exists():
        print(f"ERROR: state.json not found in {checkpoint_dir}"); raise SystemExit(1)
    with open(state_path) as f:
        state = json.load(f)
    print(f"Step: {state['step']}   Best reward: {state['best_reward']:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Load models ---
    base_model, base_tok = load_base_model(device)
    ckpt_model, ckpt_tok = load_checkpoint_model(checkpoint_dir, device)

    # --- Generate completions ---
    print(f"Generating {args.n_completions} completions from each model...")
    base_completions = generate_completions(base_model, base_tok, args.prompt,
                                            n=args.n_completions, max_new_tokens=args.max_new_tokens,
                                            temperature=args.temperature, device=device)
    ckpt_completions = generate_completions(ckpt_model, ckpt_tok, args.prompt,
                                            n=args.n_completions, max_new_tokens=args.max_new_tokens,
                                            temperature=args.temperature, device=device)
    print("  Done.")

    # --- Entropy analysis on best training completion as reference ---
    reference = state["best_completion"]
    print(f"\nComputing per-token entropy on reference ({len(reference.split())} words)...")
    base_entropy, entropy_tokens = compute_token_entropy(base_model, base_tok, reference, args.prompt, device)
    ckpt_entropy, _              = compute_token_entropy(ckpt_model, ckpt_tok, reference, args.prompt, device)
    print(f"  Base entropy: {base_entropy.mean():.3f}  |  Checkpoint entropy: {ckpt_entropy.mean():.3f}")

    # --- Token probability shifts ---
    print("Computing token probability shifts...")
    shift_tokens, shift_deltas = compute_probability_shifts(
        base_model, ckpt_model, base_tok, args.prompt, device, top_k=12,
    )

    # --- Optional TRIBE scoring ---
    base_rewards = ckpt_rewards = None
    use_tribe = args.tribe or args.mock_tribe
    if use_tribe:
        sys.path.insert(0, str(Path(__file__).parent))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tribe_inference import load_destrieux_atlas
        atlas       = load_destrieux_atlas()
        tribe_model = None
        if args.tribe:
            print("\nLoading TRIBE v2...")
            from tribe_inference import load_model as _ltm
            tribe_model = _ltm(cache_folder=args.cache)
        print(f"\nScoring completions with {'mock' if args.mock_tribe else 'real'} TRIBE...")
        base_rewards = score_with_tribe(base_completions, tribe_model, atlas, args.region, args.mock_tribe)
        ckpt_rewards = score_with_tribe(ckpt_completions, tribe_model, atlas, args.region, args.mock_tribe)
        print(f"  Base mean: {np.mean(base_rewards):.4f}  |  Checkpoint mean: {np.mean(ckpt_rewards):.4f}")

    # --- Summary and plot ---
    print_summary(state,
                  text_stats(base_completions), text_stats(ckpt_completions),
                  base_entropy, ckpt_entropy,
                  base_rewards, ckpt_rewards, args.region)

    print("\nGenerating figure...")
    plot_comparison(
        state=state,
        base_completions=base_completions, ckpt_completions=ckpt_completions,
        base_entropy=base_entropy, ckpt_entropy=ckpt_entropy, entropy_tokens=entropy_tokens,
        shift_tokens=shift_tokens, shift_deltas=shift_deltas,
        base_rewards=base_rewards, ckpt_rewards=ckpt_rewards,
        checkpoint_dir=checkpoint_dir, output_dir=args.save_dir,
        roi_key=args.region, use_tribe=use_tribe,
    )
