"""
compare_l40s_tribe_full.py — Full TRIBE fMRI comparison: base vs L40S LoRA adapter
====================================================================================

Runs the TRIBE v2 brain encoding model on completions from:
  1. Base model:  Qwen/Qwen2.5-3B-Instruct (untrained)
  2. LoRA model:  latest checkpoint from brain-optimize-output-l40s/

Produces detailed fMRI visualisations across ALL 20 Destrieux atlas regions:

  Figure 1  — All-ROI timeseries comparison (base vs LoRA, all 20 regions)
  Figure 2  — Mean activation bar chart with difference markers
  Figure 3  — Brain surface maps (base, LoRA, difference) on fsaverage5
  Figure 4  — Global BOLD + hemisphere + heatmap comparison
  Figure 5  — Training trajectory (reward over 200 steps)

Outputs saved to:  ./comparison-plots/l40s_full_tribe/
"""

import os
import sys
import json
import argparse
import warnings
import subprocess
import tempfile
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# Subprocess / torchvision patches (must run before tribev2 import)
# ---------------------------------------------------------------------------
_orig_run = subprocess.run

def _patched_run(args, **kwargs):
    if isinstance(args, (list, tuple)):
        args = list(args)
        cmd = " ".join(str(a) for a in args)
        if "whisperx" in cmd:
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
    return _orig_run(args, **kwargs)

subprocess.run = _patched_run

import types as _types

def _patch_torchvision():
    try:
        from torchvision.transforms import InterpolationMode
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

_patch_torchvision()

import transformers as _transformers
if not hasattr(_transformers, "BloomPreTrainedModel"):
    import torch.nn as _nn
    class _BloomStub(_nn.Module): pass
    _transformers.BloomPreTrainedModel = _BloomStub

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR     = "./brain-optimize-output-l40s"
SAVE_DIR       = "./comparison-plots/l40s_full_tribe"
TRIBE_CACHE    = "./tribe-cache"

# The prompt used during training
TRAIN_PROMPT   = (
    "Tell a rich, detailed story about a moment when language revealed "
    "something surprising about the nature of the mind."
)

# Region the model was trained to maximise
TARGET_ROI     = "broca"

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_base(device: str):
    print(f"\nLoading base model:  {BASE_MODEL_ID}  (device={device})")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=dtype, device_map={"": device},
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"  Base model loaded ({sum(p.numel() for p in mdl.parameters())/1e9:.2f}B params)")
    return mdl, tok


def load_lora(checkpoint_dir: str, device: str):
    print(f"\nLoading LoRA checkpoint: {checkpoint_dir}  (device={device})")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=dtype, device_map={"": device},
        trust_remote_code=True,
    )
    mdl = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
    mdl.eval()
    total = sum(p.numel() for p in mdl.parameters())
    lora  = sum(p.numel() for n, p in mdl.named_parameters() if "lora" in n.lower())
    print(f"  LoRA model loaded  ({total/1e9:.2f}B total, {lora/1e6:.1f}M LoRA params)")
    return mdl, tok


def find_latest_checkpoint(output_dir: str) -> str:
    ckpt_root = Path(output_dir) / "checkpoints"
    for d in sorted(ckpt_root.glob("step_*"), reverse=True):
        if (d / "state.json").exists():
            return str(d)
    raise FileNotFoundError(f"No checkpoint with state.json in {ckpt_root}")


def generate_one(model, tokenizer, prompt: str, device: str,
                 max_new_tokens: int = 160, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system",
         "content": "You are a highly articulate assistant. Respond with vivid, "
                    "detailed, syntactically rich sentences."},
        {"role": "user", "content": prompt},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(device)
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones(1, plen, device=device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    comp = out[0, plen:]
    eos  = (comp == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos):
        comp = comp[: eos[0].item()]
    return tokenizer.decode(comp, skip_special_tokens=True).strip()


def generate_n(model, tokenizer, prompt: str, device: str, n: int = 3,
               max_new_tokens: int = 160, temperature: float = 0.7) -> list[str]:
    return [generate_one(model, tokenizer, prompt, device, max_new_tokens, temperature)
            for _ in range(n)]

# ---------------------------------------------------------------------------
# TRIBE inference helpers
# ---------------------------------------------------------------------------

def tribe_on_text(tribe_model, text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        from tribe_inference import run_on_text
        preds, _ = run_on_text(tribe_model, tmp)
    finally:
        os.unlink(tmp)
    return preds   # (n_timesteps, 20484)


def run_tribe_on_completions(tribe_model, completions: list[str],
                              label: str) -> list[np.ndarray]:
    results = []
    for i, text in enumerate(completions):
        print(f"  [{label}] completion {i+1}/{len(completions)} "
              f"({len(text.split())} words)...")
        preds = tribe_on_text(tribe_model, text)
        results.append(preds)
    return results

# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def all_roi_means(preds: np.ndarray, atlas: dict) -> dict[str, float]:
    from tribe_inference import FRIENDLY_ROIS, extract_region_activity
    out = {}
    for key in FRIENDLY_ROIS:
        try:
            ts = extract_region_activity(preds, key, atlas)
            out[key] = float(ts.mean())
        except Exception:
            out[key] = 0.0
    return out


def avg_preds(pred_list: list[np.ndarray]) -> np.ndarray:
    min_t = min(p.shape[0] for p in pred_list)
    stacked = np.stack([p[:min_t] for p in pred_list], axis=0)
    return stacked.mean(axis=0)

# ---------------------------------------------------------------------------
# Figure 1 — All-ROI timeseries comparison
# ---------------------------------------------------------------------------

def fig_all_roi_timeseries(base_preds: np.ndarray, lora_preds: np.ndarray,
                            atlas: dict, save_dir: str, step: int):
    from tribe_inference import FRIENDLY_ROIS, extract_region_activity

    roi_keys = sorted(FRIENDLY_ROIS.keys())
    n_rois   = len(roi_keys)
    ncols    = 4
    nrows    = int(np.ceil(n_rois / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.2))
    fig.suptitle(
        f"All-ROI fMRI Timeseries — Base vs LoRA (Step {step})\n"
        f"Base: {BASE_MODEL_ID}  |  Training region: {TARGET_ROI.upper()}  "
        f"(highlighted with border)",
        fontsize=13, fontweight="bold"
    )

    t_base = np.arange(base_preds.shape[0])
    t_lora = np.arange(lora_preds.shape[0])

    for ax, roi_key in zip(axes.flat, roi_keys):
        try:
            base_ts = extract_region_activity(base_preds, roi_key, atlas)
            lora_ts = extract_region_activity(lora_preds, roi_key, atlas)
        except Exception:
            ax.set_visible(False)
            continue

        ax.plot(t_base, base_ts, color="tomato",    linewidth=1.2, label="Base",  alpha=0.9)
        ax.plot(t_lora, lora_ts, color="steelblue", linewidth=1.2, label="LoRA",  alpha=0.9)
        ax.fill_between(
            np.arange(min(len(base_ts), len(lora_ts))),
            base_ts[:min(len(base_ts), len(lora_ts))],
            lora_ts[:min(len(base_ts), len(lora_ts))],
            alpha=0.12, color="purple",
        )
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.2)

        desc   = FRIENDLY_ROIS[roi_key]["description"]
        short  = desc.split("(")[0].strip()
        hemi   = FRIENDLY_ROIS[roi_key]["hemi"]
        is_target = (roi_key == TARGET_ROI)

        diff_mean = lora_ts.mean() - base_ts.mean()
        sign      = "+" if diff_mean >= 0 else ""
        ax.set_title(f"{short}\n[{hemi}]  Δ={sign}{diff_mean:.3f}",
                     fontsize=8, fontweight="bold" if is_target else "normal")
        ax.set_xlabel("Time (s)", fontsize=6)
        ax.set_ylabel("BOLD (z)", fontsize=6)
        ax.tick_params(labelsize=6)

        if is_target:
            for spine in ax.spines.values():
                spine.set_edgecolor("gold")
                spine.set_linewidth(2.5)
            ax.set_facecolor("#fffdf0")

    for ax in axes.flat[n_rois:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="tomato",    linewidth=2, label="Base (untrained)"),
        plt.Line2D([0], [0], color="steelblue", linewidth=2, label=f"LoRA (step {step})"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9)

    path = os.path.join(save_dir, "fig1_all_roi_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Mean ROI activation bar chart
# ---------------------------------------------------------------------------

def fig_roi_bar_chart(base_preds: np.ndarray, lora_preds: np.ndarray,
                       atlas: dict, save_dir: str, step: int):
    from tribe_inference import FRIENDLY_ROIS

    base_means = all_roi_means(base_preds, atlas)
    lora_means = all_roi_means(lora_preds, atlas)

    roi_keys = sorted(FRIENDLY_ROIS.keys())
    base_vals = np.array([base_means[k] for k in roi_keys])
    lora_vals = np.array([lora_means[k] for k in roi_keys])
    deltas    = lora_vals - base_vals

    short_names = []
    for k in roi_keys:
        desc = FRIENDLY_ROIS[k]["description"]
        short_names.append(desc.split("(")[0].strip()[:22])

    sort_idx = np.argsort(deltas)[::-1]
    roi_keys_s   = [roi_keys[i]   for i in sort_idx]
    short_s      = [short_names[i] for i in sort_idx]
    base_s       = base_vals[sort_idx]
    lora_s       = lora_vals[sort_idx]
    deltas_s     = deltas[sort_idx]

    n   = len(roi_keys_s)
    y   = np.arange(n)
    h   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, n * 0.45)))
    fig.suptitle(
        f"Mean ROI Activation — Base vs LoRA (Step {step})\n"
        f"Sorted by LoRA − Base difference  |  Target region: {TARGET_ROI.upper()} "
        f"(highlighted)",
        fontsize=12, fontweight="bold"
    )

    # Left panel: grouped bars
    ax = axes[0]
    bars_b = ax.barh(y + h/2, base_s, h, color="tomato",    label="Base",         alpha=0.7)
    bars_l = ax.barh(y - h/2, lora_s, h, color="steelblue", label=f"LoRA step {step}", alpha=0.7)

    for i, (bv, lv) in enumerate(zip(base_s, lora_s)):
        ax.text(max(bv, lv) + 0.002, y[i] + h/2, f"{bv:.3f}", va="center", fontsize=6.5,
                color="tomato")
        ax.text(max(bv, lv) + 0.002, y[i] - h/2, f"{lv:.3f}", va="center", fontsize=6.5,
                color="steelblue")

    ax.set_yticks(y)
    ax.set_yticklabels(short_s, fontsize=8)
    ax.set_xlabel("Mean BOLD (z-score)")
    ax.set_title("Absolute Activation by Region", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")
    ax.axvline(0, color="black", linewidth=0.6)

    # Highlight target ROI row
    target_row = next((i for i, k in enumerate(roi_keys_s) if k == TARGET_ROI), None)
    if target_row is not None:
        ax.axhspan(target_row - 0.5, target_row + 0.5, alpha=0.12, color="gold")
        ax.text(ax.get_xlim()[0], target_row, " ★ trained", va="center",
                fontsize=7, color="goldenrod", fontweight="bold")

    # Right panel: delta bar chart
    ax2 = axes[1]
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas_s]
    ax2.barh(y, deltas_s, color=colors, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_s, fontsize=8)
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("ΔBOLD  (LoRA − Base,  z-score)")
    ax2.set_title("Activation Change due to LoRA Training", fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")

    for i, d in enumerate(deltas_s):
        sign = "+" if d >= 0 else ""
        ax2.text(d + (0.001 if d >= 0 else -0.001),
                 y[i],
                 f"{sign}{d:.3f}",
                 va="center",
                 ha="left" if d >= 0 else "right",
                 fontsize=6.5,
                 color="steelblue" if d >= 0 else "tomato")

    if target_row is not None:
        ax2.axhspan(target_row - 0.5, target_row + 0.5, alpha=0.12, color="gold")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig2_roi_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Brain surface maps: base, LoRA, difference
# ---------------------------------------------------------------------------

def _render_surface_panels(mean_act: np.ndarray, vmax: float, cmap: str) -> list[str]:
    from nilearn import plotting, datasets
    import matplotlib.image as mpimg

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    left_data  = mean_act[:10242]
    right_data = mean_act[10242:]

    views = [
        ("left",  "lateral", left_data,  fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("left",  "medial",  left_data,  fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("right", "lateral", right_data, fsaverage["infl_right"], fsaverage["sulc_right"]),
        ("right", "medial",  right_data, fsaverage["infl_right"], fsaverage["sulc_right"]),
    ]

    paths = []
    for hemi, view, data, surf_mesh, bg_map in views:
        surf_fig = plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh, stat_map=data,
            hemi=hemi, view=view, bg_map=bg_map, bg_on_data=True,
            colorbar=False, cmap=cmap, vmax=vmax, engine="matplotlib",
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = tmp.name
        surf_fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(surf_fig)
        paths.append(p)
    return paths


def fig_brain_surface(base_preds: np.ndarray, lora_preds: np.ndarray,
                       save_dir: str, step: int):
    try:
        from nilearn import plotting, datasets
        import matplotlib.image as mpimg
    except ImportError:
        print("[Fig 3] nilearn not installed — skipping surface plot")
        return

    base_mean = base_preds.mean(axis=0)
    lora_mean = lora_preds.mean(axis=0)
    diff_mean = lora_mean - base_mean

    vmax_base = float(np.abs(base_mean).max()) or 1.0
    vmax_lora = float(np.abs(lora_mean).max()) or 1.0
    vmax_both = max(vmax_base, vmax_lora)
    vmax_diff = float(np.abs(diff_mean).max()) or 1.0

    print("  Rendering base surface panels...")
    base_panels = _render_surface_panels(base_mean, vmax_both, "RdBu_r")
    print("  Rendering LoRA surface panels...")
    lora_panels = _render_surface_panels(lora_mean, vmax_both, "RdBu_r")
    print("  Rendering difference surface panels...")
    diff_panels = _render_surface_panels(diff_mean, vmax_diff, "PiYG")

    panel_labels = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    row_labels   = [
        f"Base Model\n({BASE_MODEL_ID})",
        f"LoRA Adapter\n(step {step})",
        f"Difference\n(LoRA − Base)",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        f"Cortical Surface Maps — Base vs LoRA (Step {step})\n"
        f"Time-averaged mean BOLD (z-score) on fsaverage5 inflated surface\n"
        f"Rows: Base | LoRA | Difference (LoRA−Base)",
        fontsize=12, fontweight="bold",
    )

    all_panels = [base_panels, lora_panels, diff_panels]
    for row_i, (row_panels, row_lbl) in enumerate(zip(all_panels, row_labels)):
        for col_i, (img_path, panel_lbl) in enumerate(zip(row_panels, panel_labels)):
            import matplotlib.image as mpimg
            ax = axes[row_i, col_i]
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(panel_lbl, fontsize=10, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(row_lbl, fontsize=9, labelpad=4)
            os.unlink(img_path)

    # Colorbars: one for base/lora, one for diff
    for panels_idx, (cmap_name, vmax_val, label_str, x_pos) in enumerate([
        ("RdBu_r", vmax_both, f"BOLD (z-score)  vmax={vmax_both:.3f}", 0.25),
        ("PiYG",   vmax_diff, f"Δ BOLD (LoRA−Base)  vmax={vmax_diff:.3f}", 0.75),
    ]):
        sm   = ScalarMappable(cmap=cmap_name, norm=Normalize(vmin=-vmax_val, vmax=vmax_val))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes[2 if panels_idx == 1 else 1, :].ravel().tolist(),
            orientation="horizontal",
            fraction=0.025, pad=0.08, shrink=0.45,
            location="bottom",
        )
        cbar.set_label(label_str, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(save_dir, "fig3_brain_surface.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Global BOLD, hemisphere, heatmap comparison
# ---------------------------------------------------------------------------

def fig_global_comparison(base_preds: np.ndarray, lora_preds: np.ndarray,
                           state: dict, save_dir: str, step: int):
    min_t = min(base_preds.shape[0], lora_preds.shape[0])
    b = base_preds[:min_t]
    l = lora_preds[:min_t]
    t = np.arange(min_t)

    base_global   = b.mean(axis=1)
    lora_global   = l.mean(axis=1)
    base_left_ts  = b[:, :10242].mean(axis=1)
    lora_left_ts  = l[:, :10242].mean(axis=1)
    base_right_ts = b[:, 10242:].mean(axis=1)
    lora_right_ts = l[:, 10242:].mean(axis=1)
    diff_mean_t   = (l - b).mean(axis=1)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Global fMRI Comparison — Base vs LoRA (Step {step})\n"
        f"Base: {BASE_MODEL_ID}  |  LoRA trained on: {TARGET_ROI.upper()}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # --- Panel 1: Global mean BOLD ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, base_global, color="tomato",    linewidth=1.5, label="Base",  alpha=0.9)
    ax1.plot(t, lora_global, color="steelblue", linewidth=1.5, label="LoRA",  alpha=0.9)
    ax1.fill_between(t, base_global, lora_global, alpha=0.15, color="purple",
                     label="LoRA − Base")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.set_title("Global Mean BOLD Over Time", fontweight="bold")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mean BOLD (z-score)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    delta_g = float(lora_global.mean() - base_global.mean())
    ax1.text(0.02, 0.97, f"Δ mean: {'+' if delta_g >= 0 else ''}{delta_g:.4f}",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Panel 2: Left hemisphere ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, base_left_ts, color="tomato",    linewidth=1.5, label="Base L")
    ax2.plot(t, lora_left_ts, color="steelblue", linewidth=1.5, label="LoRA L")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.set_title("Left Hemisphere", fontweight="bold")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mean BOLD (z-score)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    dl = float(lora_left_ts.mean() - base_left_ts.mean())
    ax2.text(0.02, 0.97, f"Δ mean: {'+' if dl >= 0 else ''}{dl:.4f}",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Panel 3: Right hemisphere ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, base_right_ts, color="tomato",    linewidth=1.5, label="Base R")
    ax3.plot(t, lora_right_ts, color="steelblue", linewidth=1.5, label="LoRA R")
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax3.set_title("Right Hemisphere", fontweight="bold")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Mean BOLD (z-score)")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)
    dr = float(lora_right_ts.mean() - base_right_ts.mean())
    ax3.text(0.02, 0.97, f"Δ mean: {'+' if dr >= 0 else ''}{dr:.4f}",
             transform=ax3.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Panel 4: BOLD difference over time ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, diff_mean_t, color="purple", linewidth=1.5)
    ax4.fill_between(t, 0, diff_mean_t, where=(diff_mean_t > 0),
                     alpha=0.25, color="steelblue", label="LoRA > Base")
    ax4.fill_between(t, 0, diff_mean_t, where=(diff_mean_t < 0),
                     alpha=0.25, color="tomato", label="Base > LoRA")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("Global BOLD Difference (LoRA − Base)", fontweight="bold")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("ΔBOLD (z-score)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)

    # --- Panel 5: Vertex heatmap (base) ---
    ax5 = fig.add_subplot(gs[1, 1])
    top_n    = 200
    top_verts = np.argsort(b.mean(axis=0))[-top_n:][::-1]
    base_heat = b[:, top_verts].T
    vlim = max(abs(base_heat.min()), abs(base_heat.max()))
    im5 = ax5.imshow(base_heat, aspect="auto", cmap="RdBu_r",
                     vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax5.set_title(f"Base: Top {top_n} Vertices by Mean Act.", fontweight="bold")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Vertex (ranked)")
    plt.colorbar(im5, ax=ax5, label="BOLD", fraction=0.03)

    # --- Panel 6: Vertex heatmap (LoRA) —  same vertex selection ---
    ax6 = fig.add_subplot(gs[1, 2])
    lora_heat = l[:, top_verts].T
    im6 = ax6.imshow(lora_heat, aspect="auto", cmap="RdBu_r",
                     vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax6.set_title(f"LoRA: Same {top_n} Vertices", fontweight="bold")
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Vertex (ranked)")
    plt.colorbar(im6, ax=ax6, label="BOLD", fraction=0.03)

    path = os.path.join(save_dir, "fig4_global_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 5 — Training trajectory from saved state
# ---------------------------------------------------------------------------

def fig_training_trajectory(state: dict, save_dir: str):
    step_rewards = state["step_rewards"]
    best_rewards = state["best_rewards"]
    step_losses  = state.get("step_losses",  [None] * len(step_rewards))
    step_kls     = state.get("step_kls",     [None] * len(step_rewards))
    step         = state["step"]

    means = [np.mean(r) for r in step_rewards]
    stds  = [np.std(r)  for r in step_rewards]
    steps = list(range(1, len(means) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Training Trajectory — {step} steps  |  Target: {TARGET_ROI.upper()}  |  "
        f"Best reward: {state['best_reward']:.4f}",
        fontsize=12, fontweight="bold",
    )

    # Reward
    ax = axes[0]
    ax.plot(steps, means, color="steelblue", linewidth=1.6, label="Mean reward/step")
    ax.fill_between(steps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.18, color="steelblue")
    ax.plot(steps, best_rewards, color="darkorange", linewidth=1.5, linestyle="--",
            label="Best reward")
    ax.axhline(means[0], color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
               label=f"Step-1 baseline: {means[0]:.3f}")
    ax.set_title("TRIBE Broca Reward", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("ROI activation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # Loss
    ax = axes[1]
    valid_losses = [(i+1, v) for i, v in enumerate(step_losses) if v is not None]
    if valid_losses:
        s_l, v_l = zip(*valid_losses)
        ax.plot(s_l, v_l, color="tomato", linewidth=1.5)
        ax.fill_between(s_l, v_l, min(v_l), alpha=0.15, color="tomato")
    ax.set_title("Policy Loss", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)

    # KL divergence
    ax = axes[2]
    valid_kls = [(i+1, v) for i, v in enumerate(step_kls) if v is not None]
    if valid_kls:
        s_k, v_k = zip(*valid_kls)
        ax.plot(s_k, v_k, color="purple", linewidth=1.5)
        ax.fill_between(s_k, v_k, 0, alpha=0.15, color="purple")
    ax.set_title("KL Divergence from Base Model", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("KL (nats)")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig5_training_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] Saved → {path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_roi_summary(base_preds: np.ndarray, lora_preds: np.ndarray,
                       atlas: dict, step: int):
    from tribe_inference import FRIENDLY_ROIS

    base_means = all_roi_means(base_preds, atlas)
    lora_means = all_roi_means(lora_preds, atlas)

    W = 78
    print("\n" + "=" * W)
    print(f"  ROI SUMMARY — Base vs LoRA (Step {step})  |  Target: {TARGET_ROI.upper()}")
    print("=" * W)
    print(f"  {'Region':<22}  {'Hemi':<10}  {'Base':>8}  {'LoRA':>8}  {'Δ':>8}  {'↑↓':<3}")
    print("  " + "-" * (W - 2))

    all_rois = sorted(FRIENDLY_ROIS.keys())
    for key in all_rois:
        b = base_means.get(key, 0.0)
        l = lora_means.get(key, 0.0)
        d = l - b
        hemi = FRIENDLY_ROIS[key]["hemi"]
        marker = "★" if key == TARGET_ROI else (" ↑" if d > 0 else " ↓")
        print(f"  {key:<22}  {hemi:<10}  {b:>8.4f}  {l:>8.4f}  {d:>+8.4f}  {marker}")

    print("=" * W)
    base_global = float(base_preds.mean())
    lora_global = float(lora_preds.mean())
    print(f"  Global mean  Base: {base_global:+.4f}  |  LoRA: {lora_global:+.4f}  |"
          f"  Δ: {lora_global - base_global:+.4f}")
    print(f"  Left hemi    Base: {base_preds[:, :10242].mean():+.4f}  |  "
          f"LoRA: {lora_preds[:, :10242].mean():+.4f}")
    print(f"  Right hemi   Base: {base_preds[:, 10242:].mean():+.4f}  |  "
          f"LoRA: {lora_preds[:, 10242:].mean():+.4f}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Save raw predictions
# ---------------------------------------------------------------------------

def save_predictions(base_preds: np.ndarray, lora_preds: np.ndarray,
                     base_text: str, lora_text: str,
                     base_means: dict, lora_means: dict,
                     save_dir: str, step: int):
    np.save(os.path.join(save_dir, "base_preds.npy"),  base_preds)
    np.save(os.path.join(save_dir, "lora_preds.npy"),  lora_preds)
    np.save(os.path.join(save_dir, "diff_preds.npy"),  lora_preds - base_preds[:lora_preds.shape[0]])

    summary = {
        "step":         step,
        "base_model":   BASE_MODEL_ID,
        "target_roi":   TARGET_ROI,
        "base_text":    base_text,
        "lora_text":    lora_text,
        "base_global_mean": float(base_preds.mean()),
        "lora_global_mean": float(lora_preds.mean()),
        "roi_means_base": base_means,
        "roi_means_lora": lora_means,
        "roi_deltas": {k: lora_means.get(k, 0) - base_means.get(k, 0)
                       for k in base_means},
    }
    with open(os.path.join(save_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Data] Predictions + summary saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full TRIBE fMRI comparison: base vs L40S LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint dir (default: latest in output_dir)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--save_dir",   type=str, default=SAVE_DIR)
    parser.add_argument("--cache",      type=str, default=TRIBE_CACHE)
    parser.add_argument("--prompt",     type=str, default=TRAIN_PROMPT)
    parser.add_argument("--n_completions", type=int, default=2,
                        help="Completions per model to average over (default 2)")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--mock_tribe",     action="store_true",
                        help="Use synthetic TRIBE predictions for fast testing")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Checkpoint ---
    checkpoint_dir = args.checkpoint or find_latest_checkpoint(args.output_dir)
    print(f"\nCheckpoint: {checkpoint_dir}")
    with open(Path(checkpoint_dir) / "state.json") as f:
        state = json.load(f)
    step = state["step"]
    print(f"Step: {step}   Best reward: {state['best_reward']:.4f}")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Figure 5 (fast, no models needed) ---
    print("\n[Fig 5] Plotting training trajectory...")
    fig_training_trajectory(state, args.save_dir)

    # --- Load models ---
    base_model, base_tok = load_base(device)
    lora_model, lora_tok = load_lora(checkpoint_dir, device)

    # --- Generate completions ---
    print(f"\nGenerating {args.n_completions} completion(s) from each model...")
    print(f"  Prompt: {args.prompt[:80]}...")

    base_texts = generate_n(base_model, base_tok, args.prompt, device,
                            n=args.n_completions,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature)
    lora_texts = generate_n(lora_model, lora_tok, args.prompt, device,
                            n=args.n_completions,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature)

    print("\n--- Base completions ---")
    for i, t in enumerate(base_texts):
        print(f"  [{i+1}] {t[:200]}...")

    print("\n--- LoRA completions ---")
    for i, t in enumerate(lora_texts):
        print(f"  [{i+1}] {t[:200]}...")

    # Save completions to disk
    with open(os.path.join(args.save_dir, "base_completions.txt"), "w") as f:
        for i, t in enumerate(base_texts):
            f.write(f"=== Completion {i+1} ===\n{t}\n\n")
    with open(os.path.join(args.save_dir, "lora_completions.txt"), "w") as f:
        for i, t in enumerate(lora_texts):
            f.write(f"=== Completion {i+1} ===\n{t}\n\n")

    # Free model memory before loading TRIBE (TRIBE uses LLaMA 3B + Wav2Vec)
    print("\nFreeing LM memory before loading TRIBE...")
    del base_model, lora_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # --- Load TRIBE ---
    if args.mock_tribe:
        print("\n[MOCK] Using synthetic TRIBE predictions...")
        rng  = np.random.RandomState(42)
        n_t  = 30   # ~30 seconds of audio for typical completion
        base_preds_list = [rng.randn(n_t, 20484).astype(np.float32) * 0.5
                           for _ in base_texts]
        lora_preds_list = [rng.randn(n_t, 20484).astype(np.float32) * 0.5 + 0.3
                           for _ in lora_texts]
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        print("\nLoading TRIBE v2 (facebook/tribev2)...")
        from tribe_inference import load_model as _ltm
        tribe_model = _ltm(cache_folder=args.cache)

        print("\nRunning TRIBE on base completions...")
        base_preds_list = run_tribe_on_completions(tribe_model, base_texts, "Base")

        print("\nRunning TRIBE on LoRA completions...")
        lora_preds_list = run_tribe_on_completions(tribe_model, lora_texts, "LoRA")

        del tribe_model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Average over completions
    base_preds = avg_preds(base_preds_list)
    lora_preds = avg_preds(lora_preds_list)

    print(f"\nAveraged predictions — Base: {base_preds.shape}  LoRA: {lora_preds.shape}")

    # Load atlas once
    print("\nLoading Destrieux atlas...")
    from tribe_inference import load_destrieux_atlas, FRIENDLY_ROIS
    atlas = load_destrieux_atlas()

    # Summary stats
    print_roi_summary(base_preds, lora_preds, atlas, step)

    # Save raw data
    base_means_dict = all_roi_means(base_preds, atlas)
    lora_means_dict = all_roi_means(lora_preds, atlas)
    save_predictions(base_preds, lora_preds,
                     base_texts[0], lora_texts[0],
                     base_means_dict, lora_means_dict,
                     args.save_dir, step)

    # --- Figures ---
    print("\nGenerating figures...")
    fig_all_roi_timeseries(base_preds, lora_preds, atlas, args.save_dir, step)
    fig_roi_bar_chart(base_preds, lora_preds, atlas, args.save_dir, step)
    fig_brain_surface(base_preds, lora_preds, args.save_dir, step)
    fig_global_comparison(base_preds, lora_preds, state, args.save_dir, step)

    print(f"\n{'='*60}")
    print(f"  All done! Outputs in: {args.save_dir}/")
    print(f"  fig1_all_roi_timeseries.png  — all 20 ROI comparisons")
    print(f"  fig2_roi_bar_chart.png       — mean activation + deltas")
    print(f"  fig3_brain_surface.png       — cortical surface maps")
    print(f"  fig4_global_comparison.png   — global/hemisphere/heatmap")
    print(f"  fig5_training_trajectory.png — reward/loss/KL over 200 steps")
    print(f"  base_preds.npy / lora_preds.npy / diff_preds.npy")
    print(f"  comparison_summary.json")
    print(f"{'='*60}\n")
