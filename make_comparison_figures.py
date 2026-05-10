"""
make_comparison_figures.py — Generate comparison figures from saved predictions.

Loads base_preds.npy / lora_preds.npy from a previous compare_l40s_tribe_full.py
run and generates the 4 missing figures + comparison_summary.json.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import tempfile

SAVE_DIR       = "./comparison-plots/l40s_full_tribe"
OUTPUT_DIR     = "./brain-optimize-output-l40s"
BASE_MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
TARGET_ROI     = "broca"

sys.path.insert(0, str(Path(__file__).parent))
from tribe_inference import (
    FRIENDLY_ROIS, load_destrieux_atlas, extract_region_activity,
)

# ---------------------------------------------------------------------------

def all_roi_means(preds, atlas):
    out = {}
    for key in FRIENDLY_ROIS:
        try:
            ts = extract_region_activity(preds, key, atlas)
            out[key] = float(ts.mean())
        except Exception:
            out[key] = 0.0
    return out

# ---------------------------------------------------------------------------
# Fig 1 — All-ROI timeseries
# ---------------------------------------------------------------------------

def fig_all_roi_timeseries(base_preds, lora_preds, atlas, save_dir, step):
    roi_keys = sorted(FRIENDLY_ROIS.keys())
    ncols = 4
    nrows = int(np.ceil(len(roi_keys) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.2))
    fig.suptitle(
        f"All-ROI fMRI Timeseries — Base vs LoRA (Step {step})\n"
        f"Base: {BASE_MODEL_ID}  |  Training region: {TARGET_ROI.upper()} (highlighted gold)",
        fontsize=13, fontweight="bold",
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
        min_t = min(len(base_ts), len(lora_ts))
        ax.fill_between(np.arange(min_t), base_ts[:min_t], lora_ts[:min_t],
                        alpha=0.12, color="purple")
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.2)

        desc  = FRIENDLY_ROIS[roi_key]["description"].split("(")[0].strip()
        hemi  = FRIENDLY_ROIS[roi_key]["hemi"]
        delta = lora_ts.mean() - base_ts.mean()
        sign  = "+" if delta >= 0 else ""
        is_target = (roi_key == TARGET_ROI)

        ax.set_title(f"{desc}\n[{hemi}]  Δ={sign}{delta:.3f}",
                     fontsize=8, fontweight="bold" if is_target else "normal")
        ax.set_xlabel("Time (s)", fontsize=6)
        ax.set_ylabel("BOLD (z)", fontsize=6)
        ax.tick_params(labelsize=6)

        if is_target:
            for spine in ax.spines.values():
                spine.set_edgecolor("gold"); spine.set_linewidth(2.5)
            ax.set_facecolor("#fffdf0")

    for ax in axes.flat[len(roi_keys):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="tomato",    linewidth=2, label="Base (untrained)"),
        plt.Line2D([0], [0], color="steelblue", linewidth=2, label=f"LoRA (step {step})"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9)

    path = os.path.join(save_dir, "fig1_all_roi_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] → {path}")

# ---------------------------------------------------------------------------
# Fig 2 — ROI bar chart
# ---------------------------------------------------------------------------

def fig_roi_bar_chart(base_preds, lora_preds, atlas, save_dir, step):
    base_means = all_roi_means(base_preds, atlas)
    lora_means = all_roi_means(lora_preds, atlas)
    roi_keys   = sorted(FRIENDLY_ROIS.keys())
    base_vals  = np.array([base_means[k] for k in roi_keys])
    lora_vals  = np.array([lora_means[k] for k in roi_keys])
    deltas     = lora_vals - base_vals

    short_names = [FRIENDLY_ROIS[k]["description"].split("(")[0].strip()[:22]
                   for k in roi_keys]
    sort_idx    = np.argsort(deltas)[::-1]
    roi_keys_s  = [roi_keys[i]    for i in sort_idx]
    short_s     = [short_names[i] for i in sort_idx]
    base_s      = base_vals[sort_idx]
    lora_s      = lora_vals[sort_idx]
    deltas_s    = deltas[sort_idx]

    n, h = len(roi_keys_s), 0.35
    y    = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, n * 0.45)))
    fig.suptitle(
        f"Mean ROI Activation — Base vs LoRA (Step {step})\n"
        f"Sorted by LoRA − Base  |  Target: {TARGET_ROI.upper()} (highlighted)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.barh(y + h/2, base_s, h, color="tomato",    label="Base",          alpha=0.7)
    ax.barh(y - h/2, lora_s, h, color="steelblue", label=f"LoRA step {step}", alpha=0.7)
    for i, (bv, lv) in enumerate(zip(base_s, lora_s)):
        xr = max(abs(bv), abs(lv)) + 0.003
        ax.text(xr, y[i] + h/2, f"{bv:.3f}", va="center", fontsize=6.5, color="tomato")
        ax.text(xr, y[i] - h/2, f"{lv:.3f}", va="center", fontsize=6.5, color="steelblue")
    ax.set_yticks(y); ax.set_yticklabels(short_s, fontsize=8)
    ax.set_xlabel("Mean BOLD (z-score)")
    ax.set_title("Absolute Activation by Region", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="x"); ax.axvline(0, color="black", linewidth=0.6)
    target_row = next((i for i, k in enumerate(roi_keys_s) if k == TARGET_ROI), None)
    if target_row is not None:
        ax.axhspan(target_row - 0.5, target_row + 0.5, alpha=0.12, color="gold")
        ax.text(ax.get_xlim()[0], target_row, " ★", va="center", fontsize=9, color="goldenrod")

    ax2 = axes[1]
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas_s]
    ax2.barh(y, deltas_s, color=colors, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y); ax2.set_yticklabels(short_s, fontsize=8)
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("ΔBOLD  (LoRA − Base)")
    ax2.set_title("Activation Change due to LoRA Training", fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")
    for i, d in enumerate(deltas_s):
        sign = "+" if d >= 0 else ""
        ax2.text(d + (0.002 if d >= 0 else -0.002), y[i],
                 f"{sign}{d:.3f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=6.5,
                 color="steelblue" if d >= 0 else "tomato")
    if target_row is not None:
        ax2.axhspan(target_row - 0.5, target_row + 0.5, alpha=0.12, color="gold")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig2_roi_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] → {path}")

# ---------------------------------------------------------------------------
# Fig 3 — Brain surface maps
# ---------------------------------------------------------------------------

def _render_panels(mean_act, vmax, cmap):
    from nilearn import plotting, datasets
    fsaverage  = datasets.fetch_surf_fsaverage("fsaverage5")
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
        sf = plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh, stat_map=data, hemi=hemi, view=view,
            bg_map=bg_map, bg_on_data=True, colorbar=False,
            cmap=cmap, vmax=vmax, engine="matplotlib",
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = tmp.name
        sf.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(sf)
        paths.append(p)
    return paths


def fig_brain_surface(base_preds, lora_preds, save_dir, step):
    try:
        from nilearn import plotting, datasets
        import matplotlib.image as mpimg
    except ImportError:
        print("[Fig 3] nilearn not installed — skipping"); return

    base_mean = base_preds.mean(axis=0)
    lora_mean = lora_preds.mean(axis=0)
    diff_mean = lora_mean - base_mean

    vmax_both = max(float(np.abs(base_mean).max()), float(np.abs(lora_mean).max())) or 1.0
    vmax_diff = float(np.abs(diff_mean).max()) or 1.0

    print("  Rendering base surface..."); base_panels = _render_panels(base_mean, vmax_both, "RdBu_r")
    print("  Rendering LoRA surface..."); lora_panels = _render_panels(lora_mean, vmax_both, "RdBu_r")
    print("  Rendering diff surface..."); diff_panels = _render_panels(diff_mean, vmax_diff, "PiYG")

    panel_labels = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    row_labels   = [f"Base\n({BASE_MODEL_ID})", f"LoRA\n(step {step})", "Difference\n(LoRA − Base)"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        f"Cortical Surface Maps — Base vs LoRA (Step {step})\n"
        "Time-averaged BOLD on fsaverage5 inflated  |  "
        "Rows: Base | LoRA | Diff",
        fontsize=12, fontweight="bold",
    )
    import matplotlib.image as mpimg
    for row_i, (panels, rlbl) in enumerate(zip([base_panels, lora_panels, diff_panels], row_labels)):
        for col_i, (img_path, plbl) in enumerate(zip(panels, panel_labels)):
            ax = axes[row_i, col_i]
            ax.imshow(mpimg.imread(img_path)); ax.axis("off")
            if row_i == 0: ax.set_title(plbl, fontsize=10, fontweight="bold")
            if col_i == 0: ax.set_ylabel(rlbl, fontsize=9)
            os.unlink(img_path)

    for cmap_n, vmax_v, lbl, ax_rows in [
        ("RdBu_r", vmax_both, f"BOLD z-score (vmax={vmax_both:.3f})", axes[:2, :]),
        ("PiYG",   vmax_diff, f"ΔBOLD LoRA−Base (vmax={vmax_diff:.3f})", axes[2:, :]),
    ]:
        sm = ScalarMappable(cmap=cmap_n, norm=Normalize(vmin=-vmax_v, vmax=vmax_v))
        sm.set_array([])
        fig.colorbar(sm, ax=ax_rows.ravel().tolist(),
                     orientation="horizontal", fraction=0.025, pad=0.06,
                     shrink=0.45, location="bottom").set_label(lbl, fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = os.path.join(save_dir, "fig3_brain_surface.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] → {path}")

# ---------------------------------------------------------------------------
# Fig 4 — Global comparison
# ---------------------------------------------------------------------------

def fig_global_comparison(base_preds, lora_preds, state, save_dir, step):
    min_t = min(base_preds.shape[0], lora_preds.shape[0])
    b, l  = base_preds[:min_t], lora_preds[:min_t]
    t     = np.arange(min_t)

    base_g = b.mean(axis=1); lora_g = l.mean(axis=1)
    base_L = b[:, :10242].mean(axis=1); lora_L = l[:, :10242].mean(axis=1)
    base_R = b[:, 10242:].mean(axis=1); lora_R = l[:, 10242:].mean(axis=1)
    diff_t = (l - b).mean(axis=1)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Global fMRI Comparison — Base vs LoRA (Step {step})\n"
        f"Base: {BASE_MODEL_ID}  |  LoRA trained on: {TARGET_ROI.upper()}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def annotate(ax, val):
        s = "+" if val >= 0 else ""
        ax.text(0.02, 0.97, f"Δ mean: {s}{val:.4f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, base_g, color="tomato",    linewidth=1.5, label="Base")
    ax1.plot(t, lora_g, color="steelblue", linewidth=1.5, label="LoRA")
    ax1.fill_between(t, base_g, lora_g, alpha=0.15, color="purple", label="Difference")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.set_title("Global Mean BOLD", fontweight="bold")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mean BOLD (z-score)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    annotate(ax1, float(lora_g.mean() - base_g.mean()))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, base_L, color="tomato",    linewidth=1.5, label="Base L")
    ax2.plot(t, lora_L, color="steelblue", linewidth=1.5, label="LoRA L")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.set_title("Left Hemisphere", fontweight="bold")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mean BOLD (z-score)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    annotate(ax2, float(lora_L.mean() - base_L.mean()))

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, base_R, color="tomato",    linewidth=1.5, label="Base R")
    ax3.plot(t, lora_R, color="steelblue", linewidth=1.5, label="LoRA R")
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax3.set_title("Right Hemisphere", fontweight="bold")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Mean BOLD (z-score)")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)
    annotate(ax3, float(lora_R.mean() - base_R.mean()))

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, diff_t, color="purple", linewidth=1.5)
    ax4.fill_between(t, 0, diff_t, where=(diff_t > 0), alpha=0.25, color="steelblue", label="LoRA > Base")
    ax4.fill_between(t, 0, diff_t, where=(diff_t < 0), alpha=0.25, color="tomato",    label="Base > LoRA")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("Global BOLD Difference (LoRA − Base)", fontweight="bold")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("ΔBOLD (z-score)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)

    top_n    = 200
    top_v    = np.argsort(b.mean(axis=0))[-top_n:][::-1]
    vlim     = float(max(abs(b[:, top_v].min()), abs(b[:, top_v].max())))

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(b[:, top_v].T, aspect="auto", cmap="RdBu_r",
                     vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax5.set_title(f"Base: Top {top_n} Vertices", fontweight="bold")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Vertex (ranked)")
    plt.colorbar(im5, ax=ax5, label="BOLD", fraction=0.03)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(l[:, top_v].T, aspect="auto", cmap="RdBu_r",
                     vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax6.set_title(f"LoRA: Same Top {top_n} Vertices", fontweight="bold")
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Vertex (ranked)")
    plt.colorbar(im6, ax=ax6, label="BOLD", fraction=0.03)

    path = os.path.join(save_dir, "fig4_global_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] → {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading saved predictions...")
    base_preds = np.load(os.path.join(SAVE_DIR, "base_preds.npy"))
    lora_preds = np.load(os.path.join(SAVE_DIR, "lora_preds.npy"))
    print(f"  base_preds: {base_preds.shape}   lora_preds: {lora_preds.shape}")

    # Load state for step number and training trajectory
    checkpoint_dir = None
    ckpt_root = Path(OUTPUT_DIR) / "checkpoints"
    for d in sorted(ckpt_root.glob("step_*"), reverse=True):
        if (d / "state.json").exists():
            checkpoint_dir = str(d); break
    with open(Path(checkpoint_dir) / "state.json") as f:
        state = json.load(f)
    step = state["step"]
    print(f"  Checkpoint step: {step}")

    print("\nLoading Destrieux atlas...")
    atlas = load_destrieux_atlas()

    # Save diff (with correct shape alignment)
    min_t = min(base_preds.shape[0], lora_preds.shape[0])
    diff_preds = lora_preds[:min_t] - base_preds[:min_t]
    np.save(os.path.join(SAVE_DIR, "diff_preds.npy"), diff_preds)

    # Save JSON summary
    base_means = all_roi_means(base_preds, atlas)
    lora_means = all_roi_means(lora_preds, atlas)

    # Load completions if available
    def _read(fname):
        p = os.path.join(SAVE_DIR, fname)
        return open(p).read().split("=== Completion 2 ===")[0].replace("=== Completion 1 ===\n", "").strip() if os.path.exists(p) else ""

    summary = {
        "step": step,
        "base_model": BASE_MODEL_ID,
        "target_roi": TARGET_ROI,
        "base_text": _read("base_completions.txt"),
        "lora_text": _read("lora_completions.txt"),
        "base_global_mean": float(base_preds.mean()),
        "lora_global_mean": float(lora_preds.mean()),
        "roi_means_base": base_means,
        "roi_means_lora": lora_means,
        "roi_deltas": {k: lora_means.get(k, 0) - base_means.get(k, 0) for k in base_means},
    }
    with open(os.path.join(SAVE_DIR, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Data] comparison_summary.json saved")

    print("\nGenerating figures...")
    fig_all_roi_timeseries(base_preds, lora_preds, atlas, SAVE_DIR, step)
    fig_roi_bar_chart(base_preds, lora_preds, atlas, SAVE_DIR, step)
    fig_brain_surface(base_preds, lora_preds, SAVE_DIR, step)
    fig_global_comparison(base_preds, lora_preds, state, SAVE_DIR, step)

    print(f"\nDone. All outputs in {SAVE_DIR}/")
