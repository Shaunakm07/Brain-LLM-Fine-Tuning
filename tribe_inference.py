"""
tribe_inference.py — Run TRIBE v2 on text and plot predicted brain activity
============================================================================

WHAT THIS SCRIPT DOES
---------------------
Loads the TRIBE v2 model (facebook/tribev2) and runs it on text input to
produce predicted fMRI BOLD responses across the entire cortical surface.

TRIBE v2 (Tri-modal Brain Encoding) is a foundation model from Meta AI that
predicts what brain activity a human would show while perceiving a stimulus.
For text input, it produces a (time × 20484 vertices) array of predicted
cortical activity across the fsaverage5 surface mesh.

HOW TEXT IS PROCESSED
---------------------
Text does not go directly into the model. The pipeline is:
  1. Your text is synthesised to speech using gTTS (requires internet)
  2. WhisperX extracts word-level timestamps from the synthesised audio
  3. LLaMA 3.2-3B extracts text features at 6 layer fractions
  4. Wav2Vec-BERT extracts audio features from the synthesised speech
  5. Both pathways feed into a fusion Transformer → cortical predictions

Output shape: (n_seconds, 20484) where 20484 = fsaverage5 cortical vertices

REQUIREMENTS
------------
  brew install ffmpeg                  # required for audio decoding (torchcodec)
  pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
  pip install numpy matplotlib scipy

  LLaMA access: you must accept the LLaMA 3.2 license on HuggingFace and run:
    huggingface-cli login

MAC / CPU NOTE
--------------
  ctranslate2 (used by WhisperX internally) defaults to float16 compute type,
  which is not supported on CPU or Apple Silicon. This script patches the
  subprocess call to force --compute_type int8 before tribev2 is imported.

USAGE
-----
  # Run on a text file
  python tribe_inference.py --text path/to/text.txt

  # Run on a string directly (written to a temp file)
  python tribe_inference.py --prompt "The apple fell from the tree."

  # Save the raw predictions as a numpy array
  python tribe_inference.py --prompt "Explain gravity" --output preds.npy

  # Use a pre-existing audio file (skips TTS synthesis)
  python tribe_inference.py --audio path/to/audio.wav
"""

import os
import warnings
import argparse
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Mac / CPU compatibility patch — must run before tribev2 is imported
# ---------------------------------------------------------------------------
#
# tribev2 calls WhisperX as a subprocess to transcribe synthesised speech.
# WhisperX defaults to float16 compute type via ctranslate2, which crashes
# on Mac CPU with:
#   ValueError: Requested float16 compute type, but the target device or
#   backend do not support efficient float16 computation.
#
# Fix: intercept subprocess.run and inject --compute_type int8 whenever a
# whisperx command is detected. int8 is fully supported on CPU and only
# marginally slower than float16 on GPU.

_original_subprocess_run = subprocess.run

def _patched_subprocess_run(args, **kwargs):
    if isinstance(args, (list, tuple)):
        args = list(args)
        cmd_str = " ".join(str(a) for a in args)
        if "whisperx" in cmd_str:
            # tribev2 hardcodes compute_type="float16" regardless of device.
            # float16 crashes on Mac/CPU with ctranslate2. Replace with int8,
            # which is fully supported on CPU.
            if "--compute_type" in args:
                idx = args.index("--compute_type")
                if idx + 1 < len(args) and args[idx + 1] == "float16":
                    args[idx + 1] = "int8"
            else:
                args = args + ["--compute_type", "int8"]
    return _original_subprocess_run(args, **kwargs)

subprocess.run = _patched_subprocess_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(cache_folder: str = "./tribe-cache"):
    """
    Download and load TRIBE v2 from HuggingFace.

    Downloads two files from facebook/tribev2:
      - best.ckpt   (709 MB) — model weights
      - config.yaml (18 kB)  — architecture configuration

    Args:
        cache_folder: Where to cache the downloaded weights.
    """
    try:
        from tribev2.demo_utils import TribeModel
    except ImportError:
        raise ImportError(
            "tribev2 is not installed. Run:\n"
            '  pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"'
        )

    # Determine best available device.
    # Use cpu even on Apple Silicon — LLaMA 3.2-3B has known MPS issues and
    # the image sub-configs in tribev2 use strict pydantic models that reject
    # device overrides via the nested dot-path config_update mechanism.
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config.yaml hardcodes device: cuda for the main feature extractors.
    # Only the top-level device fields can be safely overridden via config_update;
    # the nested image sub-configs use strict pydantic and reject extra fields.
    config_update = {
        "data.text_feature.device":  device,
        "data.audio_feature.device": device,
    }

    print(f"Loading TRIBE v2 from facebook/tribev2 (device: {device})...")
    print("  (First run downloads ~710 MB checkpoint — this may take a few minutes)")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_folder,
        device=device,
        config_update=config_update,
    )
    print("Model ready.\n")
    return model


def run_on_text(model, text_path: str):
    """
    Run TRIBE v2 on a text file and return predictions.

    Internally this:
      1. Detects language of the text
      2. Synthesises the text to speech (gTTS — requires internet)
      3. Extracts word timestamps via WhisperX
      4. Runs LLaMA 3.2-3B text features + Wav2Vec-BERT audio features
      5. Fuses both through the transformer → cortical predictions

    Args:
        model:     Loaded TribeModel instance
        text_path: Path to a .txt file containing the stimulus text

    Returns:
        preds:    numpy array, shape (n_timesteps, 20484)
        segments: list of aligned segment objects
    """
    print(f"Processing text: {text_path}")
    print("  Step 1/3: Synthesising text to speech (gTTS)...")
    df = model.get_events_dataframe(text_path=text_path)

    print(f"  Step 2/3: Running TRIBE inference ({len(df)} events)...")
    preds, segments = model.predict(events=df)

    print(f"  Step 3/3: Done. Output shape: {preds.shape}")
    print(f"            {preds.shape[0]} timesteps × {preds.shape[1]} cortical vertices\n")
    return preds, segments


def run_on_audio(model, audio_path: str):
    """Run TRIBE v2 on an existing audio file (skips TTS synthesis)."""
    print(f"Processing audio: {audio_path}")
    df = model.get_events_dataframe(audio_path=audio_path)
    preds, segments = model.predict(events=df)
    print(f"Output shape: {preds.shape}\n")
    return preds, segments


# ---------------------------------------------------------------------------
# Atlas-based ROI extraction  (Destrieux atlas on fsaverage5)
# ---------------------------------------------------------------------------
#
# TRIBE v2 outputs predictions on the fsaverage5 cortical surface mesh:
#   - 20484 vertices total  (10242 per hemisphere)
#   - Left  hemisphere: vertex indices 0 – 10241
#   - Right hemisphere: vertex indices 10242 – 20483
#
# We use the Destrieux (aparc.a2009s) parcellation, which nilearn provides
# natively on fsaverage5.  Each hemisphere has 75 labelled regions covering
# the entire cortical surface.  See:
#   Destrieux et al. (2010) Neuroimage 53:1–15
#   https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_surf_destrieux.html
#
# FRIENDLY_ROIS maps short, human-readable names to one or more Destrieux
# label strings and the hemisphere(s) to include.  "bilateral" combines
# both hemispheres; "left" / "right" use only one hemisphere.
#
# Destrieux gyrus/sulcus naming convention:
#   G_  = gyrus    S_  = sulcus    Lat_Fis = lateral fissure
#   front = frontal   temp = temporal   pariet = parietal   occipital
#   inf/sup/mid = inferior/superior/middle

FRIENDLY_ROIS = {
    # Language production — Broca's area (pars opercularis + pars triangularis)
    "broca": {
        "labels": ["G_front_inf-Opercular", "G_front_inf-Triangul"],
        "hemi": "left",
        "description": "Broca's area (IFG pars opercularis + triangularis, left)",
    },
    # Language comprehension — Wernicke's area (posterior STG + STS)
    "wernicke": {
        "labels": ["G_temp_sup-Lateral", "S_temporal_sup"],
        "hemi": "left",
        "description": "Wernicke's area (posterior superior temporal, left)",
    },
    # Primary auditory cortex — Heschl's gyrus (transverse temporal gyrus)
    "auditory": {
        "labels": ["G_temp_sup-G_T_transv", "S_temporal_transverse"],
        "hemi": "bilateral",
        "description": "Primary auditory cortex (Heschl's gyrus, bilateral)",
    },
    # Superior temporal sulcus — multimodal speech/language integration
    "sts": {
        "labels": ["Lat_Fis-post", "G_temp_sup-Plan_tempo"],
        "hemi": "bilateral",
        "description": "Superior temporal sulcus / planum temporale (bilateral)",
    },
    # Angular gyrus — semantic integration, reading, default mode
    "angular": {
        "labels": ["G_pariet_inf-Angular"],
        "hemi": "left",
        "description": "Angular gyrus (left inferior parietal)",
    },
    # Supramarginal gyrus — phonological working memory
    "supramarginal": {
        "labels": ["G_pariet_inf-Supramar"],
        "hemi": "left",
        "description": "Supramarginal gyrus (left inferior parietal)",
    },
    # Default mode network core — posterior cingulate + precuneus
    "default_mode": {
        "labels": ["G_cingul-Post-dorsal", "G_cingul-Post-ventral", "G_precuneus"],
        "hemi": "bilateral",
        "description": "Default mode network (posterior cingulate + precuneus)",
    },
    # Anterior cingulate — cognitive control, salience
    "anterior_cingulate": {
        "labels": ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
        "hemi": "bilateral",
        "description": "Anterior cingulate cortex",
    },
    # Dorsolateral prefrontal — working memory, executive function
    "dlpfc": {
        "labels": ["G_front_middle", "S_front_middle"],
        "hemi": "bilateral",
        "description": "Dorsolateral prefrontal cortex",
    },
    # Superior frontal gyrus — high-level cognition, self-referential processing
    "superior_frontal": {
        "labels": ["G_front_sup", "S_front_sup"],
        "hemi": "bilateral",
        "description": "Superior frontal gyrus",
    },
    # Primary visual cortex — calcarine sulcus (V1)
    "v1": {
        "labels": ["S_calcarine"],
        "hemi": "bilateral",
        "description": "Primary visual cortex (V1, calcarine sulcus)",
    },
    # Lateral occipital — higher-order visual processing
    "lateral_occipital": {
        "labels": ["G_occipital_middle", "G_occipital_sup", "S_oc_middle_and_Lunatus"],
        "hemi": "bilateral",
        "description": "Lateral occipital cortex",
    },
    # Fusiform gyrus — word-form recognition, face processing
    "fusiform": {
        "labels": ["G_oc-temp_lat-fusifor"],
        "hemi": "bilateral",
        "description": "Fusiform gyrus (visual word-form area region)",
    },
    # Parahippocampal gyrus — scene/context memory
    "parahippocampal": {
        "labels": ["G_oc-temp_med-Parahip"],
        "hemi": "bilateral",
        "description": "Parahippocampal gyrus",
    },
    # Insula — salience, interoception, speech articulation
    "insula": {
        "labels": ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
        "hemi": "bilateral",
        "description": "Insula",
    },
    # Middle temporal gyrus — semantic memory, narrative
    "middle_temporal": {
        "labels": ["G_temporal_middle", "S_temporal_inf"],
        "hemi": "bilateral",
        "description": "Middle temporal gyrus",
    },
    # Inferior temporal gyrus — object recognition, semantic access
    "inferior_temporal": {
        "labels": ["G_temporal_inf"],
        "hemi": "bilateral",
        "description": "Inferior temporal gyrus",
    },
    # Superior parietal lobule — spatial attention
    "superior_parietal": {
        "labels": ["G_parietal_sup", "S_intrapariet_and_P_trans"],
        "hemi": "bilateral",
        "description": "Superior parietal lobule",
    },
    # Motor cortex — precentral gyrus
    "motor": {
        "labels": ["G_precentral", "S_precentral-inf-part", "S_precentral-sup-part"],
        "hemi": "bilateral",
        "description": "Primary motor cortex (precentral gyrus)",
    },
    # Somatosensory cortex
    "somatosensory": {
        "labels": ["G_postcentral", "S_postcentral", "S_central"],
        "hemi": "bilateral",
        "description": "Primary somatosensory cortex (postcentral gyrus)",
    },
}

# Module-level cache so the atlas is only loaded once per process
_ATLAS_CACHE: dict = {}


def load_destrieux_atlas() -> dict:
    """
    Load the Destrieux surface parcellation for fsaverage5.

    Downloads on first call (~500 KB), then cached on disk by nilearn.
    Returns a dict with keys:
        'labels_left'  : (10242,) int array — Destrieux label index per vertex, left hemi
        'labels_right' : (10242,) int array — same for right hemisphere
        'label_names'  : list[str]           — label name for each index (index 0 = Unknown)

    The label_names list has 76 entries (0 = Unknown, 1–75 = named Destrieux regions).
    Vertex indices in labels_left/right range from 1 to 75; 0 = unlabelled medial wall.
    """
    global _ATLAS_CACHE
    if _ATLAS_CACHE:
        return _ATLAS_CACHE

    try:
        from nilearn import datasets, surface as nilearn_surface
    except ImportError:
        raise ImportError("nilearn is required for atlas ROI extraction. Run: pip install nilearn")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        destrieux = datasets.fetch_atlas_surf_destrieux()

    # map_left / map_right are already (10242,) numpy arrays in nilearn 0.13
    labels_left  = np.asarray(destrieux["map_left"],  dtype=np.int32)
    labels_right = np.asarray(destrieux["map_right"], dtype=np.int32)

    _ATLAS_CACHE = {
        "labels_left":  labels_left,
        "labels_right": labels_right,
        "label_names":  list(destrieux["labels"]),
    }
    return _ATLAS_CACHE


def _resolve_vertices(roi_key: str, atlas: dict) -> np.ndarray:
    """
    Return the global vertex indices (0–20483) for a named ROI.

    Left hemisphere vertices: 0–10241 (atlas labels_left)
    Right hemisphere vertices: 10242–20483 (atlas labels_right + 10242 offset)

    Args:
        roi_key: Key in FRIENDLY_ROIS (e.g. "broca", "auditory")
        atlas:   Dict returned by load_destrieux_atlas()

    Returns:
        1-D integer array of vertex indices into the (20484,) prediction array.
    """
    if roi_key not in FRIENDLY_ROIS:
        available = ", ".join(sorted(FRIENDLY_ROIS))
        raise ValueError(f"Unknown ROI '{roi_key}'. Available: {available}")

    roi_def      = FRIENDLY_ROIS[roi_key]
    label_names  = atlas["label_names"]
    labels_left  = atlas["labels_left"]
    labels_right = atlas["labels_right"]

    # Resolve Destrieux label names → integer indices
    label_indices = []
    for name in roi_def["labels"]:
        if name in label_names:
            label_indices.append(label_names.index(name))
        # silently skip names not present in this atlas version

    if not label_indices:
        raise ValueError(f"None of the Destrieux labels for '{roi_key}' found in atlas.")

    hemi = roi_def["hemi"]
    verts = []

    if hemi in ("left", "bilateral"):
        for idx in label_indices:
            verts.extend(np.where(labels_left == idx)[0].tolist())

    if hemi in ("right", "bilateral"):
        for idx in label_indices:
            # right hemisphere vertices are offset by 10242 in the global array
            verts.extend((np.where(labels_right == idx)[0] + 10242).tolist())

    return np.array(sorted(set(verts)), dtype=np.int32)


def extract_region_activity(preds: np.ndarray, roi_key: str, atlas: dict | None = None) -> np.ndarray:
    """
    Extract mean BOLD timeseries for a named brain region.

    Uses the Destrieux atlas parcellation on fsaverage5 to identify which of
    the 20484 cortical vertices belong to the requested region, then averages
    predicted BOLD across those vertices at each timestep.

    Args:
        preds:   (n_timesteps, 20484) predicted BOLD array from TRIBE v2
        roi_key: Short region name — one of the keys in FRIENDLY_ROIS
                 e.g. "broca", "auditory", "default_mode"
        atlas:   Pre-loaded atlas dict (from load_destrieux_atlas()).
                 If None, loads the atlas automatically.

    Returns:
        (n_timesteps,) array of mean predicted BOLD in the requested region.

    Example:
        atlas  = load_destrieux_atlas()
        preds, _ = run_on_text(model, "path/to/text.txt")
        broca_ts = extract_region_activity(preds, "broca", atlas)
        print(f"Mean Broca activation: {broca_ts.mean():.4f}")
    """
    if atlas is None:
        atlas = load_destrieux_atlas()

    vertices = _resolve_vertices(roi_key, atlas)
    return preds[:, vertices].mean(axis=1)


def list_available_regions(atlas: dict | None = None) -> None:
    """
    Print a table of all available named brain regions with vertex counts.

    Args:
        atlas: Pre-loaded atlas dict. If None, loads automatically.
    """
    if atlas is None:
        atlas = load_destrieux_atlas()

    print(f"\n{'Region key':<22}  {'Hemi':<10}  {'Vertices':>8}  Description")
    print("-" * 80)
    for key in sorted(FRIENDLY_ROIS):
        try:
            verts = _resolve_vertices(key, atlas)
            hemi  = FRIENDLY_ROIS[key]["hemi"]
            desc  = FRIENDLY_ROIS[key]["description"]
            print(f"  {key:<20}  {hemi:<10}  {len(verts):>8}  {desc}")
        except ValueError as e:
            print(f"  {key:<20}  (error: {e})")
    print()


def roi_timeseries(preds: np.ndarray, atlas: dict | None = None) -> dict:
    """
    Extract mean BOLD timeseries for all FRIENDLY_ROIS.

    Uses the Destrieux atlas for accurate vertex selection.

    Args:
        preds: (n_timesteps, 20484) predicted BOLD array
        atlas: Pre-loaded atlas dict (loads automatically if None)

    Returns:
        dict mapping region key → mean timeseries array of shape (n_timesteps,)
    """
    if atlas is None:
        atlas = load_destrieux_atlas()
    return {
        key: extract_region_activity(preds, key, atlas)
        for key in FRIENDLY_ROIS
    }


def whole_brain_stats(preds: np.ndarray) -> dict:
    """Compute summary statistics across all vertices and timesteps."""
    return {
        "mean_activation":    preds.mean(axis=0),          # (20484,) — avg over time
        "peak_vertex":        int(np.argmax(preds.mean(axis=0))),
        "peak_activation":    float(preds.mean(axis=0).max()),
        "global_mean":        float(preds.mean()),
        "global_std":         float(preds.std()),
        "left_hemi_mean":     float(preds[:, :10242].mean()),
        "right_hemi_mean":    float(preds[:, 10242:].mean()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(preds: np.ndarray, output_dir: str, title_prefix: str = "",
                 atlas: dict | None = None):
    """
    Produce a 4-panel plot of TRIBE v2 predictions:

    Panel 1 — Global mean BOLD over time
        Shows the average predicted activity across all cortical vertices at
        each second of the stimulus. A rise indicates the model predicts more
        global cortical engagement at that moment.

    Panel 2 — ROI timeseries
        Mean predicted BOLD in 6 approximate regions of interest. For text input:
        - Language/temporal regions (left hemisphere) should show strong responses
        - Auditory regions (both hemispheres) respond to the synthesised speech
        - Visual/occipital regions may be relatively quiet

    Panel 3 — Vertex activation heatmap (sorted by mean activation)
        Each row is a cortical vertex (sorted by mean activation), each column
        is a timestep. Shows which vertices are most consistently active and
        how activation evolves over time.

    Panel 4 — Hemisphere comparison
        Left vs right hemisphere mean activation over time. Language processing
        is left-lateralised in most people — left hemisphere should respond more
        strongly to text/speech input than right.

    Args:
        preds:      (n_timesteps, 20484) predicted BOLD array
        output_dir: Directory to save the plot
        title_prefix: Optional string prepended to the plot title
    """
    os.makedirs(output_dir, exist_ok=True)
    n_timesteps, n_vertices = preds.shape
    time_axis = np.arange(n_timesteps)

    stats     = whole_brain_stats(preds)
    roi_ts    = roi_timeseries(preds, atlas=atlas)
    left_ts   = preds[:, :10242].mean(axis=1)
    right_ts  = preds[:, 10242:].mean(axis=1)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{title_prefix}TRIBE v2 — Predicted Cortical Activity\n"
        f"{n_timesteps}s stimulus | {n_vertices} vertices | "
        f"global mean={stats['global_mean']:.3f}  std={stats['global_std']:.3f}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: Global mean BOLD over time ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis, preds.mean(axis=1), color="steelblue", linewidth=1.5)
    ax1.fill_between(time_axis, preds.mean(axis=1), alpha=0.2, color="steelblue")
    ax1.set_title("Global Mean BOLD Over Time", fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Mean predicted BOLD (z-score)")
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: ROI timeseries (Destrieux atlas regions, subset for readability) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Show a curated subset of the most language-relevant regions
    display_rois = ["broca", "wernicke", "auditory", "default_mode",
                    "angular", "middle_temporal"]
    display_ts   = {k: roi_ts[k] for k in display_rois if k in roi_ts}
    colors = plt.cm.tab10(np.linspace(0, 1, len(display_ts)))
    for (roi_name, ts), color in zip(display_ts.items(), colors):
        label = FRIENDLY_ROIS[roi_name]["description"].split("(")[0].strip()
        ax2.plot(time_axis, ts, label=label, color=color, linewidth=1.2)
    ax2.set_title("Key Language/Auditory ROIs (Destrieux atlas)", fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Mean BOLD (z-score)")
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Vertex activation heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    mean_per_vertex = preds.mean(axis=0)
    top_n           = 300   # show top 300 most-active vertices
    top_indices     = np.argsort(mean_per_vertex)[-top_n:][::-1]
    heatmap_data    = preds[:, top_indices].T          # (top_n, n_timesteps)

    im = ax3.imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1.5, vmax=1.5,
    )
    ax3.set_title(f"Top {top_n} Vertices by Mean Activation", fontweight="bold")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Vertex (sorted by mean activation)")
    ax3.set_xticks(np.linspace(0, n_timesteps - 1, min(6, n_timesteps)).astype(int))
    ax3.set_xticklabels(np.linspace(0, n_timesteps - 1, min(6, n_timesteps)).astype(int))
    plt.colorbar(im, ax=ax3, label="BOLD (z-score)", fraction=0.03)

    # --- Panel 4: Left vs right hemisphere ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_axis, left_ts,  color="royalblue",  label="Left hemisphere",  linewidth=1.5)
    ax4.plot(time_axis, right_ts, color="tomato",     label="Right hemisphere", linewidth=1.5)
    ax4.fill_between(time_axis, left_ts, right_ts, alpha=0.15, color="purple",
                     label="L−R difference")
    ax4.set_title("Left vs Right Hemisphere", fontweight="bold")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Mean BOLD (z-score)")
    ax4.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Annotation: left lateralisation index
    lateralisation = float(left_ts.mean() - right_ts.mean())
    direction      = "left-lateralised" if lateralisation > 0 else "right-lateralised"
    ax4.text(0.02, 0.97, f"Lateralisation index: {lateralisation:+.3f} ({direction})",
             transform=ax4.transAxes, fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plot_path = os.path.join(output_dir, "tribe_predictions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_path}")

    # Print summary
    print("\n--- Prediction Summary ---")
    print(f"  Timesteps          : {n_timesteps}")
    print(f"  Cortical vertices  : {n_vertices}")
    print(f"  Global mean BOLD   : {stats['global_mean']:+.4f}")
    print(f"  Global std         : {stats['global_std']:.4f}")
    print(f"  Left hemi mean     : {stats['left_hemi_mean']:+.4f}")
    print(f"  Right hemi mean    : {stats['right_hemi_mean']:+.4f}")
    print(f"  Lateralisation     : {lateralisation:+.4f} ({direction})")
    print(f"  Peak vertex        : {stats['peak_vertex']} (mean BOLD = {stats['peak_activation']:.4f})")


def plot_brain_surface(preds: np.ndarray, output_dir: str, title_prefix: str = ""):
    """
    Render mean predicted BOLD on the fsaverage5 cortical surface.

    Produces a 4-panel figure showing the time-averaged cortical activation
    mapped onto the inflated brain surface from 4 viewpoints:
      - Left hemisphere: lateral view  (outer surface, language/temporal areas)
      - Left hemisphere: medial view   (inner surface, default mode network)
      - Right hemisphere: lateral view
      - Right hemisphere: medial view

    The background uses sulcal depth (gyri bright, sulci dark) for anatomical
    orientation. Color scale is symmetric around zero so positive (red) and
    negative (blue) deflections are clearly distinguishable.

    Args:
        preds:      (n_timesteps, 20484) predicted BOLD array
        output_dir: Directory to save the plot
        title_prefix: Optional string prepended to the plot title
    """
    try:
        from nilearn import plotting, datasets
    except ImportError:
        print("nilearn not installed — skipping surface plot. Run: pip install nilearn")
        return

    import matplotlib.image as mpimg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    os.makedirs(output_dir, exist_ok=True)

    # Average over time → (20484,) mean activation per vertex
    mean_act   = preds.mean(axis=0)
    left_data  = mean_act[:10242]   # fsaverage5 left hemisphere vertices
    right_data = mean_act[10242:]   # fsaverage5 right hemisphere vertices

    # Fetch fsaverage5 surface files (cached after first download, ~20 MB)
    print("Fetching fsaverage5 surface mesh (cached after first download)...")
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    # Symmetric color scale so zero is white/neutral
    vmax = float(np.abs(mean_act).max())
    if vmax == 0:
        vmax = 1.0   # avoid degenerate colorbar

    # Four views: (hemi, view, data, surface_mesh, bg_map)
    views = [
        ("left",  "lateral", left_data,  fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("left",  "medial",  left_data,  fsaverage["infl_left"],  fsaverage["sulc_left"]),
        ("right", "lateral", right_data, fsaverage["infl_right"], fsaverage["sulc_right"]),
        ("right", "medial",  right_data, fsaverage["infl_right"], fsaverage["sulc_right"]),
    ]

    panel_titles = [
        "Left Hemisphere — Lateral",
        "Left Hemisphere — Medial",
        "Right Hemisphere — Lateral",
        "Right Hemisphere — Medial",
    ]

    # Render each view to a temporary PNG, then stitch them into one figure.
    # Capture the returned matplotlib Figure and call .savefig() directly —
    # nilearn's output_file parameter is unreliable across versions, and
    # engine='matplotlib' is required to ensure we get a Figure back rather
    # than a plotly object (nilearn v0.10+ supports both engines).
    panel_images = []
    for hemi, view, data, surf_mesh, bg_map in views:
        surf_fig = plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=data,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            bg_on_data=True,
            colorbar=False,
            cmap="RdBu_r",
            vmax=vmax,
            engine="matplotlib",
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        surf_fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
        plt.close(surf_fig)
        panel_images.append(tmp_path)

    # Build the stitched 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, img_path, panel_title in zip(axes.flat, panel_images, panel_titles):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.axis("off")
        os.unlink(img_path)   # clean up temp file

    fig.suptitle(
        f"{title_prefix}TRIBE v2 — Mean Predicted BOLD on Cortical Surface\n"
        f"Inflated fsaverage5 | colormap: RdBu_r (red=positive, blue=negative) | "
        f"vmax={vmax:.3f}",
        fontsize=12, fontweight="bold",
    )

    # Shared horizontal colorbar beneath all panels
    sm   = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.02, pad=0.04, shrink=0.55,
    )
    cbar.set_label("Mean Predicted BOLD (z-score)", fontsize=10)

    surface_path = os.path.join(output_dir, "tribe_brain_surface.png")
    plt.savefig(surface_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Surface plot saved to {surface_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TRIBE v2 on text and plot predicted cortical activity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tribe_inference.py --prompt "The apple fell from the tree."
  python tribe_inference.py --text path/to/text.txt
  python tribe_inference.py --audio path/to/audio.wav
  python tribe_inference.py --prompt "Explain gravity" --output preds.npy
  python tribe_inference.py --prompt "Speech" --region broca
  python tribe_inference.py --list_regions
        """,
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text string to encode directly")
    parser.add_argument("--text",   type=str, default=None,
                        help="Path to a .txt file")
    parser.add_argument("--audio",  type=str, default=None,
                        help="Path to an audio file (skips TTS synthesis)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save raw predictions to this .npy file")
    parser.add_argument("--output_dir", type=str, default="./tribe-output",
                        help="Directory for plots (default: ./tribe-output)")
    parser.add_argument("--cache",  type=str, default="./tribe-cache",
                        help="Cache folder for model weights (default: ./tribe-cache)")
    parser.add_argument("--region", type=str, default=None,
                        help="Print detailed stats for a specific brain region "
                             "(e.g. --region broca). See --list_regions for all options.")
    parser.add_argument("--list_regions", action="store_true",
                        help="List all available named brain regions and exit")
    args = parser.parse_args()

    # --list_regions: print atlas info and exit (no model needed)
    if args.list_regions:
        print("Loading Destrieux atlas to count vertices per region...")
        atlas = load_destrieux_atlas()
        list_available_regions(atlas)
        raise SystemExit(0)

    # Resolve input
    text_path  = None
    audio_path = None
    tmp_file   = None

    if args.audio:
        audio_path = args.audio
    elif args.text:
        text_path = args.text
    elif args.prompt:
        # Write the prompt string to a temp file so TRIBE can read it
        tmp_file  = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp_file.write(args.prompt)
        tmp_file.close()
        text_path = tmp_file.name
        print(f"Prompt written to temp file: {text_path}")
    else:
        # Default demo text — short enough to run quickly
        demo_text = (
            "The scientist carefully examined the brain scans. "
            "She noticed unusual patterns of activation in the temporal lobe. "
            "Language areas showed strong bilateral engagement during the task."
        )
        tmp_file  = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp_file.write(demo_text)
        tmp_file.close()
        text_path = tmp_file.name
        print(f"Using demo text: {demo_text!r}\n")

    title_prefix = ""
    if args.prompt:
        title_prefix = f'"{args.prompt[:60]}{"..." if len(args.prompt) > 60 else ""}"\n'

    # Load atlas (used by plot_results and optional --region stats)
    print("Loading Destrieux brain atlas...")
    atlas = load_destrieux_atlas()

    # Load TRIBE model and run
    model = load_model(cache_folder=args.cache)

    if audio_path:
        preds, segments = run_on_audio(model, audio_path)
    else:
        preds, segments = run_on_text(model, text_path)

    # Clean up temp file
    if tmp_file is not None:
        os.unlink(tmp_file.name)

    # Save predictions
    if args.output:
        np.save(args.output, preds)
        print(f"Predictions saved to {args.output}  shape={preds.shape}")

    # Optional: print detailed stats for a specific region
    if args.region:
        region_key = args.region.lower()
        try:
            ts   = extract_region_activity(preds, region_key, atlas)
            desc = FRIENDLY_ROIS[region_key]["description"]
            verts = _resolve_vertices(region_key, atlas)
            print(f"\n--- Region: {desc} ---")
            print(f"  Vertices   : {len(verts)}")
            print(f"  Mean BOLD  : {ts.mean():+.4f}")
            print(f"  Peak BOLD  : {ts.max():+.4f}  (at t={ts.argmax()}s)")
            print(f"  Timeseries : {np.array2string(ts, precision=3, floatmode='fixed')}")
        except ValueError as e:
            print(f"ERROR: {e}")

    # Plot time-series / heatmap panels (uses atlas for accurate ROI labels)
    plot_results(preds, output_dir=args.output_dir, title_prefix=title_prefix, atlas=atlas)

    # Plot mean BOLD on the 3-D cortical surface
    plot_brain_surface(preds, output_dir=args.output_dir, title_prefix=title_prefix)
