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
# Analysis utilities
# ---------------------------------------------------------------------------

# fsaverage5 vertex index ranges for approximate ROIs.
# These are rough but usable — for precise ROIs use nilearn's atlas-based masking.
#
# fsaverage5 has 10242 vertices per hemisphere, 20484 total.
# Left hemisphere: indices 0–10241
# Right hemisphere: indices 10242–20483
#
# These approximate ranges come from the known topology of the fsaverage5 mesh.
APPROXIMATE_ROIS = {
    "Left Temporal (language/auditory)": (1800, 2500),
    "Right Temporal (auditory)":         (12042, 12742),
    "Left Frontal (Broca)":              (500,  900),
    "Left Parietal (integration)":       (3000, 3500),
    "Left Occipital (visual)":           (4000, 4600),
    "Right Occipital (visual)":          (14242, 14842),
}


def roi_timeseries(preds: np.ndarray) -> dict:
    """
    Extract mean BOLD timeseries for approximate cortical ROIs.

    Args:
        preds: (n_timesteps, 20484) predicted BOLD array

    Returns:
        dict mapping ROI name → mean timeseries array of shape (n_timesteps,)
    """
    return {
        name: preds[:, start:end].mean(axis=1)
        for name, (start, end) in APPROXIMATE_ROIS.items()
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

def plot_results(preds: np.ndarray, output_dir: str, title_prefix: str = ""):
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
    roi_ts    = roi_timeseries(preds)
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

    # --- Panel 2: ROI timeseries ---
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, len(roi_ts)))
    for (roi_name, ts), color in zip(roi_ts.items(), colors):
        ax2.plot(time_axis, ts, label=roi_name, color=color, linewidth=1.2)
    ax2.set_title("ROI Timeseries (approximate)", fontweight="bold")
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
    args = parser.parse_args()

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

    # Load and run
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

    # Plot
    plot_results(preds, output_dir=args.output_dir, title_prefix=title_prefix)
