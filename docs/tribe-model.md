# TRIBE v2 — How It Works on Text

**Paper**: _A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience_ (d'Ascoli et al., 2026)
**Model**: `facebook/tribev2` on HuggingFace
**Code**: https://github.com/facebookresearch/tribev2
**License**: CC-BY-NC-4.0 (non-commercial use only)

---

## What TRIBE v2 Does

TRIBE v2 (**Tri-modal Brain Encoding**) is a foundation model for **in-silico neuroscience**. Given a naturalistic stimulus — video, audio, or text — it predicts the fMRI BOLD response that a human brain would produce while perceiving it.

It is not a language model, classifier, or generative model. It is a **regression model** that outputs predicted cortical activity as a function of time.

```
Input stimulus (text / audio / video)
              │
              ▼
    TRIBE v2 (709 MB checkpoint)
              │
              ▼
  Predicted fMRI BOLD signal
  shape: (n_timesteps, 20484 vertices)
  1 timestep = 1 second of stimulus
  20484 = full cortical surface (fsaverage5 mesh)
```

---

## Text Input Pipeline — Step by Step

Text does not go directly into TRIBE. The model was trained on naturalistic audiovisual stimuli, so text is first converted into a form the model can process. The full pipeline for a `.txt` input is:

```
Your text.txt
     │
     ▼  [1] langdetect — detect language
     │
     ▼  [2] gTTS (Google Text-to-Speech) — synthesise text to audio (MP3)
     │
     ▼  [3] WhisperX — word-level transcription with timestamps
     │           → structured events DataFrame:
     │             columns: type, start, duration, text, context
     │
     ├──── Text / Word events ──────────► [4a] LLaMA 3.2-3B (text pathway)
     │                                         6 layer fractions: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
     │                                         Features aggregated at 2 Hz
     │
     └──── Audio events ────────────────► [4b] Wav2Vec-BERT 2.0 (audio pathway)
                                               3 layer fractions: [0.5, 0.75, 1.0]
                                               Processes the synthesised speech waveform
              │                    │
              └────────┬───────────┘
                       ▼
           [5] Multimodal Fusion Transformer
               - Projects all features to 1152-dim shared space
               - 8 layers, 8 heads, rotary positional embeddings
               - Subject embedding added (average subject at inference)
                       │
                       ▼
           [6] Subject-specific output head
               - Low-rank bottleneck (rank 2048)
               - Linear projection to 20484 cortical vertices
                       │
                       ▼
           Predicted BOLD: (n_timesteps, 20484)
```

**Why TTS?** Text alone has no temporal structure. TRIBE operates at 1 TR/second and needs word-level timestamps. By synthesising speech first, it recovers when each word occurs and can align LLaMA features to real time.

**Why two pathways?** When you read text aloud (or hear it), your brain processes both the semantic content (via language areas) and the acoustic properties of the speech (via auditory cortex). TRIBE models both pathways simultaneously for a complete prediction.

---

## Architecture Details

### Stage 1 — Feature Extractors (all frozen)

| Modality | Model | Layers extracted |
|----------|-------|-----------------|
| Text / Language | `meta-llama/Llama-3.2-3B` | 6 fractions: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` of total depth |
| Audio | `facebook/w2v-bert-2.0` | 3 fractions: `[0.5, 0.75, 1.0]` |
| Video | `facebook/vjepa2-vitg-fpc64-256` | 3 fractions: `[0.5, 0.75, 1.0]` |
| Static frames | `facebook/dinov2-large` | ~`0.67` fraction |

Multiple layer fractions are extracted because different depths encode different levels of abstraction. Early LLaMA layers capture syntax and surface form; later layers capture semantics and pragmatics. The brain uses all of these.

### Stage 2 — Multimodal Fusion Transformer

- All modality features projected to **1152 hidden dimensions**
- Concatenated across the time axis, up to **1024 max tokens**
- **8 transformer layers**, **8 attention heads**
- Feed-forward size: 4608 (4× expansion)
- Positional encoding: **Rotary (RoPE)** — better for variable-length sequences than learned positional embeddings
- Normalisation: **ScaleNorm** (not LayerNorm) — more stable for multimodal fusion
- Subject embedding added before transformer — allows subject-specific predictions

### Stage 3 — Output Head

- Low-rank bottleneck projection (rank **2048**) compresses the fused representation
- Subject-specific linear layer maps to **20484 cortical vertices** (fsaverage5 surface mesh)
- At inference, the "average subject" is used — gives a subject-agnostic cortical prediction

---

## Output Format

```python
preds, segments = model.predict(events=df)

# preds: numpy array, shape (n_timesteps, 20484)
#   - n_timesteps: number of 1-second windows (= stimulus duration in seconds - ~5s HRF delay)
#   - 20484: vertices of the fsaverage5 cortical mesh (both hemispheres)
#   - values: predicted z-scored BOLD signal (normalised per vertex during training)

# segments: list of aligned segment objects for temporal reference
```

The **20484 vertices** cover the entire cortical surface. You can use `nilearn` or `nibabel` to map these onto an anatomical brain image or select specific ROIs (e.g. Broca's area, auditory cortex, visual cortex).

**Hemodynamic delay**: fMRI responses lag ~5 seconds behind the stimulus. TRIBE compensates for this internally — the output is aligned to stimulus time, not scanner time.

---

## What the Output Tells You

The predicted BOLD signal shows which cortical regions are active at each moment in time while processing the input text:

| Region (approximate vertices) | What activation means for text input |
|-------------------------------|--------------------------------------|
| Temporal cortex (superior temporal gyrus) | Auditory speech processing — driven by synthesised speech |
| Left frontal operculum / Broca's area | Syntactic and semantic language processing |
| Posterior temporal / angular gyrus | Semantic meaning integration |
| Default mode network | Narrative comprehension, world knowledge retrieval |
| Primary auditory cortex | Low-level acoustic features of synthesised speech |

For text-only input, you generally see strong activation in **language areas** (left hemisphere dominant) and **auditory processing areas** (from the synthesised speech pathway).

---

## Training Details

| Setting | Value |
|---------|-------|
| Loss | MSE (regression against real fMRI z-scores) |
| Optimizer | Adam, lr=1e-4 |
| Scheduler | OneCycleLR, 10% warmup |
| Epochs | 15, batch size 8 |
| Training subjects | 25 across 4 fMRI datasets |
| Training data | >1,000 hours of fMRI (Algonauts2025, Lahner2024, Lebel2023, Wen2017) |
| Modality dropout | 0.3 (random modality masking during training for robustness) |
| Subject dropout | 0.1 |

---

## Requirements and Caveats

**Installation:**
```bash
pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

**LLaMA access required:** The text pathway uses `meta-llama/Llama-3.2-3B`, which is a gated model. You need to:
1. Accept the LLaMA license at https://huggingface.co/meta-llama/Llama-3.2-3B
2. Run `huggingface-cli login` with a token that has access

**License:** CC-BY-NC-4.0 — non-commercial research use only.

**No `transformers` API:** This model does not use `AutoModel.from_pretrained`. You must use the `tribev2` package.

**Text goes through TTS:** An internet connection is required on first run for gTTS synthesis (or provide a `.wav` file directly to skip TTS).

---

## Running the Model on Text

See `tribe_inference.py` in this repo for a complete script that:
- Loads TRIBE v2 from HuggingFace
- Runs it on text input
- Produces plots of the predicted cortical activity over time

```bash
# Run on a text file
python tribe_inference.py --text path/to/text.txt

# Run on a string directly
python tribe_inference.py --prompt "The apple fell from the tree and landed on the ground."

# Save output embeddings
python tribe_inference.py --prompt "Explain gravity" --output preds.npy
```
