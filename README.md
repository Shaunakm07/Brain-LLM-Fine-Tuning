# Brain — LLM Fine-Tuning

Fine-tune small language models on a laptop or GPU using LoRA, with a live LLM judge as the reward signal — or go further and use **TRIBE v2**, a Meta AI neuroscience model, to optimise text for specific cortical brain region activation.

---

## What this repo does

| Capability | Script |
|------------|--------|
| Run Qwen 0.5B locally (interactive) | `inference.py` |
| Fine-tune Qwen with a frozen LLM judge as reward | `train.py` |
| Compare base vs fine-tuned model side by side | `compare.py` |
| Predict fMRI brain activity from text (TRIBE v2) | `tribe_inference.py` |
| Optimise Qwen to maximise/minimise a brain region | `brain_optimize.py` |
| Same as above, tuned for NVIDIA L40S GPU | `brain_optimize_l40s.py` |

The core technique is **Advantage-Weighted SFT with a KL penalty** — the same mechanism as RLHF/PPO but simpler and stable enough to run on a laptop CPU:

```
loss = advantage × CE(policy, completion) + kl_coef × KL(policy ∥ base)
```

- **Advantage** = `(reward − mean) / std` across completions — makes the gradient independent of the absolute scale of the reward signal
- **KL penalty** — prevents the policy from drifting so far from the base that outputs become incoherent

---

## Installation

### Core dependencies (required for all scripts)

```bash
pip install torch transformers peft matplotlib scipy
```

### TRIBE v2 (required only for `tribe_inference.py` and `brain_optimize*.py`)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
pip install nilearn

# TRIBE's text pathway uses LLaMA 3.2-3B — you must accept the license and log in
# 1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B
# 2. Run:
huggingface-cli login
```

> **TRIBE license:** CC-BY-NC-4.0 (non-commercial research use only)

---

## Hardware requirements

| Script | Minimum | Recommended |
|--------|---------|-------------|
| `inference.py` | 2 GB RAM, any CPU | — |
| `train.py` | 6 GB RAM, any CPU | Apple Silicon / NVIDIA GPU |
| `compare.py` | 6 GB RAM | — |
| `tribe_inference.py` | 10 GB RAM, CPU OK | GPU (A100/L40S) |
| `brain_optimize.py` | 12 GB RAM, CPU OK | GPU — ~4 min/step on CPU |
| `brain_optimize_l40s.py` | NVIDIA L40S (46 GB VRAM) | — |

**On CPU:** each TRIBE call (TTS → WhisperX → LLaMA 3.2-3B → BOLD) takes 1–3 minutes. A 10-step run with 4 completions takes 40–120 minutes.  
**On GPU (A100/L40S):** each TRIBE call takes ~5–15 seconds. 200 steps in a few hours.

---

## Quick start

```bash
# Interactive inference
python inference.py

# Fine-tune with a frozen LLM judge
python train.py --criteria "responses should be concise and use simple language"

# Compare base model vs fine-tuned side by side
python compare.py --adapter ./lora-adapter --prompts "Explain gravity" "What is DNA?"

# Predict brain activity from text (fast test — no GPU needed)
python tribe_inference.py --prompt "The scientist examined the brain scans."

# Optimise Qwen for Broca's area — mock rewards (fast test, ~2 min)
python brain_optimize.py --mock_tribe --n_steps 10
```

---

## Scripts

### `inference.py` — interactive chat

Loads Qwen2.5-0.5B-Instruct and runs an interactive REPL. Type a prompt, get a response.

```bash
python inference.py
```

No arguments. Edit `MODEL_ID` in the file to change the model.

---

### `train.py` — LoRA fine-tuning with LLM judge

Fine-tunes Qwen 0.5B (LoRA) using Qwen 1.5B as a frozen judge. You provide a plain-English criteria string; the judge scores completions 1–10 against it.

```bash
# Quickstart — uses 10 built-in demo prompts
python train.py --criteria "responses should be concise and use simple language"

# Custom prompts
python train.py --criteria "formal and professional tone" \
                --prompts "Explain gravity" "What is DNA?"

# Prompts from file (one per line)
python train.py --criteria "detailed with examples" --prompts_file prompts.txt
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--criteria` | required | What makes a good response (plain English) |
| `--prompts` | — | Prompts on the command line |
| `--prompts_file` | — | Path to a `.txt` file, one prompt per line |
| `--epochs` | `5` | Passes over the prompt dataset |
| `--lr` | `2e-4` | AdamW learning rate |
| `--lora_r` | `8` | LoRA rank (higher = more expressive, more compute) |
| `--completions` | `4` | Completions per prompt per epoch — more = lower variance |
| `--output_dir` | `./lora-adapter` | Where to save adapter, metrics, and plot |

**Outputs** (`./lora-adapter/`):
- `adapter_model.safetensors` — LoRA weights (~6 MB)
- `metrics.json` — per-step reward, advantage, loss, KL
- `training_metrics.png` — 6-panel training plot

**Choosing effective criteria:**  
If the judge scores all completions similarly, advantages are zero and steps are skipped. Good criteria describe something the base model clearly struggles with:

| Criteria | Skip rate |
|----------|-----------|
| `"be helpful and concise"` | Very high — base model already does this |
| `"respond only using an analogy to cooking"` | Low — model rarely does this |
| `"answer in exactly 3 numbered steps, no more"` | Low — model frequently violates this |

See [`docs/reward-maximization.md`](docs/reward-maximization.md) for a full guide.

---

### `compare.py` — side-by-side evaluation

Loads the base model and your fine-tuned adapter, then generates responses to prompts with both and prints them side by side. Optionally scores both with the judge.

```bash
# Single prompt
python compare.py --adapter ./lora-adapter --prompt "Explain gravity"

# Multiple prompts, scored by judge
python compare.py --adapter ./lora-adapter \
                  --prompts "Explain gravity" "What is DNA?" \
                  --criteria "concise and simple" --judge

# Save results to JSON
python compare.py --adapter ./lora-adapter --prompts_file prompts.txt \
                  --criteria "formal tone" --judge --output results.json
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--adapter` | required | Path to saved LoRA adapter directory |
| `--prompt` | — | Single prompt |
| `--prompts` | — | Multiple prompts on command line |
| `--prompts_file` | — | `.txt` file, one prompt per line |
| `--criteria` | — | Criteria string (required with `--judge`) |
| `--judge` | off | Score both responses with Qwen 1.5B judge |
| `--output` | — | Save all results to this JSON file |

---

### `tribe_inference.py` — brain activity prediction

Runs TRIBE v2 on text and produces predicted fMRI BOLD activity across the full cortical surface. Saves time-series plots and an inflated brain surface map.

```bash
# Quick demo (default text)
python tribe_inference.py

# Your own text
python tribe_inference.py --prompt "The apple fell from the tree."

# From a text file
python tribe_inference.py --text path/to/text.txt

# From a pre-existing audio file (skips TTS)
python tribe_inference.py --audio path/to/audio.wav

# Show detailed stats for a specific brain region
python tribe_inference.py --prompt "She spoke in a quiet voice." --region broca

# Save raw predictions for downstream use
python tribe_inference.py --prompt "Explain gravity" --output preds.npy

# List all 20 available brain regions
python tribe_inference.py --list_regions
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | demo text | Text string to encode |
| `--text` | — | Path to a `.txt` file |
| `--audio` | — | Path to an audio file (skips TTS) |
| `--output` | — | Save raw `(n_timesteps, 20484)` array as `.npy` |
| `--output_dir` | `./tribe-output` | Directory for plots |
| `--cache` | `./tribe-cache` | Where to cache TRIBE model weights (~710 MB) |
| `--region` | — | Print stats for a named region (e.g. `broca`) |
| `--list_regions` | — | Print all available regions and exit |

**Outputs** (`./tribe-output/`):
- `tribe_predictions.png` — 4-panel plot (global BOLD, ROI timeseries, vertex heatmap, hemisphere comparison)
- `tribe_brain_surface.png` — inflated cortical surface from 4 viewpoints

**Available brain regions** (use with `--region`):

| Key | Description |
|-----|-------------|
| `broca` | Broca's area (IFG pars opercularis + triangularis, left) |
| `wernicke` | Wernicke's area (posterior superior temporal, left) |
| `auditory` | Primary auditory cortex (Heschl's gyrus, bilateral) |
| `sts` | Superior temporal sulcus / planum temporale (bilateral) |
| `angular` | Angular gyrus (left inferior parietal) |
| `supramarginal` | Supramarginal gyrus (left inferior parietal) |
| `default_mode` | Default mode network (posterior cingulate + precuneus) |
| `anterior_cingulate` | Anterior cingulate cortex |
| `dlpfc` | Dorsolateral prefrontal cortex |
| `superior_frontal` | Superior frontal gyrus |
| `v1` | Primary visual cortex (calcarine sulcus) |
| `lateral_occipital` | Lateral occipital cortex |
| `fusiform` | Fusiform gyrus |
| `parahippocampal` | Parahippocampal gyrus |
| `insula` | Insula |
| `middle_temporal` | Middle temporal gyrus |
| `inferior_temporal` | Inferior temporal gyrus |
| `superior_parietal` | Superior parietal lobule |
| `motor` | Primary motor cortex (precentral gyrus) |
| `somatosensory` | Primary somatosensory cortex (postcentral gyrus) |

See [`docs/tribe-model.md`](docs/tribe-model.md) for the full TRIBE v2 architecture and text processing pipeline.

---

### `brain_optimize.py` — brain-guided LLM optimisation

Fine-tunes Qwen 0.5B (LoRA) so it generates text that maximally (or minimally) activates a target cortical region, as measured by TRIBE v2.

```bash
# Fast test — mock rewards, no TRIBE needed (~2 min)
python brain_optimize.py --mock_tribe --n_steps 10

# Maximise Broca's area (real TRIBE — slow on CPU)
python brain_optimize.py --region broca --n_steps 10 --n_completions 4

# Minimise primary visual cortex
python brain_optimize.py --region v1 --minimize --n_steps 10

# Custom prompt
python brain_optimize.py \
    --prompt "Describe a vivid scene from nature." \
    --region auditory --n_steps 20 --n_completions 8

# List all available regions
python brain_optimize.py --list_regions
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `"Describe something interesting about language and the brain."` | Policy model prompt |
| `--region` | `broca` | Target brain region |
| `--minimize` | off | Minimise activation instead of maximising |
| `--n_steps` | `10` | Optimisation steps |
| `--n_completions` | `4` | Completions per step — more = lower variance gradient |
| `--max_new_tokens` | `80` | Max completion length in tokens |
| `--temperature` | `0.85` | Sampling temperature (0 = greedy) |
| `--kl_coef` | `0.5` | KL penalty weight — increase if outputs become incoherent |
| `--lr` | `5e-5` | AdamW learning rate |
| `--mock_tribe` | off | Use fast mock rewards (for testing only) |
| `--output_dir` | `./brain-optimize-output` | Where to save outputs |
| `--cache` | `./tribe-cache` | TRIBE model cache directory |

**Outputs** (`./brain-optimize-output/`):
- `best_completion.txt` — highest-reward completion found during training
- `training_curves.png` — 4-panel plot (mean reward, best reward, loss, KL divergence)
- `brain_surface_best.png` — cortical surface map for the best completion (real TRIBE only)
- `optimizer_metrics.json` — all per-step metrics
- `lora-brain-adapter/` — saved LoRA adapter weights

---

### `brain_optimize_l40s.py` — L40S-optimised variant

Extended version of `brain_optimize.py` with additional features for long GPU runs:

| Feature | `brain_optimize.py` | `brain_optimize_l40s.py` |
|---------|-------------------|--------------------------|
| LoRA rank | r=8, q+v only | r=16, all attention + FFN |
| Default completions | 4 | 8 |
| Max tokens | 80 | 120 |
| Temperature | fixed | anneals 1.0 → 0.6 |
| LR schedule | none | warmup + cosine |
| Gradient clip | 1.0 | 0.5 |
| Checkpointing | no | every 10 steps |
| Resume | no | `--resume` |
| Infinite mode | no | `--n_steps 0` |
| Step timing / ETA | no | yes |
| GPU memory log | no | yes |

```bash
# Fast test
python brain_optimize_l40s.py --mock_tribe --n_steps 5

# Full run — submit via SLURM using run_brain_optimize_l40s.sh
python brain_optimize_l40s.py --region broca

# Resume from latest checkpoint after an interruption
python brain_optimize_l40s.py --region broca --resume

# Run indefinitely (Ctrl-C finishes current step and saves cleanly)
python brain_optimize_l40s.py --region broca --n_steps 0
```

**Additional arguments** (beyond those in `brain_optimize.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--t_start` | `1.0` | Initial sampling temperature |
| `--t_end` | `0.6` | Final temperature after linear annealing |
| `--warmup_steps` | `5` | Steps for linear LR warmup before cosine decay |
| `--resume` | off | Resume from latest checkpoint in `--output_dir` |
| `--n_steps` | `200` | Steps (`0` = run until Ctrl-C) |

**Submitting to SLURM:**

```bash
sbatch run_brain_optimize_l40s.sh
```

The script requests 1 GPU, 64 GB RAM, 8 CPUs, and a 24-hour time limit. Edit the `#SBATCH` headers to match your cluster. Logs are written to `brain_optimize_l40s_<job_id>.log`.

---

## Understanding training output

### The metric that matters: reward, not loss

The training loss in this algorithm **is not expected to decrease monotonically**. It can be negative (suppression steps) and will drift upward as the KL penalty grows. This is normal.

**Watch the reward** — specifically Panel 1 (mean reward) and Panel 2 (best reward found so far) in `training_curves.png`.

### Reading `training_curves.png`

| Panel | What it shows | Healthy sign |
|-------|---------------|--------------|
| Mean reward per step | Average TRIBE activation across completions | Trending up over training |
| Best reward found so far | Best completion found at any point | Rising curve that plateaus |
| Loss per step | Advantage × CE + KL penalty | Fluctuates; not monotone |
| KL divergence | Policy drift from base model | Stays well below 1.0 |

**If KL rises above 2.0:** increase `--kl_coef` (try `1.0`).  
**If reward plateaus early:** try more completions (`--n_completions 16`) or more steps.  
**If reward never improves:** check that `--mock_tribe` is not accidentally set when you want real rewards.

### Reducing training variance

The wide ±1 std band in the reward plot indicates noisy gradients. To reduce it:

1. **More completions per step** — `--n_completions 16` or `32` (most direct fix; linear cost)
2. **Lower temperature** — `--temperature 0.6` (less diversity, but also less noise)
3. **Advantage std-normalisation** — already applied in this repo; advantages are divided by the reward std per step so gradient scale is independent of reward spread

---

## Hyperparameter quick reference

### `train.py`

| Symptom | Fix |
|---------|-----|
| `[Skip]` on > 30% of steps | Criteria too easy — make it harder or more specific |
| Reward flat after 3+ epochs | Increase `--completions`, more diverse prompts |
| Loss NaN | Reduce `--lr` by 10× |
| KL > 2–3 in early training | Increase `kl_coef` in `advantage_weighted_loss()` (default 0.5 → 1.0) |
| Reward improves then collapses | Lower `--lr`, increase `kl_coef` |

### `brain_optimize.py` / `brain_optimize_l40s.py`

| Symptom | Fix |
|---------|-----|
| High variance in mean reward | Increase `--n_completions`, lower `--temperature` |
| Reward plateaus after a few steps | More steps, increase LoRA rank (`r=32`) |
| KL > 2.0 | Increase `--kl_coef` to `1.0` |
| OOM on GPU | Reduce `--n_completions` or `--max_new_tokens` |
| Very slow on CPU | Use `--mock_tribe` to test; run real TRIBE on a GPU node |

---

## Loading a trained adapter

After any training run, the LoRA adapter is saved to the output directory. To use it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float32,
)
model = PeftModel.from_pretrained(base, "./lora-adapter")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")
inputs  = tokenizer("Explain gravity:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

To merge the adapter permanently into the base weights (removes the PEFT dependency):

```python
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

---

## Supported models

Any causal LM on HuggingFace works. Recommended for local use:

| Model | Params | Notes |
|-------|--------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Fastest; minimal RAM |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Good quality/speed balance |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Best local quality |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Strong reasoning |
| `meta-llama/Llama-3.2-1B` | 1B | Good general baseline |

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/reward-maximization.md`](docs/reward-maximization.md) | Full guide to reward-maximisation training: advantage vs raw reward, KL penalty, criteria selection, failure modes, hyperparameter guide |
| [`docs/tribe-model.md`](docs/tribe-model.md) | TRIBE v2 architecture, text processing pipeline (TTS → WhisperX → LLaMA → BOLD), output format, training details |
| [`docs/lora-configuration.md`](docs/lora-configuration.md) | LoRA rank selection, QLoRA (4-bit), saving and loading adapters |
| [`docs/optimization-functions.md`](docs/optimization-functions.md) | Loss functions (CE, focal, DPO, contrastive, reward-weighted) and optimizers (AdamW, Lion, Adafactor, GaLore) |
| [`docs/training-pipeline.md`](docs/training-pipeline.md) | End-to-end example using `SFTTrainer` with a custom loss |

---

## Troubleshooting

**`torchvision ABI mismatch` on import**  
A version mismatch between PyTorch and torchvision is patched automatically at runtime. To fix permanently:
```bash
pip install "torchvision==0.26.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

**`Requested float16 compute type` (WhisperX crash on CPU/Mac)**  
The subprocess patch in `tribe_inference.py` and `brain_optimize*.py` automatically forces `--compute_type int8` for CPU. If it still fails, ensure you imported from the patched file (the patch runs at module level before tribev2 is imported).

**`BloomPreTrainedModel` import error (peft + transformers mismatch)**  
Injected automatically — the scripts stub the missing class before importing peft. Permanent fix:
```bash
pip install "peft>=0.15.0" "transformers>=4.57.0"
```

**TRIBE model not found / download fails**  
```bash
huggingface-cli login
# Then retry — first download is ~710 MB
```

**`[Warning] Could not parse score` during training**  
The judge returned a non-numeric token. Usually transient. If persistent, the judge model may not be following instructions well — try a larger judge or simplify the criteria.

**All advantages are 0.0 / frequent `[Skip]`**  
The judge scored all completions identically. The criteria is too easy for the base model to already satisfy. See the [criteria selection guide](docs/reward-maximization.md#choosing-effective-criteria).
