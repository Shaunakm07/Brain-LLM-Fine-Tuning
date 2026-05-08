# Brain — LLM Fine-Tuning

A toolkit for fine-tuning small LLMs locally using LoRA, with support for custom loss functions, optimizers, reward-maximisation objectives, and **brain-guided optimization via TRIBE v2**.

---

## What this repo does

- Fine-tunes small LLMs (sub-3B parameters) on a laptop or single GPU using **LoRA** adapters
- Provides a plug-and-play system for swapping **loss functions** and **optimizers** independently of the model architecture
- Supports **reward-maximisation training** — fine-tuning a model to maximise the output of a separate reward model or LLM judge
- Includes **TRIBE v2 integration** — predict fMRI brain activity from text, extract per-region signals using the Destrieux atlas, and visualize on a 3D cortical surface
- Supports **brain-guided LLM optimization** — fine-tune Qwen to generate text that maximally (or minimally) activates a target cortical region

---

## Quickstart

```bash
pip install torch transformers peft matplotlib nilearn scipy

# Run local inference
python inference.py

# Fine-tune with a local LLM judge
python train.py --criteria "responses should be concise and use simple language"

# Compare base model vs fine-tuned model side by side
python compare.py --adapter ./lora-adapter --prompts "Explain gravity" "What is DNA?"

# Run TRIBE v2 — predict brain activity from text + surface plot
pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
brew install ffmpeg
huggingface-cli login   # LLaMA 3.2-3B access required
python tribe_inference.py --prompt "The scientist examined the brain scans."

# Show activity in a specific brain region (Destrieux atlas)
python tribe_inference.py --prompt "She spoke in a quiet voice." --region broca

# List all 20 available brain regions
python tribe_inference.py --list_regions

# Optimize Qwen to maximally activate Broca's area (test mode, fast)
python brain_optimize.py --mock_tribe --region broca --n_steps 10

# Real brain-guided optimization (slow on CPU — ~1h for 10 steps)
python brain_optimize.py --region broca --n_steps 10 --n_completions 4
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `inference.py` | Run `Qwen2.5-0.5B-Instruct` locally |
| `train.py` | Fine-tune policy model using a frozen LLM judge as reward signal |
| `compare.py` | Compare base model vs fine-tuned model side by side |
| `tribe_inference.py` | Run TRIBE v2, extract brain region activity, plot cortical surface |
| `brain_optimize.py` | Optimize Qwen using TRIBE brain region activation as reward |

`train.py` uses advantage-weighted cross-entropy with a KL penalty. See [`docs/reward-maximization.md`](docs/reward-maximization.md).

`tribe_inference.py` runs the full TRIBE v2 pipeline (text → TTS → WhisperX → LLaMA features → BOLD predictions), extracts region-specific activity using the Destrieux atlas (20 named regions), and produces a 3D inflated cortical surface plot.

`brain_optimize.py` fine-tunes Qwen with TRIBE as the reward model instead of an LLM judge. The reward is mean predicted BOLD in a target cortical region. Use `--mock_tribe` to test the optimizer without running TRIBE.

---

## Brain Region Optimization

```bash
# List all 20 available target regions
python brain_optimize.py --list_regions

# Maximize Broca's area (left IFG — speech production)
python brain_optimize.py --region broca --n_steps 10

# Maximize auditory cortex (Heschl's gyrus)
python brain_optimize.py --region auditory --n_steps 10

# Minimize visual cortex (V1)
python brain_optimize.py --region v1 --minimize --n_steps 10

# Mock mode — verifies the optimizer runs without TRIBE (~2 min)
python brain_optimize.py --mock_tribe --n_steps 10 --n_completions 4
```

Available regions include: `broca`, `wernicke`, `auditory`, `sts`, `angular`,
`supramarginal`, `default_mode`, `anterior_cingulate`, `dlpfc`, `superior_frontal`,
`v1`, `lateral_occipital`, `fusiform`, `parahippocampal`, `insula`,
`middle_temporal`, `inferior_temporal`, `superior_parietal`, `motor`, `somatosensory`.

Outputs saved to `./brain-optimize-output/`:
- `best_completion.txt` — highest-reward generated text found during training
- `training_curves.png` — reward, loss, and KL divergence per step
- `brain_surface_best.png` — cortical surface map of best completion (real TRIBE only)
- `lora-brain-adapter/` — saved LoRA adapter weights

---

## Documentation

| Document | Description |
|----------|-------------|
| [LoRA Configuration](docs/lora-configuration.md) | How to configure LoRA adapters, rank selection, and QLoRA setup |
| [Optimization Functions](docs/optimization-functions.md) | Loss functions (cross-entropy, focal, DPO, contrastive, reward-weighted) and optimizers (AdamW, Lion, Adafactor, GaLore) |
| [Training Pipeline](docs/training-pipeline.md) | End-to-end fine-tuning example with a custom trainer |
| [Reward Maximisation](docs/reward-maximization.md) | Fine-tuning to maximise the output of another model (GRPO, PPO, DPO, reward-weighted SFT) |
| [TRIBE v2 Model](docs/tribe-model.md) | How TRIBE v2 predicts fMRI brain activity from text — architecture, text pipeline, output format |

---

## Supported Models

Any causal LM on HuggingFace works. Recommended models for local use:

| Model | Params | Notes |
|-------|--------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Fastest, minimal RAM |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Good quality/speed balance |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Best local quality |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Strong reasoning |
| `meta-llama/Llama-3.2-1B` | 1B | Good general baseline |
