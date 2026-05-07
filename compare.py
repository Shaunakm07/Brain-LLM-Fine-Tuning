"""
compare.py — Side-by-side comparison of base model vs LoRA fine-tuned model
============================================================================

WHAT THIS SCRIPT DOES
---------------------
Loads two versions of the same model:
  - Base model  : the original Qwen2.5-0.5B-Instruct, unmodified
  - Tuned model : the same base model with your LoRA adapter applied

For each prompt you provide, both models generate a response. The responses
are printed side by side so you can see what changed after fine-tuning.

Optionally, a frozen judge model scores both responses against the same
criteria you trained on, giving a quantitative measure of improvement.

USAGE
-----
  # Compare on a single prompt
  python compare.py --adapter ./lora-adapter --prompt "Explain gravity"

  # Compare on multiple prompts
  python compare.py --adapter ./lora-adapter \
                    --prompts "Explain gravity" "What is DNA?" "How do computers work?"

  # Compare with prompts from a file (one per line)
  python compare.py --adapter ./lora-adapter --prompts_file prompts.txt

  # Also score both responses with the judge model
  python compare.py --adapter ./lora-adapter \
                    --prompts "Explain gravity" \
                    --criteria "concise and uses simple language" \
                    --judge

  # Save results to a JSON file
  python compare.py --adapter ./lora-adapter --prompts_file prompts.txt \
                    --criteria "formal tone" --judge --output results.json
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float32
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_id: str = BASE_MODEL_ID):
    """Load the unmodified base model."""
    print(f"Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_tuned_model(adapter_path: str, model_id: str = BASE_MODEL_ID):
    """
    Load the base model and apply the LoRA adapter on top.

    The adapter only adds a few MB of weights — the base model weights are
    unchanged. PeftModel merges the adapter matrices with the base weights
    during the forward pass.
    """
    print(f"Loading tuned model: {model_id} + adapter from {adapter_path}...")
    tokenizer  = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE)
    model      = PeftModel.from_pretrained(base_model, adapter_path).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_judge_model(model_id: str = JUDGE_MODEL_ID):
    """Load the frozen judge model for scoring."""
    print(f"Loading judge model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation and scoring
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response from a model for a given prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic for fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score(judge_model, judge_tokenizer, criteria: str, prompt: str, response: str) -> float:
    """Ask the judge model to score a response 1–10 against the criteria."""
    judge_prompt = (
        f"You are an impartial judge evaluating AI responses.\n"
        f"Criteria: {criteria}\n\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n\n"
        f"Score this response from 1 (worst) to 10 (best) based strictly on "
        f"the criteria above. Reply with only a single integer. Score:"
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator. Reply with only a single integer from 1 to 10."},
        {"role": "user",   "content": judge_prompt},
    ]
    text   = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw        = judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for token in raw.split():
        try:
            s = int("".join(c for c in token if c.isdigit()))
            return max(1, min(10, s))
        except ValueError:
            continue
    return 5  # fallback


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    prompts:      list[str],
    adapter_path: str,
    criteria:     str  = None,
    use_judge:    bool = False,
    output_path:  str  = None,
):
    """
    Run the side-by-side comparison for all prompts.

    For each prompt:
      1. Generate response from the base model (greedy, deterministic)
      2. Generate response from the fine-tuned model (greedy, deterministic)
      3. Optionally score both with the judge

    Greedy decoding is used for both models so the comparison is fair —
    sampling would introduce randomness that could mask the effect of training.

    Args:
        prompts:      List of prompts to compare on
        adapter_path: Path to the saved LoRA adapter directory
        criteria:     Criteria to score against (required if use_judge=True)
        use_judge:    Whether to score both responses with the judge model
        output_path:  If set, save all results to this JSON file
    """
    base_model,  base_tokenizer  = load_base_model()
    tuned_model, tuned_tokenizer = load_tuned_model(adapter_path)

    judge_model, judge_tokenizer = None, None
    if use_judge:
        if not criteria:
            raise ValueError("--criteria is required when using --judge")
        judge_model, judge_tokenizer = load_judge_model()

    print(f"\nDevice : {DEVICE}")
    print(f"Adapter: {adapter_path}")
    if criteria:
        print(f"Criteria: {criteria}")
    print("=" * 80)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt}")
        print("-" * 80)

        base_response  = generate(base_model,  base_tokenizer,  prompt)
        tuned_response = generate(tuned_model, tuned_tokenizer, prompt)

        base_score  = None
        tuned_score = None
        if use_judge:
            base_score  = score(judge_model, judge_tokenizer, criteria, prompt, base_response)
            tuned_score = score(judge_model, judge_tokenizer, criteria, prompt, tuned_response)

        # Print side by side
        print(f"\n{'BASE MODEL':^38} | {'FINE-TUNED MODEL':^38}")
        print("-" * 79)

        # Word-wrap both responses to 36 chars per column for side-by-side display
        base_lines  = _wrap(base_response,  36)
        tuned_lines = _wrap(tuned_response, 36)
        max_lines   = max(len(base_lines), len(tuned_lines))

        for j in range(max_lines):
            left  = base_lines[j]  if j < len(base_lines)  else ""
            right = tuned_lines[j] if j < len(tuned_lines) else ""
            print(f"{left:<38} | {right}")

        if use_judge:
            delta = tuned_score - base_score
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            print(f"\n  Score — Base: {base_score}/10   Tuned: {tuned_score}/10   {arrow} {delta:+d}")

        print("=" * 80)

        results.append({
            "prompt":         prompt,
            "base_response":  base_response,
            "tuned_response": tuned_response,
            "base_score":     base_score,
            "tuned_score":    tuned_score,
        })

    # Summary if judge was used
    if use_judge:
        avg_base  = sum(r["base_score"]  for r in results) / len(results)
        avg_tuned = sum(r["tuned_score"] for r in results) / len(results)
        delta     = avg_tuned - avg_base
        arrow     = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"\nSUMMARY")
        print(f"  Avg base score : {avg_base:.1f}/10")
        print(f"  Avg tuned score: {avg_tuned:.1f}/10")
        print(f"  Improvement    : {arrow} {delta:+.1f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({
                "adapter":  adapter_path,
                "criteria": criteria,
                "results":  results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to a fixed column width, preserving newlines."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        while len(paragraph) > width:
            lines.append(paragraph[:width])
            paragraph = paragraph[width:]
        lines.append(paragraph)
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare base model vs LoRA fine-tuned model side by side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare.py --adapter ./lora-adapter --prompt "Explain gravity"
  python compare.py --adapter ./lora-adapter --prompts "Explain gravity" "What is DNA?"
  python compare.py --adapter ./lora-adapter --prompts_file prompts.txt --criteria "concise" --judge
        """,
    )
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="Path to the saved LoRA adapter directory (output of train.py)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to compare on"
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        help="Multiple prompts passed on the command line"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None,
        help="Path to a .txt file with one prompt per line"
    )
    parser.add_argument(
        "--criteria", type=str, default=None,
        help="Criteria string used during training (required with --judge)"
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Score both responses with the judge model (requires --criteria)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to this JSON file"
    )
    args = parser.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Default to a small set of general prompts
        prompts = [
            "Explain what a neural network is.",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
            "Write a short poem about the ocean.",
            "What is machine learning?",
        ]

    compare(
        prompts=prompts,
        adapter_path=args.adapter,
        criteria=args.criteria,
        use_judge=args.judge,
        output_path=args.output,
    )
