"""
Simple local inference script for Apple Silicon (M2 Pro).
Model: Qwen2.5-0.5B-Instruct
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32


def load_model(model_id: str = MODEL_ID):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    model, tokenizer = load_model()

    print("Type your prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        prompt = input("You: ").strip()
        if not prompt or prompt.lower() in {"quit", "exit"}:
            break
        response = generate(model, tokenizer, prompt)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
