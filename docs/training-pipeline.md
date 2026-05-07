# Training Pipeline

End-to-end example of LoRA fine-tuning with a custom loss and optimizer.

---

## Dependencies

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

---

## Full Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import torch

# 1. Load model and tokenizer
model_id  = "meta-llama/Llama-3.2-1B"
# alternatives: "microsoft/phi-3-mini-4k-instruct", "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer  = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)

# 3. Custom Trainer — swap loss function here
class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        loss    = focal_loss(outputs.logits, labels, gamma=2.0)  # swap as needed
        return (loss, outputs) if return_outputs else loss

# 4. Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 5. Train
trainer = CustomTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.optimizer = Lion(model.parameters(), lr=1e-4)  # swap optimizer here
trainer.train()

# 6. Save adapter weights only
model.save_pretrained("./lora-adapter")
```

---

## Inference with Saved Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model      = PeftModel.from_pretrained(base_model, "./lora-adapter")
model.eval()

inputs  = tokenizer("Explain quantum entanglement:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Merging Adapter into Base Model

Produces a single standalone model with no PEFT dependency at inference time:

```python
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```
