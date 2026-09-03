"""
pipeline/finetune.py

LoRA fine-tuning of Mistral-7B (or Llama 3.1 8B) on Darwin's corpus.

Requires:
  - GPU with ~20GB VRAM (A100 on RunPod/Colab works well)
  - pip install transformers peft accelerate bitsandbytes datasets trl

Cost estimate:
  - RunPod A100 (~$1.50/hr) x ~10 hours = ~$15 for 3 epochs on full corpus
  - Colab Pro+ works for smaller corpus subsets

Why fine-tune at all?
  RAG alone captures what Darwin said. Fine-tuning captures how he said it —
  the sentence rhythms, the self-deprecating hedges, the way he builds to
  a conclusion. Both matter. The failure modes are different.

Usage:
    python pipeline/finetune.py --corpus data/cleaned/ --output models/darwin-mistral/
    python pipeline/finetune.py --corpus data/cleaned/ --base-model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_DIR, MODELS_DIR, FINETUNE_CONFIG, LOCAL_BASE_MODEL


def load_training_texts(corpus_dir: Path, max_tokens: Optional[int] = None) -> list[str]:
    """
    Load Darwin texts formatted for causal language modeling.

    Each training example is: [DARWIN | {source} | {date}]\n{text}
    The source/date prefix helps the model learn register variation.
    """
    texts = []
    total_tokens = 0

    json_files = [f for f in corpus_dir.rglob("*.json") if f.name != "manifest.json"]
    json_files.sort(key=lambda f: f.name)

    for filepath in json_files:
        try:
            doc = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception:
            continue

        text = doc.get("text", "")
        if not text or len(text) < 200:
            continue

        # Format prefix carries metadata as context signal
        source = doc.get("source", "unknown")
        date = doc.get("date_str", "unknown")
        register = doc.get("register", "unknown")
        recipient = doc.get("recipient", "")

        prefix = f"[DARWIN | {source} | {date} | {register}"
        if recipient:
            prefix += f" | to: {recipient}"
        prefix += "]\n"

        formatted = prefix + text + "\n"
        est_tokens = len(formatted) // 4

        if max_tokens and total_tokens + est_tokens > max_tokens:
            break

        texts.append(formatted)
        total_tokens += est_tokens

    print(f"  Loaded {len(texts)} documents (~{total_tokens:,} tokens)")
    return texts


def run_finetuning(
    corpus_dir: Path,
    output_dir: Path,
    base_model: str = LOCAL_BASE_MODEL,
    config: dict = None,
):
    """Run LoRA fine-tuning. Requires GPU and transformers/peft."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install transformers peft accelerate bitsandbytes datasets trl")
        sys.exit(1)

    cfg = config or FINETUNE_CONFIG
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"── Fine-tuning Darwin model ───────────────────────────")
    print(f"  Base model: {base_model}")
    print(f"  Output: {output_dir}")

    # Load texts
    texts = load_training_texts(corpus_dir)
    if not texts:
        print("  ✗ No training texts found. Run scraper + cleaner first.")
        sys.exit(1)

    # Load tokenizer
    print(f"\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    print(f"  Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Split train/eval
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model with quantization for memory efficiency
    print(f"\n  Loading model (this takes a few minutes)...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback without quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        fp16=cfg.get("fp16", True),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",  # disable wandb by default
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"\n  Starting training...")
    trainer.train()

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✓ Fine-tuning complete → {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CLEANED_DIR)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "darwin-mistral")
    parser.add_argument("--base-model", default=LOCAL_BASE_MODEL)
    args = parser.parse_args()

    run_finetuning(args.corpus, args.output, args.base_model)


if __name__ == "__main__":
    main()
