"""
Step 4: Train Small Language Models with LoRA for faithfulness detection.

Models (in priority order):
1. HuggingFaceTB/SmolLM-360M-Instruct
2. HuggingFaceTB/SmolLM-135M-Instruct
3. Qwen/Qwen2.5-0.5B-Instruct
4. TinyLlama/TinyLlama-1.1B-Chat-v1.0

Usage:
    python scripts/train_slm.py --model smollm360m
    python scripts/train_slm.py --model smollm135m
    python scripts/train_slm.py --model qwen05b
    python scripts/train_slm.py --model tinyllama
    python scripts/train_slm.py --model all
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# MPS optimization: allow MPS to use all available memory without artificial limits
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch.nn as nn


class FocalLossTrainer(Trainer):
    """Trainer with focal loss for class imbalance.
    Focal loss down-weights easy examples and focuses on hard ones.
    """
    def __init__(self, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        ce_loss = loss_fct(flat_logits, flat_labels)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss = focal_weight * ce_loss
        mask = (flat_labels != -100).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed", "microguard_combined")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
MAX_SEQ_LEN = 768

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

MODEL_CONFIGS = {
    "smollm135m": {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "chat_format": "smollm",
    },
    "gemma3_270m": {
        "name": "google/gemma-3-270m-it",
        "chat_format": "gemma",
    },
    "smollm360m": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "chat_format": "smollm",
    },
    "qwen05b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "chat_format": "qwen",
    },
    "gemma3_1b": {
        "name": "google/gemma-3-1b-it",
        "chat_format": "gemma",
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "chat_format": "tinyllama",
    },
}

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

SYSTEM_PROMPT = "You are a faithfulness evaluator for RAG systems. You must respond with exactly one word."

USER_TEMPLATE = """Context: {context}
Question: {query}
Answer: {answer}

Is every claim in the answer supported by the context? Respond with exactly one word: FAITHFUL or UNFAITHFUL."""

# Reserve tokens: system prompt + template chrome + query + answer + target + safety margin
# We budget tokens as: context gets whatever is left after everything else
TARGET_RESERVE_TOKENS = 10  # for "FAITHFUL"/"UNFAITHFUL" + EOS
PROMPT_OVERHEAD_TOKENS = 80  # system prompt + chat template markup + task instruction


def smart_truncate_context(query, context, answer, tokenizer=None, max_total=MAX_SEQ_LEN):
    """Truncate inputs with character limits calibrated so total tokens stay under max_total.

    For 768 token budget: ~300 chars query + ~500 chars answer + ~1500 chars context
    leaves ~50 tokens for template overhead + target.
    """
    query = query[:300]
    answer = answer[:500]
    context = context[:1500]
    return query, context, answer


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting per model
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(query, context, answer, label, chat_format, tokenizer):
    """Format as chat and return tokenizer-applied string."""
    query, context, answer = smart_truncate_context(query, context, answer, tokenizer)
    user_msg = USER_TEMPLATE.format(
        context=context,
        query=query,
        answer=answer,
    )
    target = "FAITHFUL" if label == "faithful" else "UNFAITHFUL"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": target},
    ]

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback manual formatting
        if chat_format == "tinyllama":
            text = (
                f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
                f"<|user|>\n{user_msg}</s>\n"
                f"<|assistant|>\n{target}</s>"
            )
        else:
            # Default chatml format
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{target}<|im_end|>"
            )

    return text


def format_prompt_inference(query, context, answer, chat_format, tokenizer):
    """Format prompt for inference (no target label)."""
    query, context, answer = smart_truncate_context(query, context, answer, tokenizer)
    user_msg = USER_TEMPLATE.format(
        context=context,
        query=query,
        answer=answer,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if chat_format == "tinyllama":
            text = (
                f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
                f"<|user|>\n{user_msg}</s>\n"
                f"<|assistant|>\n"
            )
        else:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_dataset(dataset, tokenizer, chat_format, max_len=MAX_SEQ_LEN):
    """Tokenize dataset for causal LM training with labels masked on input portion."""

    def tokenize_fn(examples):
        all_input_ids = []
        all_labels = []

        for i in range(len(examples["query"])):
            # Full text with target
            full_text = format_prompt(
                examples["query"][i],
                examples["context"][i],
                examples["answer"][i],
                examples["label"][i],
                chat_format,
                tokenizer,
            )

            # Prompt only (for masking)
            prompt_text = format_prompt_inference(
                examples["query"][i],
                examples["context"][i],
                examples["answer"][i],
                chat_format,
                tokenizer,
            )

            # Tokenize
            full_tokens = tokenizer(
                full_text, truncation=True, max_length=max_len,
                padding=False, return_tensors=None
            )
            prompt_tokens = tokenizer(
                prompt_text, truncation=True, max_length=max_len,
                padding=False, return_tensors=None
            )

            input_ids = full_tokens["input_ids"]
            # Mask prompt portion in labels (set to -100)
            prompt_len = len(prompt_tokens["input_ids"])
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # Ensure same length
            labels = labels[:len(input_ids)]

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": [[1] * len(ids) for ids in all_input_ids],
            "labels": all_labels,
        }

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation during training
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, dataset, chat_format, device, max_samples=2000):
    """Run inference with constrained decoding — compare logits for FAITHFUL vs UNFAITHFUL."""
    model.eval()
    y_true = []
    y_pred = []

    # Subsample if too large
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        np.random.seed(SEED)
        indices = np.random.choice(indices, max_samples, replace=False).tolist()

    # Get token IDs for constrained decoding
    faithful_ids = tokenizer.encode("FAITHFUL", add_special_tokens=False)
    unfaithful_ids = tokenizer.encode("UNFAITHFUL", add_special_tokens=False)

    for i, idx in enumerate(indices):
        ex = dataset[idx]
        prompt = format_prompt_inference(
            ex["query"], ex["context"], ex["answer"], chat_format, tokenizer
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            next_logits = outputs.logits[:, -1, :]
            f_score = next_logits[0, faithful_ids[0]].item() if faithful_ids else -999
            u_score = next_logits[0, unfaithful_ids[0]].item() if unfaithful_ids else -999
            pred = "faithful" if f_score > u_score else "unfaithful"

        y_true.append(ex["label"])
        y_pred.append(pred)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(indices)}...")

    # Compute metrics
    y_true_bin = [1 if l == "faithful" else 0 for l in y_true]
    y_pred_bin = [1 if l == "faithful" else 0 for l in y_pred]

    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average="macro")

    return {"balanced_accuracy": bal_acc, "f1_macro": f1, "n_samples": len(y_true), "n_skipped": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model_key, max_train_samples=None, num_epochs=3):
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    chat_format = config["chat_format"]

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    if max_train_samples:
        print(f"  (using {max_train_samples} training samples)")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*70}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if DEVICE == "mps" else torch.float16,
        trust_remote_code=True,
    )

    # Apply LoRA (with fallback for models with different attention names)
    try:
        model = get_peft_model(model, LORA_CONFIG)
    except ValueError:
        print("  Adjusting LoRA targets (fallback to q_proj, v_proj)...")
        fallback_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none",
        )
        model = get_peft_model(model, fallback_config)
    model.print_trainable_parameters()

    # Move to device
    model = model.to(DEVICE)

    # Load dataset
    ds = load_from_disk(DATA_DIR)
    print(f"Full dataset — Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

    # Subsample training data if requested (stratified)
    train_split = ds["train"]
    if max_train_samples and max_train_samples < len(train_split):
        labels = train_split["label"]
        from sklearn.model_selection import train_test_split
        keep_idx, _ = train_test_split(
            range(len(train_split)), train_size=max_train_samples,
            stratify=labels, random_state=SEED
        )
        train_split = train_split.select(sorted(keep_idx))
        print(f"Subsampled to {len(train_split)} training examples")

    # Subsample validation too for speed
    val_split = ds["validation"]
    if max_train_samples and len(val_split) > 3000:
        val_labels = val_split["label"]
        keep_idx, _ = train_test_split(
            range(len(val_split)), train_size=3000,
            stratify=val_labels, random_state=SEED
        )
        val_split = val_split.select(sorted(keep_idx))

    # Tokenize
    train_ds = tokenize_dataset(train_split, tokenizer, chat_format)
    val_ds = tokenize_dataset(val_split, tokenizer, chat_format)

    # Data collator — use DYNAMIC padding (pad to longest in batch, not max_length)
    # This is the #1 speedup: most examples are ~400 tokens, so we avoid padding to 768
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",  # Dynamic padding — huge speedup vs padding to max_length
        return_tensors="pt",
    )

    # Training arguments — optimized for MPS
    output_dir = os.path.join(MODELS_DIR, model_key)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,   # Smaller batch = less MPS memory pressure
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # Effective batch still 32
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        bf16=False,  # MPS doesn't support bf16
        fp16=False,  # Keep fp32 for MPS stability
        dataloader_num_workers=0,  # MPS compatibility
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,  # MPS doesn't support pinned memory
        group_by_length=True,  # Group similar-length examples → less padding waste
    )

    # Track training time
    start_time = time.time()

    trainer = FocalLossTrainer(
        focal_gamma=2.0,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Save model
    trainer.save_model(os.path.join(output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best"))

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, tokenizer, ds["validation"], chat_format, DEVICE)

    # Peak memory
    if DEVICE == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    elif DEVICE == "mps":
        peak_memory_mb = torch.mps.driver_allocated_memory() / 1e6
    else:
        peak_memory_mb = 0

    # Save results
    results = {
        "model": model_name,
        "model_key": model_key,
        "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time/3600:.1f}h",
        "peak_memory_mb": peak_memory_mb,
        "train_loss": train_result.training_loss,
        "train_samples": len(train_ds),
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_samples_evaluated": val_metrics["n_samples"],
        "hyperparameters": {
            "learning_rate": 2e-4,
            "batch_size": 8,
            "gradient_accumulation": 4,
            "effective_batch_size": 32,
            "epochs": num_epochs,
            "max_seq_len": MAX_SEQ_LEN,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
    }

    results_path = os.path.join(RESULTS_DIR, f"train_{model_key}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete: {model_name}")
    print(f"  Time: {results['training_time_formatted']}")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Val balanced accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Val F1 macro: {val_metrics['f1_macro']:.4f}")
    print(f"  Peak memory: {peak_memory_mb:.0f} MB")
    print(f"  Results saved to {results_path}")
    print(f"  Model saved to {os.path.join(output_dir, 'best')}")
    print(f"{'='*70}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="Model to train")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples (stratified subsample)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    args = parser.parse_args()

    if args.model == "all":
        all_results = {}
        for key in MODEL_CONFIGS:
            try:
                all_results[key] = train_model(key, args.max_train_samples, args.epochs)
            except Exception as e:
                print(f"ERROR training {key}: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = {"error": str(e)}

        # Save combined results
        combined_path = os.path.join(RESULTS_DIR, "train_all_slms.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_path}")
    else:
        train_model(args.model, args.max_train_samples, args.epochs)
