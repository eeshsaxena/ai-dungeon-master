"""
QLoRA fine-tuning script for the HP DM narrator (Phase 8).

Trains a LoRA adapter on the HP DM dataset using PEFT + TRL SFTTrainer.
Default base model: microsoft/phi-3-mini-4k-instruct (3.8 B params).
  - 4-bit QLoRA fits on 8 GB VRAM for training.
  - Resulting adapter is ~50-150 MB, stored in lora_train/adapter/.

Usage:
    python train.py [options]

    --model      HuggingFace model ID or local path   (default: Phi-3-mini)
    --adapter    Output adapter directory              (default: ./adapter)
    --data       Training JSONL file                   (default: auto-built)
    --epochs     Training epochs                       (default: 3)
    --lr         Learning rate                         (default: 2e-4)
    --batch      Per-device batch size                 (default: 2)
    --grad-acc   Gradient accumulation steps           (default: 4)
    --max-len    Max sequence length                   (default: 1024)
    --lora-r     LoRA rank                             (default: 16)
    --lora-a     LoRA alpha                            (default: 32)
    --no-4bit    Disable 4-bit quantization (uses fp16)
"""
import argparse
import os
from pathlib import Path

# ── Parse args before heavy imports (faster --help) ────────────────────────────
parser = argparse.ArgumentParser(description="QLoRA fine-tuning for HP DM")
parser.add_argument("--model",    default="microsoft/phi-3-mini-4k-instruct")
parser.add_argument("--adapter",  default=str(Path(__file__).parent / "adapter"))
parser.add_argument("--data",     default=None, help="Path to training JSONL (built if omitted)")
parser.add_argument("--epochs",   type=int, default=3)
parser.add_argument("--lr",       type=float, default=2e-4)
parser.add_argument("--batch",    type=int, default=2)
parser.add_argument("--grad-acc", type=int, default=4, dest="grad_acc")
parser.add_argument("--max-len",  type=int, default=1024, dest="max_len")
parser.add_argument("--lora-r",   type=int, default=16, dest="lora_r")
parser.add_argument("--lora-a",   type=int, default=32, dest="lora_a")
parser.add_argument("--no-4bit",  action="store_true", dest="no_4bit")
args = parser.parse_args()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer

sys_path_parent = str(Path(__file__).parent.parent)
import sys; sys.path.insert(0, sys_path_parent)
from lora_train.dataset_builder import build_dataset, save_dataset, OUT_PATH


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_data(data_path: str | None) -> Dataset:
    if data_path and Path(data_path).exists():
        import json
        rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
        print(f"[Train] Loaded {len(rows)} examples from {data_path}")
    else:
        # Build from lore + handcrafted examples
        rows = build_dataset()
        save_dataset(rows, OUT_PATH)
        print(f"[Train] Built {len(rows)} examples from lore corpus")
    return Dataset.from_list(rows)


# ── LoRA target modules by model family ────────────────────────────────────────

_LORA_TARGETS = {
    "phi":    ["q_proj", "v_proj", "k_proj", "dense"],
    "llama":  ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen":   ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default":["q_proj", "v_proj"],
}

def _lora_targets(model_id: str) -> list[str]:
    lower = model_id.lower()
    for key in _LORA_TARGETS:
        if key in lower:
            return _LORA_TARGETS[key]
    return _LORA_TARGETS["default"]


# ── Formatting function for SFTTrainer ─────────────────────────────────────────

def _format_chat(example: dict, tokenizer) -> dict:
    """Apply the model's chat template to a conversations example."""
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[Train] Base model : {args.model}")
    print(f"[Train] Adapter out: {args.adapter}")
    print(f"[Train] Epochs     : {args.epochs}")
    print(f"[Train] LoRA rank  : {args.lora_r}  alpha: {args.lora_a}")
    print(f"[Train] 4-bit QLoRA: {'no' if args.no_4bit else 'yes'}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Quantization config
    bnb_cfg = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.no_4bit else None,
        attn_implementation="eager",   # avoid flash-attn dependency
    )

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_a,
        lora_dropout=0.05,
        bias="none",
        target_modules=_lora_targets(args.model),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5. Dataset
    dataset = load_data(args.data)
    dataset = dataset.map(lambda ex: _format_chat(ex, tokenizer))

    # 6. Training args
    train_args = TrainingArguments(
        output_dir=args.adapter,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        dataloader_pin_memory=False,
    )

    # 7. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        args=train_args,
        packing=False,
    )

    print("[Train] Starting training…")
    trainer.train()

    # 8. Save adapter
    adapter_path = Path(args.adapter)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[Train] Adapter saved → {adapter_path}")
    print("[Train] Done! Set LLM_PROVIDER=lora and LORA_ADAPTER_PATH in .env to use it.")


if __name__ == "__main__":
    main()
