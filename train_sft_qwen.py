"""
SFT trainer for Long Horizon Memory action generation.

Trains a LoRA adapter on top of Qwen/Qwen2.5-1.5B-Instruct using seed data in
data.py. The model learns to emit action JSON parseable by LongHorizonMemoryAction.

Usage:
    python train_sft_qwen.py

Recommended install (GPU):
    pip install -U transformers datasets peft trl accelerate bitsandbytes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data import SEED_DATA, SYSTEM_PROMPT, action_to_json, format_observation


@dataclass
class TrainConfig:
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir: str = os.getenv("OUTPUT_DIR", "./memory_action_sft_qwen15b")
    max_length: int = int(os.getenv("MAX_LENGTH", "768"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    num_train_epochs: int = int(os.getenv("NUM_TRAIN_EPOCHS", "2"))
    per_device_train_batch_size: int = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    seed_repeat_factor: int = int(os.getenv("SEED_REPEAT_FACTOR", "10"))
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))


CFG = TrainConfig()


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model() -> AutoModelForCausalLM:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for Qwen2.5-1.5B SFT in this script.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model


def apply_chat_template(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user_prompt}\n"
            "<|assistant|>\n"
        )


def build_rows(tokenizer: AutoTokenizer) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for sample in SEED_DATA:
        obs = sample["observation"]
        action = sample["response"]
        prompt_body = format_observation(obs)
        prompt = apply_chat_template(tokenizer, prompt_body)
        target = action_to_json(action)
        rows.append({"prompt": prompt, "full_text": prompt + target})
    return rows


def tokenize_and_mask(example: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    full = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=CFG.max_length,
        padding="max_length",
    )
    prompt = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=CFG.max_length,
        padding="max_length",
    )

    labels = list(full["input_ids"])
    prompt_len = sum(1 for t in prompt["attention_mask"] if t == 1)
    for i in range(prompt_len):
        labels[i] = -100
    labels = [
        -100 if tok == tokenizer.pad_token_id else lab
        for tok, lab in zip(full["input_ids"], labels)
    ]
    full["labels"] = labels
    return full


def main() -> None:
    print(f"[SFT] Loading tokenizer: {CFG.model_name}")
    tokenizer = build_tokenizer()

    rows = build_rows(tokenizer)
    rows = rows * max(1, CFG.seed_repeat_factor)
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda ex: tokenize_and_mask(ex, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=1,  # keep single process for broad compatibility
    )

    print(f"[SFT] Training samples: {len(dataset)}")
    print("[SFT] Loading 4-bit base model + LoRA adapters...")
    model = build_model()

    args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        seed=42,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("[SFT] Starting training...")
    trainer.train()
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    print(f"[SFT] Saved adapter and tokenizer to: {CFG.output_dir}")


if __name__ == "__main__":
    main()
