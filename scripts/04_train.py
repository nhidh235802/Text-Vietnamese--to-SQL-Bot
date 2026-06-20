"""
Script 04: Fine-tune Gemma-2-2B trên dữ liệu Text-to-SQL tiếng Việt.

Chạy:
    python scripts/04_train.py                          (mặc định)
    python scripts/04_train.py --epochs 5 --batch-size 8
    python scripts/04_train.py --resume                 (tiếp tục từ checkpoint)

Yêu cầu: GPU NVIDIA (RTX 3060 12GB VRAM trở lên)
"""

import torch
import argparse
from pathlib import Path

# ─── CẤU HÌNH MẶC ĐỊNH ──────
MODEL_NAME_UNSLOTH = "unsloth/gemma-2-2b-it-bnb-4bit"
MODEL_NAME_HF = "google/gemma-2-2b-it"
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

ALPACA_PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-2-2B cho Text-to-SQL")
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--eval-data", default="data/processed/val.jsonl")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Lưu checkpoint + evaluate mỗi N steps")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục training từ checkpoint gần nhất")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  BƯỚC 4: FINE-TUNE GEMMA-2-2B (TEXT-TO-SQL)")
    print("=" * 55)

    # ─── 1. LOAD MODEL ──────────────────────────────
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("\n✅ Unsloth detected — training sẽ nhanh hơn ~2x!")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME_UNSLOTH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGETS,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    except ImportError:
        print("\n⚠️ Unsloth chưa cài hoặc không tương thích Windows.")
        print("  Dùng peft + transformers (chậm hơn nhưng ổn định)...")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"  Đang tải {MODEL_NAME_HF}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_HF,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Đảm bảo có pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # ─── 2. LOAD DATA ───────────────────────────────
    from datasets import load_dataset

    print(f"\nĐang tải dữ liệu training...")
    dataset = load_dataset("json", data_files={
        "train": args.train_data,
        "eval": args.eval_data,
    })

    EOS_TOKEN = tokenizer.eos_token or ""

    def format_prompts(examples):
        texts = []
        for inst, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = ALPACA_PROMPT.format(inst, inp, out) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Eval:  {len(dataset['eval']):,} samples")

    # ─── 3. TRAINING CONFIG ─────────────────────────
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,

        # Batch & Steps
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,

        # Learning rate
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,

        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # Logging & Saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.save_steps,

        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
    )

    # ─── 4. BẮT ĐẦU TRAINING ───────────────────────
    print(f"\n{'─' * 55}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Save/Eval mỗi: {args.save_steps} steps")
    print(f"{'─' * 55}")

    print("\n🚀 BẮT ĐẦU TRAINING...\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ─── 5. LƯU MODEL CUỐI CÙNG ────────────────────
    final_dir = output_dir / "gemma2-text2sql-lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n{'=' * 55}")
    print(f"  ✅ TRAINING HOÀN TẤT!")
    print(f"  Model LoRA saved → {final_dir}")
    print(f"\n  Tiếp theo:")
    print(f"    python scripts/05_evaluate.py --model {final_dir}")
    print(f"    python scripts/05_evaluate.py --eval-all")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
