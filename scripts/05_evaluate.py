"""
Script 05: Đánh giá model Text-to-SQL.

Metrics:
  - Syntax Accuracy:    SQL sinh ra có hợp lệ (parse được) không?
  - Execution Accuracy: SQL chạy trên DB mẫu có ra đúng kết quả không?
  - Exact Match:        SQL sinh ra có giống hệt SQL đáp án không?

Chạy:
    python scripts/05_evaluate.py --model base                         (baseline)
    python scripts/05_evaluate.py --model models/gemma2-text2sql-lora  (fine-tuned)
    python scripts/05_evaluate.py --eval-all                           (tất cả checkpoints)
"""

import json
import re
import sqlite3
import argparse
import time
from pathlib import Path

MODEL_NAME_UNSLOTH = "unsloth/gemma-2-2b-it-bnb-4bit"
MODEL_NAME_HF = "google/gemma-2-2b-it"
MAX_SEQ_LENGTH = 512

INSTRUCTION = "Dựa vào bảng dữ liệu được cung cấp, hãy viết truy vấn SQL để trả lời câu hỏi."

ALPACA_PROMPT_INFERENCE = """### Instruction:
{}

### Input:
{}

### Response:
"""


# ─── SQL EXECUTION ──────────────────────────────────

def execute_sql_on_context(sql, sql_context):
    """
    Tạo DB SQLite tạm, chạy lệnh khởi tạo (sql_context), sau đó chạy SQL cần test.

    Returns:
        (results, error): results là list of tuples nếu thành công, None nếu lỗi.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    try:
        # Khởi tạo Schema và dữ liệu mẫu từ sql_context
        cursor.executescript(sql_context)
        
        # Chạy truy vấn SQL sinh ra
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None

    except Exception as e:
        conn.close()
        return None, str(e)


def check_sql_syntax(sql, sql_context):
    """Kiểm tra SQL có hợp lệ không (parse bằng SQLite)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(sql_context)
        conn.execute(f"EXPLAIN {sql}")
        conn.close()
        return True
    except Exception:
        conn.close()
        return False


# ─── MODEL INFERENCE ────────────────────────────────

def load_model(model_path):
    """Load model (base hoặc fine-tuned LoRA)."""
    import torch

    is_base = model_path == "base"

    try:
        from unsloth import FastLanguageModel

        if is_base:
            print(f"  Loading base model: {MODEL_NAME_UNSLOTH}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME_UNSLOTH,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            print(f"  Loading LoRA model: {model_path}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )

        FastLanguageModel.for_inference(model)

    except ImportError:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        )

        if is_base:
            print(f"  Loading base model: {MODEL_NAME_HF}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_HF,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            print(f"  Loading LoRA model: {model_path}")
            # Load base model + LoRA adapter
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_HF,
                quantization_config=bnb_config,
                device_map="auto",
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, model_path)

        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_sql(model, tokenizer, input_text, max_new_tokens=128):
    """Sinh SQL từ model."""
    import torch

    prompt = ALPACA_PROMPT_INFERENCE.format(INSTRUCTION, input_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Chỉ lấy phần model sinh ra (bỏ prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Lấy dòng SQL đầu tiên (bỏ giải thích nếu có)
    sql = response.split("\n")[0].strip()

    # Loại bỏ ký tự thừa ở cuối
    sql = sql.rstrip(";").strip()

    return sql


# ─── EVALUATION ─────────────────────────────────────

def evaluate_model_on_data(model, tokenizer, eval_data, max_samples=200):
    """Đánh giá model trên tập eval, trả về dict metrics."""
    results = {
        "total": 0,
        "syntax_correct": 0,
        "execution_correct": 0,
        "exact_match": 0,
        "samples": [],
    }

    total = min(len(eval_data), max_samples)
    print(f"\n  Đang evaluate {total} câu...")
    start_time = time.time()

    for i, record in enumerate(eval_data[:total]):
        input_text = record["input"]
        gold_sql = record["output"]
        sql_context = record.get("sql_context", "")

        # Sinh SQL
        pred_sql = generate_sql(model, tokenizer, input_text)

        # 1. Syntax check
        is_syntax_ok = check_sql_syntax(pred_sql, sql_context)
        if is_syntax_ok:
            results["syntax_correct"] += 1

        # 2. Execution accuracy
        is_exec_ok = False
        if is_syntax_ok:
            pred_result, pred_err = execute_sql_on_context(pred_sql, sql_context)
            gold_result, gold_err = execute_sql_on_context(gold_sql, sql_context)

            if pred_result is not None and gold_result is not None:
                is_exec_ok = pred_result == gold_result

        if is_exec_ok:
            results["execution_correct"] += 1

        # 3. Exact match (normalize whitespace)
        pred_norm = " ".join(pred_sql.lower().split())
        gold_norm = " ".join(gold_sql.lower().split())
        is_exact = pred_norm == gold_norm
        if is_exact:
            results["exact_match"] += 1

        results["total"] += 1

        # Lưu sample
        if i < 20:
            results["samples"].append({
                "input": input_text,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "syntax_ok": is_syntax_ok,
                "exec_ok": is_exec_ok,
                "exact_match": is_exact,
            })

        # Log tiến trình
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            syn_acc = results["syntax_correct"] / results["total"] * 100
            exec_acc = results["execution_correct"] / results["total"] * 100
            print(
                f"    [{i+1}/{total}] "
                f"Syntax: {syn_acc:.1f}% | "
                f"Exec: {exec_acc:.1f}% | "
                f"ETA: {eta:.0f}s"
            )

    # Tính tỉ lệ
    t = results["total"]
    results["syntax_accuracy"] = results["syntax_correct"] / t * 100 if t else 0
    results["execution_accuracy"] = results["execution_correct"] / t * 100 if t else 0
    results["exact_match_accuracy"] = results["exact_match"] / t * 100 if t else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model Text-to-SQL")
    parser.add_argument("--model", type=str, default="base",
                        help="'base' cho model gốc, hoặc đường dẫn tới LoRA adapter")
    parser.add_argument("--eval-data", type=str, default="data/processed/val.jsonl",
                        help="Validation set (dùng để đánh giá mỗi checkpoint)")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl",
                        help="Test set (chỉ dùng cho --final-test)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Số câu tối đa để evaluate")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--eval-all", action="store_true",
                        help="Evaluate tất cả checkpoints trên val set")
    parser.add_argument("--final-test", action="store_true",
                        help="Evaluate model cuối cùng trên TEST set (chỉ chạy 1 lần)")
    args = parser.parse_args()

    # Đọc eval data
    eval_data = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    print(f"  Eval dataset: {len(eval_data)} câu (dùng {min(len(eval_data), args.max_samples)})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_all:
        # ─── Evaluate tất cả checkpoints ─────────────
        models_dir = Path("models")
        checkpoints = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]),
        )

        # Thêm baseline
        model_list = [("base", "base")] + [(cp.name, str(cp)) for cp in checkpoints]

        # Thêm final model nếu có
        final_model = models_dir / "gemma2-text2sql-lora"
        if final_model.exists():
            model_list.append(("final", str(final_model)))

        print(f"\n  Tìm thấy {len(model_list)} models cần evaluate:")
        for name, path in model_list:
            print(f"    - {name}: {path}")

        all_results = {}
        for name, model_path in model_list:
            print(f"\n{'═' * 55}")
            print(f"  ĐANG EVALUATE: {name}")
            print(f"{'═' * 55}")

            model, tokenizer = load_model(model_path)
            results = evaluate_model_on_data(model, tokenizer, eval_data, args.max_samples)
            all_results[name] = results

            # Lưu report riêng
            report_file = output_dir / f"eval_{name}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n  📊 {name}:")
            print(f"    Syntax Accuracy:    {results['syntax_accuracy']:.1f}%")
            print(f"    Execution Accuracy: {results['execution_accuracy']:.1f}%")
            print(f"    Exact Match:        {results['exact_match_accuracy']:.1f}%")

            # Giải phóng VRAM
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()

        # Lưu tổng hợp
        summary_file = output_dir / "eval_summary.json"
        summary = {}
        for name, results in all_results.items():
            summary[name] = {
                "syntax_accuracy": results["syntax_accuracy"],
                "execution_accuracy": results["execution_accuracy"],
                "exact_match_accuracy": results["exact_match_accuracy"],
            }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'═' * 55}")
        print(f"  ✅ TẤT CẢ EVALUATION HOÀN TẤT!")
        print(f"  Summary: {summary_file}")
        print(f"  Tiếp theo: python scripts/06_plot_results.py")
        print(f"{'═' * 55}")

    elif args.final_test:
        # ─── Evaluate trên TEST SET (chỉ chạy 1 lần cuối) ──
        test_data_path = Path(args.test_data)
        if not test_data_path.exists():
            print(f"❌ Không tìm thấy test set: {test_data_path}")
            print("   Hãy dịch test set trước:")
            print("   python scripts/02_translate_data.py --split test --max-samples 500")
            return

        test_data = []
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))

        model_path = args.model
        if model_path == "base":
            # Mặc định dùng final model cho --final-test
            final = Path("models/gemma2-text2sql-lora")
            if final.exists():
                model_path = str(final)
            else:
                print("⚠️ Không tìm thấy final model, dùng base model...")

        print(f"\n{'═' * 55}")
        print(f"  🏆 FINAL TEST — {model_path}")
        print(f"  Test set: {len(test_data)} câu (CHƯA BAO GIỜ ĐƯỢC DÙNG TRƯỚC ĐÂY)")
        print(f"{'═' * 55}")

        model, tokenizer = load_model(model_path)
        results = evaluate_model_on_data(model, tokenizer, test_data, args.max_samples)

        report_file = output_dir / "final_test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'═' * 55}")
        print(f"  🏆 KẾT QUẢ FINAL TEST:")
        print(f"{'─' * 55}")
        print(f"  Syntax Accuracy:    {results['syntax_accuracy']:.1f}%")
        print(f"  Execution Accuracy: {results['execution_accuracy']:.1f}%")
        print(f"  Exact Match:        {results['exact_match_accuracy']:.1f}%")
        print(f"{'─' * 55}")
        print(f"  Report: {report_file}")
        print(f"{'═' * 55}")

    else:
        # ─── Evaluate 1 model ────────────────────────
        print(f"\n{'═' * 55}")
        print(f"  ĐÁNH GIÁ MODEL: {args.model}")
        print(f"{'═' * 55}")

        model, tokenizer = load_model(args.model)
        results = evaluate_model_on_data(model, tokenizer, eval_data, args.max_samples)

        # Tên file output
        model_name = args.model.replace("/", "_").replace("\\", "_")
        if args.model == "base":
            model_name = "base"
        report_file = output_dir / f"eval_{model_name}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'═' * 55}")
        print(f"  📊 KẾT QUẢ ({args.model}):")
        print(f"{'─' * 55}")
        print(f"  Syntax Accuracy:    {results['syntax_accuracy']:.1f}%")
        print(f"  Execution Accuracy: {results['execution_accuracy']:.1f}%")
        print(f"  Exact Match:        {results['exact_match_accuracy']:.1f}%")
        print(f"{'─' * 55}")
        print(f"  Report: {report_file}")

        # Hiển thị vài ví dụ
        print(f"\n  VÍ DỤ:")
        for s in results["samples"][:5]:
            status = "✅" if s["exec_ok"] else ("⚠️" if s["syntax_ok"] else "❌")
            print(f"    {status} Gold: {s['gold_sql']}")
            print(f"       Pred: {s['pred_sql']}")
            print()

        print(f"{'═' * 55}")


if __name__ == "__main__":
    main()

