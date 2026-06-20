"""
Script 03: Chuyển dữ liệu đã dịch sang format Alpaca cho SFT training.

Chạy:
    python scripts/03_prepare_training.py
    python scripts/03_prepare_training.py --train-input data/translated/train_vi.jsonl
"""

import json
import random
import argparse
from pathlib import Path

INSTRUCTION = "Dựa vào bảng dữ liệu được cung cấp, hãy viết truy vấn SQL để trả lời câu hỏi."


def format_record(record):
    """Chuyển 1 record thành format Alpaca (instruction / input / output)."""
    # Context chứa CREATE TABLE
    input_text = f"Lược đồ cơ sở dữ liệu:\n{record['sql_context']}\n\nCâu hỏi:\n{record['question_vi']}"

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": record["sql"],
        # Giữ lại context để chạy test execution ở bước sau
        "sql_context": record["sql_context"],
    }


def load_and_format(filepath):
    """Đọc file JSONL đã dịch và format thành Alpaca."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            records.append(format_record(raw))
    return records


def save_jsonl(records, filepath):
    """Lưu danh sách records ra file JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu training (Alpaca format)")
    parser.add_argument("--train-input", default="data/translated/train_vi.jsonl")
    parser.add_argument("--val-input", default="data/translated/dev_vi.jsonl")
    parser.add_argument("--test-input", default="data/translated/test_vi.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-split-ratio", type=float, default=0.1,
                        help="Nếu không có val input riêng, tách %% từ train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  BƯỚC 3: CHUẨN BỊ DỮ LIỆU TRAINING")
    print("=" * 55)

    # ─── Train data ─────────────────────────────
    train_path = Path(args.train_input)
    if not train_path.exists():
        print(f"❌ Không tìm thấy: {train_path}")
        print("   Hãy chạy scripts/02_translate_data.py trước!")
        return

    train_records = load_and_format(train_path)
    print(f"  Train input: {len(train_records)} records")

    # ─── Validation data (dùng để theo dõi trong lúc train + vẽ learning curve) ──
    val_path = Path(args.val_input)
    if val_path.exists():
        val_records = load_and_format(val_path)
        print(f"  Val input:   {len(val_records)} records (từ {val_path})")
    else:
        print(f"  ⚠️ Không tìm thấy {val_path}, tách {args.val_split_ratio*100:.0f}% từ train...")
        random.seed(args.seed)
        random.shuffle(train_records)
        split_idx = int(len(train_records) * (1 - args.val_split_ratio))
        val_records = train_records[split_idx:]
        train_records = train_records[:split_idx]

    # ─── Test data (chỉ dùng 1 lần cuối cùng để báo cáo) ──
    test_path = Path(args.test_input)
    test_records = []
    if test_path.exists():
        test_records = load_and_format(test_path)
        print(f"  Test input:  {len(test_records)} records (từ {test_path})")
    else:
        print(f"  ⚠️ Không có test set riêng ({test_path})")

    # ─── Lưu ─────────────────────────────────────
    splits = [
        ("train", train_records),
        ("val", val_records),
    ]
    if test_records:
        splits.append(("test", test_records))

    print()
    for name, data in splits:
        out_file = output_dir / f"{name}.jsonl"
        save_jsonl(data, out_file)
        print(f"  ✅ {name:5s}: {len(data):,} records → {out_file}")

    # Hiển thị ví dụ
    print(f"\n{'─' * 55}")
    print("  VÍ DỤ MẪU:")
    print(f"{'─' * 55}")
    sample = train_records[0]
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Input:       {sample['input']}")
    print(f"  Output:      {sample['output']}")

    print(f"\n{'=' * 55}")
    print("  HOÀN TẤT! Tiếp theo chạy: python scripts/04_train.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

