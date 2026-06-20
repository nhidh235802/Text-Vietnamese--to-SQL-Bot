"""
Script 01: Tải GretelAI Synthetic Text-to-SQL dataset từ HuggingFace.

Dataset này chứa các schema phức tạp (nhiều bảng, JOINs) và đã có sẵn 
cấu trúc tạo bảng + dữ liệu mẫu trong cột `sql_context`.

Chạy:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --max-samples 100   (test nhanh)
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Tải GretelAI Text-to-SQL từ HuggingFace")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Thư mục lưu dữ liệu thô")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Giới hạn số mẫu mỗi split (để test nhanh)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  BƯỚC 1: TẢI GRETEL-AI TEXT-TO-SQL TỪ HUGGINGFACE")
    print("=" * 55)
    print("\nĐang tải dataset (~35MB)...")
    ds = load_dataset("gretelai/synthetic_text_to_sql")

    # Dataset này có split 'train' (100k) và 'test' (~5.8k).
    # Ta sẽ lưu ra train.jsonl và test.jsonl
    split_map = {
        "train": "train",
        "test": "test",
    }

    for out_name, hf_name in split_map.items():
        if hf_name not in ds:
            continue
            
        data = ds[hf_name]
        out_file = output_dir / f"{out_name}.jsonl"

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for i, record in enumerate(data):
                if args.max_samples and i >= args.max_samples:
                    break

                # Mapping key sang format của pipeline chúng ta
                out_record = {
                    "question_en": record["sql_prompt"],
                    "sql": record["sql"],
                    "sql_context": record["sql_context"],
                    "complexity": record.get("sql_complexity", ""),
                    "domain": record.get("domain", "")
                }
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                count += 1

        print(f"  ✅ {out_name}: {count:,} records → {out_file}")

    print(f"\n{'=' * 55}")
    print("  HOÀN TẤT! Tiếp theo chạy: python scripts/02_translate_data.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

