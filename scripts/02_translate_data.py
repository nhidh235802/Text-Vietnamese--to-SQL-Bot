"""
Script 02: Dịch câu hỏi từ tiếng Anh sang tiếng Việt bằng Google Translate (miễn phí).

Có checkpoint: nếu bị ngắt giữa chừng, chạy lại sẽ tiếp tục từ chỗ dừng.

Chạy:
    python scripts/02_translate_data.py                        (dịch 5000 câu train)
    python scripts/02_translate_data.py --max-samples 100      (test nhanh 100 câu)
    python scripts/02_translate_data.py --split dev --max-samples 500  (dịch dev set)
"""

import json
import time
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Dịch câu hỏi EN → VN bằng Google Translate")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev", "test"],
                        help="Split cần dịch")
    parser.add_argument("--input-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/translated")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Số câu tối đa cần dịch")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Thời gian chờ (giây) giữa mỗi request")
    args = parser.parse_args()

    input_path = Path(args.input_dir) / f"{args.split}.jsonl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_vi.jsonl"

    if not input_path.exists():
        print(f"❌ Không tìm thấy file: {input_path}")
        print("   Hãy chạy scripts/01_download_data.py trước!")
        return

    # Import translator
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("❌ Chưa cài deep-translator!")
        print("   Chạy: pip install deep-translator")
        return

    translator = GoogleTranslator(source="en", target="vi")

    # Đọc dữ liệu đầu vào
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            records.append(json.loads(line))

    total = len(records)

    # Kiểm tra tiến trình đã dịch (checkpoint)
    existing_count = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)

    if existing_count >= total:
        print(f"✅ Đã dịch hết {total} câu! Không cần dịch thêm.")
        return

    remaining = records[existing_count:]

    print("=" * 55)
    print(f"  BƯỚC 2: DỊCH CÂU HỎI SANG TIẾNG VIỆT ({args.split})")
    print("=" * 55)
    print(f"  Tổng cần dịch: {total} câu")
    if existing_count > 0:
        print(f"  Đã dịch trước đó: {existing_count} câu")
    print(f"  Còn lại: {len(remaining)} câu")
    print(f"  Thời gian ước tính: ~{len(remaining) * args.delay / 60:.0f} phút")
    print()

    translated_count = 0
    error_count = 0
    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as f:
        for i, record in enumerate(remaining):
            # Dịch câu hỏi
            try:
                question_vi = translator.translate(record["question_en"])
                if not question_vi:
                    question_vi = record["question_en"]
                    error_count += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi câu {existing_count + i + 1}: {e}")
                question_vi = record["question_en"]
                error_count += 1
                time.sleep(2)  # Đợi lâu hơn khi lỗi

            # Lưu record với câu hỏi đã dịch
            record["question_vi"] = question_vi
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            translated_count += 1

            # Log tiến trình mỗi 100 câu
            if translated_count % 100 == 0:
                elapsed = time.time() - start_time
                speed = translated_count / elapsed
                eta = (len(remaining) - translated_count) / speed if speed > 0 else 0
                print(
                    f"  ✅ {existing_count + translated_count}/{total} câu "
                    f"({error_count} lỗi) — "
                    f"ETA: {eta / 60:.1f} phút"
                )

            time.sleep(args.delay)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 55}")
    print(f"  HOÀN TẤT! Đã dịch {translated_count} câu trong {elapsed / 60:.1f} phút")
    print(f"  Số lỗi: {error_count}")
    print(f"  Output: {output_path}")
    print()
    print("  Tiếp theo:")
    if args.split == "train":
        print("    python scripts/02_translate_data.py --split dev --max-samples 500")
    else:
        print("    python scripts/03_prepare_training.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
