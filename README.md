# Vietnamese Text-to-SQL Fine-Tuning

Fine-tune **Gemma-2-2B-IT** (QLoRA 4-bit) để chuyển câu hỏi tiếng Việt thành truy vấn SQL.

## Ví dụ

```
👤 Input:  "Có bao nhiêu động cơ có công suất 1000kw?"
🤖 Output: SELECT COUNT(Engine) FROM table WHERE Power = '1000kw'
```

## Công nghệ

| Thành phần | Chi tiết |
|---|---|
| Base Model | `google/gemma-2-2b-it` (quantized 4-bit) |
| Fine-tuning | QLoRA (r=16) via Unsloth + SFTTrainer |
| Dataset | `gretelai/synthetic_text_to_sql` (~5k câu, dịch EN→VN) |
| Hardware | NVIDIA RTX 3060 (12GB VRAM) |
| Metrics | Syntax Accuracy, Execution Accuracy, Exact Match |

## Cấu trúc thư mục

```
├── data/
│   ├── raw/              # GretelAI gốc (auto-download)
│   ├── translated/       # Câu hỏi đã dịch sang tiếng Việt
│   └── processed/        # Data huấn luyện (train.jsonl, eval.jsonl)
├── scripts/
│   ├── 01_download_data.py       # Tải WikiSQL
│   ├── 02_translate_data.py      # Dịch EN → VN (Google Translate)
│   ├── 03_prepare_training.py    # Format dữ liệu cho SFT
│   ├── 04_train.py               # Fine-tune model
│   ├── 05_evaluate.py            # Đánh giá (Syntax + Execution Accuracy)
│   └── 06_plot_results.py        # Vẽ biểu đồ
├── models/               # Checkpoints và LoRA adapter
└── reports/              # Kết quả evaluation và biểu đồ
```

## Hướng dẫn chạy

### 1. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Pipeline (chạy theo thứ tự)

```bash
# Bước 1: Tải GretelAI Synthetic Text-to-SQL (~35MB)
python scripts/01_download_data.py

# Bước 2: Dịch câu hỏi sang tiếng Việt
python scripts/02_translate_data.py --split train --max-samples 5000   # Train + Val (~20 phút)
python scripts/02_translate_data.py --split test --max-samples 500    # Test (1 lần cuối)

# Bước 3: Chuẩn bị dữ liệu (script sẽ tự động tách 500 câu từ train làm tập validation)
python scripts/03_prepare_training.py

# Bước 4: Fine-tune (~2-3 giờ trên RTX 3060)
python scripts/04_train.py

# Bước 5: Đánh giá tất cả checkpoints (trên val set)
python scripts/05_evaluate.py --eval-all

# Bước 6: Đánh giá final model trên test set (chạy 1 lần duy nhất)
python scripts/05_evaluate.py --final-test

# Bước 7: Vẽ biểu đồ
python scripts/06_plot_results.py
```
