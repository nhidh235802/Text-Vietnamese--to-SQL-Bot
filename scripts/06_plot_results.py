"""
Script 06: Vẽ biểu đồ kết quả training và evaluation.

Biểu đồ:
  1. Training Loss curve
  2. Learning Curve (Accuracy per checkpoint)
  3. Before/After comparison (bar chart)

Chạy:
    python scripts/06_plot_results.py
    python scripts/06_plot_results.py --trainer-state models/trainer_state.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_loss(trainer_state_path, output_path):
    """Vẽ biểu đồ Training Loss từ trainer_state.json."""
    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    steps_train, loss_train = [], []
    steps_eval, loss_eval = [], []

    for log in log_history:
        step = log.get("step", 0)
        if "loss" in log:
            steps_train.append(step)
            loss_train.append(log["loss"])
        elif "eval_loss" in log:
            steps_eval.append(step)
            loss_eval.append(log["eval_loss"])

    if not steps_train:
        print("  ⚠️ Không tìm thấy dữ liệu training loss!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps_train, loss_train, label="Training Loss", color="#4F46E5", alpha=0.7, linewidth=1.5)
    if steps_eval:
        ax.plot(steps_eval, loss_eval, label="Eval Loss", color="#EF4444",
                marker="o", markersize=5, linewidth=2)

    ax.set_title("Training & Evaluation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Training Loss → {output_path}")
    plt.close()


def plot_learning_curve(summary_path, output_path):
    """Vẽ Learning Curve (accuracy tại mỗi checkpoint) từ eval_summary.json."""
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Sắp xếp theo thứ tự checkpoint
    ordered = []
    for name, metrics in summary.items():
        if name == "base":
            step = 0
        elif name == "final":
            step = float("inf")
        elif name.startswith("checkpoint-"):
            step = int(name.split("-")[1])
        else:
            continue
        ordered.append((step, name, metrics))

    ordered.sort(key=lambda x: x[0])

    labels = [item[1] for item in ordered]
    syntax_acc = [item[2]["syntax_accuracy"] for item in ordered]
    exec_acc = [item[2]["execution_accuracy"] for item in ordered]
    exact_acc = [item[2]["exact_match_accuracy"] for item in ordered]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(labels))
    ax.plot(x, exec_acc, label="Execution Accuracy", color="#10B981",
            marker="o", markersize=8, linewidth=2.5, zorder=3)
    ax.plot(x, syntax_acc, label="Syntax Accuracy", color="#6366F1",
            marker="s", markersize=6, linewidth=2, alpha=0.7)
    ax.plot(x, exact_acc, label="Exact Match", color="#F59E0B",
            marker="^", markersize=6, linewidth=2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Learning Curve — Accuracy Per Checkpoint", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Annotate execution accuracy values
    for i, val in enumerate(exec_acc):
        ax.annotate(f"{val:.1f}%", (i, val), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold",
                    color="#10B981")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Learning Curve → {output_path}")
    plt.close()


def plot_before_after(summary_path, output_path):
    """Vẽ bar chart so sánh baseline vs fine-tuned model."""
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Tìm baseline và model tốt nhất
    base_metrics = summary.get("base", {})
    best_name = "final"
    best_metrics = summary.get("final", {})

    # Nếu không có final, lấy checkpoint cuối cùng
    if not best_metrics:
        checkpoints = {k: v for k, v in summary.items() if k.startswith("checkpoint-")}
        if checkpoints:
            best_name = max(checkpoints.keys(), key=lambda k: int(k.split("-")[1]))
            best_metrics = checkpoints[best_name]

    if not base_metrics or not best_metrics:
        print("  ⚠️ Cần có cả baseline và fine-tuned results!")
        return

    metrics = ["syntax_accuracy", "execution_accuracy", "exact_match_accuracy"]
    labels = ["Syntax\nAccuracy", "Execution\nAccuracy", "Exact\nMatch"]

    base_vals = [base_metrics.get(m, 0) for m in metrics]
    best_vals = [best_metrics.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(len(labels))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], base_vals, width,
                   label="Baseline (chưa fine-tune)", color="#94A3B8", edgecolor="white")
    bars2 = ax.bar([i + width / 2 for i in x], best_vals, width,
                   label=f"Fine-tuned ({best_name})", color="#10B981", edgecolor="white")

    # Annotate bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color="#64748B")

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color="#059669")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Baseline vs Fine-tuned — Text-to-SQL Accuracy",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Before/After → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ kết quả")
    parser.add_argument("--trainer-state", default=None,
                        help="Đường dẫn tới trainer_state.json (tự tìm nếu không chỉ định)")
    parser.add_argument("--summary", default="reports/eval_summary.json",
                        help="Đường dẫn tới eval_summary.json")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  VẼ BIỂU ĐỒ KẾT QUẢ")
    print("=" * 55)

    # 1. Training Loss
    trainer_state = args.trainer_state
    if trainer_state is None:
        # Tự tìm trainer_state.json trong models/
        candidates = list(Path("models").rglob("trainer_state.json"))
        if candidates:
            trainer_state = str(candidates[0])

    if trainer_state and Path(trainer_state).exists():
        plot_training_loss(trainer_state, output_dir / "training_loss.png")
    else:
        print("  ⚠️ Không tìm thấy trainer_state.json, bỏ qua biểu đồ Training Loss")

    # 2. Learning Curve
    summary_path = Path(args.summary)
    if summary_path.exists():
        plot_learning_curve(summary_path, output_dir / "learning_curve.png")
        plot_before_after(summary_path, output_dir / "before_after.png")
    else:
        print(f"  ⚠️ Không tìm thấy {summary_path}")
        print("     Hãy chạy: python scripts/05_evaluate.py --eval-all")

    print(f"\n{'=' * 55}")
    print("  ✅ HOÀN TẤT! Biểu đồ đã lưu trong reports/")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
