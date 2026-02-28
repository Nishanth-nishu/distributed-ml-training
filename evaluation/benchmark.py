"""
Comprehensive model evaluation and benchmark report generator.
Compares multiple training runs and produces a detailed markdown report.
"""

import sys
import json
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PRIMARY = "#4285F4"
COLORS = ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#9B59B6"]


def load_all_runs() -> list:
    """Load all experiment result JSON files."""
    results = []
    for path in glob.glob(str(CHECKPOINT_DIR / "*_results.json")):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = path
        results.append(data)
    return sorted(results, key=lambda r: r.get("best_val_auc", 0), reverse=True)


def plot_comparison(runs: list) -> plt.Figure:
    if not runs:
        return None

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.patch.set_facecolor("white")
    fig.suptitle("Distributed ML Training — Experiment Comparison", fontsize=16, fontweight="bold")

    names = [r.get("run_name", f"run_{i}") for i, r in enumerate(runs)][:5]
    aucs = [r.get("best_val_auc", 0) for r in runs[:5]]
    test_metrics = [r.get("test_metrics", {}) for r in runs[:5]]

    # 1. Val AUC comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(names[::-1], aucs[::-1], color=COLORS[:len(names)], alpha=0.85)
    ax1.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax1.set_title("Best Validation AUC by Run", fontweight="bold")
    ax1.set_xlabel("AUC")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="x", alpha=0.3)
    ax1.legend()

    for bar, val in zip(bars, aucs[::-1]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

    # 2. Training curves (best run)
    ax2 = fig.add_subplot(gs[0, 1])
    best_run = runs[0]
    h = best_run.get("history", {})
    if h.get("val_auc"):
        epochs = range(1, len(h["val_auc"]) + 1)
        ax2.plot(epochs, h["val_auc"], color=PRIMARY, label="Val AUC", lw=2)
        ax2.plot(epochs, h.get("val_f1", [0]*len(epochs)), color="#34A853", label="Val F1", lw=2)
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title(f"Training Curves — {best_run.get('run_name', 'best')}", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(alpha=0.3)

    # 3. Loss curves
    ax3 = fig.add_subplot(gs[1, 0])
    if h.get("train_loss"):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax3.plot(epochs, h["train_loss"], color=PRIMARY, label="Train Loss", lw=2)
        ax3.plot(epochs, h["val_loss"], color="#EA4335", label="Val Loss", lw=2, ls="--")
        ax3.set_title("Loss Curves — Best Run", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("BCE Loss")
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 4. Test metrics heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_labels = ["auc", "f1", "accuracy"]
    matrix = []
    for r in runs[:5]:
        tm = r.get("test_metrics", {})
        matrix.append([tm.get(m, 0) for m in metrics_labels])

    if matrix:
        im = ax4.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax4.set_xticks(range(len(metrics_labels)))
        ax4.set_xticklabels(["AUC", "F1", "Accuracy"])
        ax4.set_yticks(range(len(names[:5])))
        ax4.set_yticklabels(names[:5], fontsize=8)
        ax4.set_title("Test Metrics Heatmap", fontweight="bold")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        for i in range(len(matrix)):
            for j in range(len(metrics_labels)):
                ax4.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center",
                         fontsize=9, color="black")

    return fig


def generate_report(runs: list) -> str:
    """Generate a markdown benchmark report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Distributed ML Training — Benchmark Report",
        f"\nGenerated: {timestamp}\n",
        "## Summary\n",
    ]

    if not runs:
        lines.append("No experiment results found. Run experiments first.\n")
        return "\n".join(lines)

    lines.append(f"Total experiments: **{len(runs)}**\n")
    best = runs[0]
    lines.append(f"Best run: **{best.get('run_name', 'N/A')}** — Val AUC: `{best.get('best_val_auc', 0):.4f}`\n")

    lines.append("## Results Table\n")
    lines.append("| Run | World Size | Val AUC | Test AUC | Test F1 | Test Acc |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        tm = r.get("test_metrics", {})
        lines.append(
            f"| {r.get('run_name', 'N/A')} "
            f"| {r.get('world_size', 1)} "
            f"| {r.get('best_val_auc', 0):.4f} "
            f"| {tm.get('auc', 0):.4f} "
            f"| {tm.get('f1', 0):.4f} "
            f"| {tm.get('accuracy', 0):.4f} |"
        )

    lines.append("\n## Best Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(best.get("config", {}), indent=2))
    lines.append("```\n")

    # Save plot
    fig = plot_comparison(runs)
    if fig:
        plot_path = REPORTS_DIR / "comparison_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        lines.append(f"\n![Comparison Plot](comparison_plot.png)\n")

    return "\n".join(lines)


def main():
    logger.info("Loading experiment results...")
    runs = load_all_runs()

    if not runs:
        logger.warning("No results found in checkpoints/. Run experiments first.")
        return

    logger.info(f"Found {len(runs)} experiments.")
    report = generate_report(runs)

    report_path = REPORTS_DIR / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")
    print(f"\n{'='*60}")
    print(f"Best run: {runs[0].get('run_name')} | AUC: {runs[0].get('best_val_auc', 0):.4f}")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
