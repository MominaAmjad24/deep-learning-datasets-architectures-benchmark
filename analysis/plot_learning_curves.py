# analysis/plot_learning_curves.py
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    base = Path("runs")
    # Example comparisons (edit as you like)
    experiments = [
        base / "cifar100_mlp" / "result.json",
        base / "cifar100_cnn" / "result.json",
        base / "cifar100_vit" / "result.json",
    ]

    results = [load_result(p) for p in experiments if p.exists()]
    if not results:
        raise FileNotFoundError("No result.json files found. Run run_experiment.py first.")

    # Plot validation accuracy
    plt.figure()
    for r in results:
        y = r["history"]["val_acc"]
        label = f"{r['dataset']}-{r['architecture']}"
        plt.plot(range(1, len(y) + 1), y, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Curves: Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    out = base / "learning_curves_val_acc.png"
    plt.savefig(out, dpi=200)
    plt.show()

    # Plot train loss
    plt.figure()
    for r in results:
        y = r["history"]["train_loss"]
        label = f"{r['dataset']}-{r['architecture']}"
        plt.plot(range(1, len(y) + 1), y, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Learning Curves: Training Loss")
    plt.legend()
    plt.tight_layout()
    out2 = base / "learning_curves_train_loss.png"
    plt.savefig(out2, dpi=200)
    plt.show()

    print(f"Saved: {out}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

