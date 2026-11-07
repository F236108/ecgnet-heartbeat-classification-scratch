"""
Task 5 – Test once.

5.1  Load the saved parameters and predict the class of every test row.
5.2  Compute accuracy, confusion matrix, per-class accuracy (from metrics.py).
5.3  Report results and save confusion matrix heatmap.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import ECGNet
from prepare import prepare_data
from metrics import accuracy, confusion_matrix, per_class_accuracy, balanced_accuracy

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def test_model(data):
    """Load best model, predict on test set, report all metrics."""
    # 5.1  Load saved parameters
    model = ECGNet()
    state = torch.load("results/ecgnet_best.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    X_test = data["X_test"]
    y_test = data["y_test"]

    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)

    y_true = y_test.tolist()
    y_pred = preds.tolist()

    # 5.2  Metrics
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes=5)
    pca = per_class_accuracy(cm)
    bal_acc = balanced_accuracy(cm)

    # 5.3  Report
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nTest accuracy:     {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Balanced accuracy: {bal_acc:.4f}  ({bal_acc*100:.2f}%)")

    print(f"\nPer-class accuracy:")
    for c in range(5):
        print(f"  {CLASS_NAMES[c]}: {pca[c]:.4f}  ({pca[c]*100:.2f}%)")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    header = "       " + "".join(f"{CLASS_NAMES[c]:>7}" for c in range(5)) + "    Row Sum"
    print(header)
    for j in range(5):
        row_str = f"  {CLASS_NAMES[j]:>3}  " + "".join(f"{cm[j][k]:>7}" for k in range(5))
        row_sum = sum(cm[j])
        row_str += f"    {row_sum:>7}"
        print(row_str)
    total = sum(sum(row) for row in cm)
    print(f"\n  Total entries: {total}  (expected 21,892)")

    # Task 6.1  Always-"N" accuracy
    always_n_acc = sum(1 for y in y_true if y == 0) / len(y_true)
    print(f"\n  Always-'N' accuracy: {always_n_acc:.4f}  ({always_n_acc*100:.2f}%)")

    # ── Confusion matrix heatmap ──
    cm_arr = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Test Confusion Matrix")
    # Annotate cells
    for i in range(5):
        for j in range(5):
            val = int(cm_arr[i, j])
            color = "white" if cm_arr[i, j] > cm_arr.max() * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    print("\nSaved results/confusion_matrix.png")

    # Save results to a text file
    with open("results/test_results.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
        f.write(f"Always-N accuracy: {always_n_acc:.4f}\n")
        for c in range(5):
            f.write(f"Per-class accuracy {CLASS_NAMES[c]}: {pca[c]:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        for j in range(5):
            f.write(",".join(str(cm[j][k]) for k in range(5)) + "\n")
    print("Saved results/test_results.txt")

    return acc, bal_acc, cm, pca, always_n_acc


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_data()
    test_model(data)
