"""
Task 4 – Train.

Mini-batch SGD with momentum and L2 weight decay, implemented manually.

Update rule (for each parameter P, velocity v_P starting at 0):
    g = dJ/dP + lambda * P   (weights only; biases: g = dJ/dP)
    v_P = beta * v_P + (1 - beta) * g
    P   = P - alpha * v_P

Settings:
    Batch size B   = 128
    Momentum beta  = 0.9
    Weight decay   = 1e-4
    Epochs         = 30
    alpha          in {0.01, 0.1, 0.5}
    Shuffle        every epoch
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ECGNet, cross_entropy
from prepare import prepare_data


# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 128
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS       = 30
ALPHAS       = [0.01, 0.1, 0.5]


def _is_weight(name):
    """Return True for weight parameters (apply weight decay), False for biases."""
    return name.startswith("W")


def evaluate(model, X, y):
    """Return (loss, accuracy) on the given data (no gradient)."""
    with torch.no_grad():
        logits = model(X)
        loss = cross_entropy(logits, y).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc


def train(alpha, data, verbose=True):
    """Train a fresh ECGNet (seed 0) for EPOCHS epochs at learning rate alpha.

    Returns: model, history dict, best_state_dict, best_epoch
    """
    torch.manual_seed(0)
    model = ECGNet()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]

    n_train = X_train.size(0)

    # Initialise velocities to zero
    velocities = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 1

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Shuffle training data each epoch
        perm = torch.randperm(n_train)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            # Forward
            logits = model(xb)
            loss = cross_entropy(logits, yb)

            # Backward
            model.zero_grad()
            loss.backward()

            # Manual SGD with momentum + weight decay (inside no_grad)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    grad = p.grad
                    if _is_weight(name):
                        grad = grad + WEIGHT_DECAY * p          # L2 penalty on weights only
                    v = velocities[name]
                    v.mul_(MOMENTUM).add_((1 - MOMENTUM) * grad)  # v = beta*v + (1-beta)*g
                    p.sub_(alpha * v)                              # P = P - alpha * v

            epoch_loss += loss.item() * (end - start)
            epoch_correct += (logits.argmax(1) == yb).sum().item()

        train_loss = epoch_loss / n_train
        train_acc  = epoch_correct / n_train
        val_loss, val_acc = evaluate(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if verbose:
            print(f"  Epoch {epoch:2d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training time: {elapsed:.1f}s")
    history["time"] = elapsed

    return model, history, best_state_dict, best_epoch


def run_training_experiment(data):
    """Train for each alpha in ALPHAS, record metrics, save plots and best model."""
    results = {}
    for alpha in ALPHAS:
        print(f"\n==========================================")
        print(f"Training with alpha = {alpha}")
        print(f"==========================================")
        model, hist, best_state, best_ep = train(alpha, data)
        results[alpha] = {
            "model": model,
            "history": hist,
            "best_state": best_state,
            "best_epoch": best_ep
        }

    # Task 4.2 – Plot validation loss vs epoch for all three alphas
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in ALPHAS:
        ax.plot(range(1, EPOCHS + 1), results[alpha]["history"]["val_loss"],
                label=f"α = {alpha}", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title("Figure 2: Validation Loss vs Epoch for Different Learning Rates", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/lr_comparison.png", dpi=150)
    print("\nSaved results/lr_comparison.png")

    # Table 3 – Learning-rate choice
    print("\n" + "=" * 60)
    print("Table 3 – Learning-rate choice:")
    print("=" * 60)
    print(f"{'Alpha':>8}  {'Lowest Val Loss':>16}  {'Val Acc at that epoch':>22}")
    for alpha in ALPHAS:
        vl = results[alpha]["history"]["val_loss"]
        va = results[alpha]["history"]["val_acc"]
        best_ep = int(np.argmin(vl))
        print(f"{alpha:>8.2f}  {vl[best_ep]:>16.6f}  {va[best_ep]:>22.4f}")

    # Choose the alpha with lowest validation loss
    best_alpha = min(ALPHAS, key=lambda a: min(results[a]["history"]["val_loss"]))
    best_res = results[best_alpha]
    best_epoch = best_res["best_epoch"]
    hist = best_res["history"]

    print("\n" + "=" * 60)
    print(f"Chosen alpha: {best_alpha}")
    print(f"Best epoch (lowest val loss): {best_epoch}")
    print(f"  Training loss at best epoch:   {hist['train_loss'][best_epoch-1]:.6f}")
    print(f"  Validation loss at best epoch: {hist['val_loss'][best_epoch-1]:.6f}")
    print(f"  Training acc at best epoch:    {hist['train_acc'][best_epoch-1]:.4f}")
    print(f"  Validation acc at best epoch:  {hist['val_acc'][best_epoch-1]:.4f}")
    print(f"  Training time:                 {hist['time']:.1f}s")
    print("=" * 60)

    # Save model from best epoch
    torch.save(best_res["best_state"], "results/ecgnet_best.pt")
    print(f"\nSaved results/ecgnet_best.pt (from epoch {best_epoch} with alpha={best_alpha})")

    # Task 4.3 – Plot training & validation loss and accuracy for chosen alpha
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs_range, hist["train_loss"], label="Train Loss", color="royalblue", linewidth=1.5)
    ax1.plot(epochs_range, hist["val_loss"],   label="Validation Loss", color="crimson", linewidth=1.5)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(f"Figure 3a: Loss Curves (α = {best_alpha})", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs_range, hist["train_acc"], label="Train Accuracy", color="royalblue", linewidth=1.5)
    ax2.plot(epochs_range, hist["val_acc"],   label="Validation Accuracy", color="crimson", linewidth=1.5)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"Figure 3b: Accuracy Curves (α = {best_alpha})", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/training_curves.png", dpi=150)
    print("Saved results/training_curves.png")

    # Save training summary
    with open("results/training_summary.txt", "w") as f:
        f.write(f"Chosen alpha: {best_alpha}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Train loss at best epoch: {hist['train_loss'][best_epoch-1]:.6f}\n")
        f.write(f"Val loss at best epoch: {hist['val_loss'][best_epoch-1]:.6f}\n")
        f.write(f"Train acc at best epoch: {hist['train_acc'][best_epoch-1]:.4f}\n")
        f.write(f"Val acc at best epoch: {hist['val_acc'][best_epoch-1]:.4f}\n")
        f.write(f"Training time (s): {hist['time']:.1f}\n")
        for a in ALPHAS:
            v_loss = results[a]["history"]["val_loss"]
            v_acc  = results[a]["history"]["val_acc"]
            b_ep   = int(np.argmin(v_loss))
            f.write(f"alpha_{a}_min_val_loss: {v_loss[b_ep]:.6f}\n")
            f.write(f"alpha_{a}_val_acc: {v_acc[b_ep]:.4f}\n")

    return best_alpha, results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_data()
    best_alpha, results = run_training_experiment(data)
