"""
Task 1 – Load and prepare the data.

Steps:
  1.1  Load the two CSV files and print shapes.
  1.2  Separate signal (cols 0-186) from label (col 187); convert labels to int.
  1.3  Report min/max signal value in the training file.
  1.4  Report rows per class in both files.
  1.5  Run plot_beats.py (done separately).
  1.6  Split training file into train/validation (seed 42, 10 % validation).
  1.7  Convert to torch tensors.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from load_data import load_raw

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


def prepare_data(data_dir="data"):
    """Load, split and convert the MIT-BIH data.  Returns a dict of tensors."""

    # 1.1  Load raw data
    raw = load_raw(data_dir)
    print("=== Task 1.1: Shapes ===")
    for split, df in raw.items():
        print(f"  {split}: {df.shape[0]} rows, {df.shape[1]} columns")

    train_df = raw["train"]
    test_df  = raw["test"]

    # 1.2  Separate signal from label
    X_train_full = train_df.iloc[:, :187].values
    y_train_full = train_df.iloc[:, 187].values.astype(int)
    X_test  = test_df.iloc[:, :187].values
    y_test  = test_df.iloc[:, 187].values.astype(int)

    print(f"\n=== Task 1.2: Label dtype after conversion ===")
    print(f"  y_train dtype: {y_train_full.dtype}")

    # 1.3  Min / max signal value in training file
    sig_min = X_train_full.min()
    sig_max = X_train_full.max()
    print(f"\n=== Task 1.3: Signal range (train) ===")
    print(f"  min = {sig_min:.6f},  max = {sig_max:.6f}")

    # 1.4  Rows per class
    print(f"\n=== Task 1.4: Rows per class ===")
    print("  Train file:")
    for c in range(5):
        n = (y_train_full == c).sum()
        print(f"    {CLASS_NAMES[c]}: {n}")
    print("  Test file:")
    for c in range(5):
        n = (y_test == c).sum()
        print(f"    {CLASS_NAMES[c]}: {n}")

    # 1.6  Split training file into train / validation  (seed 42, 10 % val)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val   = X_train_full[val_idx]
    y_val   = y_train_full[val_idx]

    print(f"\n=== Task 1.6: Split sizes ===")
    print(f"  Train:      {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test:       {len(X_test)}")

    # 1.7  Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    print(f"\n=== Task 1.7: Tensor dtypes ===")
    print(f"  X_train: {X_train_t.dtype},  y_train: {y_train_t.dtype}")

    return {
        "X_train": X_train_t, "y_train": y_train_t,
        "X_val":   X_val_t,   "y_val":   y_val_t,
        "X_test":  X_test_t,  "y_test":  y_test_t,
        "sig_min": sig_min,   "sig_max": sig_max,
    }


if __name__ == "__main__":
    data = prepare_data()
