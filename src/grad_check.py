"""
Task 3.2 – Gradient check.

Takes the first 64 training rows, runs a forward pass, computes the loss,
calls loss.backward(), then computes the output-layer gradient by hand:

    dJ/dW3 = (1/B) * (Y_hat - Y).T @ A2
    dJ/db3 = (1/B) * sum_i (y_hat_i - y_i)

and reports the maximum absolute difference from PyTorch's .grad values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import ECGNet, cross_entropy
from prepare import prepare_data


def gradient_check(data):
    """Perform gradient check on the first 64 training rows."""
    torch.manual_seed(0)
    model = ECGNet()

    B = 64
    X = data["X_train"][:B]
    y = data["y_train"][:B]

    # Forward pass
    logits = model(X)                        # (B, 5)
    loss = cross_entropy(logits, y)

    print(f"Initial loss: {loss.item():.6f}  (expected ~{torch.log(torch.tensor(5.0)).item():.4f})")

    # Backward pass (PyTorch autograd)
    loss.backward()

    # ---- Hand-computed gradients ----
    # Softmax probabilities  Y_hat  (B, 5)
    Y_hat = torch.softmax(logits.detach(), dim=1)

    # One-hot labels  Y  (B, 5)
    Y = torch.zeros_like(Y_hat)
    Y[torch.arange(B), y] = 1.0

    # Second hidden layer activations  A2  (B, 64)
    A2 = model._a2.detach()

    # dJ/dW3 = (1/B) * (Y_hat - Y)^T @ A2     shape (5, 64)
    dW3_hand = (1.0 / B) * (Y_hat - Y).t() @ A2

    # dJ/db3 = (1/B) * sum_i (y_hat_i - y_i)   shape (5,)
    db3_hand = (1.0 / B) * (Y_hat - Y).sum(dim=0)

    # Compare with PyTorch gradients
    dW3_auto = model.W3.grad
    db3_auto = model.b3.grad

    max_diff_W3 = (dW3_hand - dW3_auto).abs().max().item()
    max_diff_b3 = (db3_hand - db3_auto).abs().max().item()

    print(f"\nGradient check (first {B} training rows):")
    print(f"  Max |dJ/dW3_hand - dJ/dW3_auto| = {max_diff_W3:.2e}")
    print(f"  Max |dJ/db3_hand - dJ/db3_auto| = {max_diff_b3:.2e}")
    print(f"  Threshold: 1e-5")
    print(f"  W3 PASS: {max_diff_W3 < 1e-5}")
    print(f"  b3 PASS: {max_diff_b3 < 1e-5}")

    return max_diff_W3, max_diff_b3


if __name__ == "__main__":
    data = prepare_data()
    gradient_check(data)
