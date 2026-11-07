"""
Task 2 – Build the network   &   Task 3.1 – Cross-entropy loss.

ECGNet: a 3-layer fully-connected network built from raw nn.Parameter weights.
  Layer 1:  187 -> 128, ReLU
  Layer 2:  128 ->  64, ReLU
  Layer 3:   64 ->   5, logits (no activation)

Total parameters: (187*128 + 128) + (128*64 + 64) + (64*5 + 5) = 32,645

cross_entropy(logits, y): numerically stable softmax cross-entropy using logsumexp.
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    """3-layer fully-connected ECG classifier (all weights created manually)."""

    def __init__(self):
        super().__init__()
        # Layer sizes
        n = [187, 128, 64, 5]

        # He initialisation:  W ~ N(0, sqrt(2/n_in)),  b = 0
        # Create parameters manually (no nn.Linear!)
        self.W1 = nn.Parameter(torch.randn(n[1], n[0]) * (2.0 / n[0]) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n[1]))

        self.W2 = nn.Parameter(torch.randn(n[2], n[1]) * (2.0 / n[1]) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n[2]))

        self.W3 = nn.Parameter(torch.randn(n[3], n[2]) * (2.0 / n[2]) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(n[3]))

    def forward(self, x):
        """
        x: (B, 187)
        returns: logits (B, 5)

        Layer formula (batched, row-vectors):
            Z[l] = A[l-1] @ W[l].T + b[l]
            A[l] = ReLU(Z[l])   for hidden layers
        """
        # Layer 1
        z1 = x @ self.W1.t() + self.b1             # (B, 128)
        a1 = torch.clamp(z1, min=0)                 # ReLU

        # Layer 2
        z2 = a1 @ self.W2.t() + self.b2             # (B, 64)
        a2 = torch.clamp(z2, min=0)                 # ReLU

        # Layer 3 (output – logits, no activation)
        z3 = a2 @ self.W3.t() + self.b3             # (B, 5)

        # Store activations for gradient check (Task 3.2)
        self._a2 = a2

        return z3


def cross_entropy(logits, y):
    """
    Softmax cross-entropy loss (mean over batch).

    L_i = -z_{y_i} + log(sum_j exp(z_j))

    Uses torch.logsumexp for numerical stability.

    Args:
        logits: (B, 5) raw scores from the network
        y:      (B,)   integer class labels 0-4
    Returns:
        scalar mean loss
    """
    # Gather the logit of the correct class for each example
    z_y = logits[torch.arange(logits.size(0)), y]          # (B,)

    # log-sum-exp over classes
    lse = torch.logsumexp(logits, dim=1)                   # (B,)

    # Per-sample loss and mean
    loss = (-z_y + lse).mean()
    return loss


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(0)
    model = ECGNet()

    # 2.4  Total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")
    print("  Expected: 32,645")
    print(f"  Calculation: (187*128+128) + (128*64+64) + (64*5+5) "
          f"= {187*128+128} + {128*64+64} + {64*5+5} = {187*128+128+128*64+64+64*5+5}")

    # Quick forward pass test
    x_dummy = torch.randn(4, 187)
    logits = model(x_dummy)
    print(f"\nForward pass test: input {x_dummy.shape} -> logits {logits.shape}")

    # Loss on random data (should be close to log(5) ≈ 1.61)
    y_dummy = torch.randint(0, 5, (4,))
    loss = cross_entropy(logits, y_dummy)
    print(f"Loss on random data: {loss.item():.4f}  (expected ~{torch.log(torch.tensor(5.0)).item():.4f})")
