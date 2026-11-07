# Experimental Results & Tables

## 4.1 Table 1 – Data

| Quantity / Split | Value |
|:---|:---:|
| Training rows (after 90/10 split) | 78,799 |
| Validation rows (after 90/10 split) | 8,755 |
| Test rows (separate test file) | 21,892 |
| Minimum signal value in train file | 0.000000 |
| Maximum signal value in train file | 1.000000 |
| **Rows per class, train file (87,554 total)** | |
| N – Normal | 72,471 |
| S – Supraventricular ectopic | 2,223 |
| V – Ventricular ectopic | 5,788 |
| F – Fusion | 641 |
| Q – Unknown / Paced | 6,431 |
| **Rows per class, test file (21,892 total)** | |
| N – Normal | 18,118 |
| S – Supraventricular ectopic | 556 |
| V – Ventricular ectopic | 1,448 |
| F – Fusion | 162 |
| Q – Unknown / Paced | 1,608 |

---

## 4.2 Table 2 – Network and Training

| Specification / Metric | Value |
|:---|:---:|
| Total trainable parameters | 32,645 |
| Initial loss (before training) | 1.741343 |
| Gradient check: max absolute difference for $W^{[3]}$ | $2.98 \times 10^{-8}$ |
| Gradient check: max absolute difference for $b^{[3]}$ | $5.96 \times 10^{-8}$ |
| Chosen learning rate ($\alpha$) | 0.50 |
| Best epoch (lowest validation loss) | 23 |
| Training loss at best epoch | 0.060877 |
| Validation loss at best epoch | 0.072164 |
| Training accuracy at best epoch | 98.16% (0.9816) |
| Validation accuracy at best epoch | 97.93% (0.9793) |
| Training time & Hardware | 43.9 seconds (CPU) |

---

## 4.3 Table 3 – Learning-Rate Choice

| Learning Rate ($\alpha$) | Lowest Validation Loss | Validation Accuracy at that Epoch | Best Epoch |
|:---:|:---:|:---:|:---:|
| 0.01 | 0.161820 | 95.79% (0.9579) | 30 |
| 0.10 | 0.086183 | 97.64% (0.9764) | 30 |
| **0.50 (Chosen)** | **0.072164** | **97.93% (0.9793)** | **23** |

---

## 4.4 Table 4 – Test Results

| Metric / Class | N | S | V | F | Q | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Per-Class Accuracy** | **99.60%** (18,045 / 18,118) | **66.37%** (369 / 556) | **91.37%** (1,323 / 1,448) | **62.96%** (102 / 162) | **97.95%** (1,575 / 1,608) | — |
| **Overall Test Accuracy** | — | — | — | — | — | **97.82%** (21,414 / 21,892) |
| **Balanced Accuracy** | — | — | — | — | — | **83.65%** |
| **Always-"N" Baseline Accuracy** | — | — | — | — | — | **82.76%** (18,118 / 21,892) |

### Full $5 \times 5$ Test Confusion Matrix

| True \ Predicted | N (Predicted) | S (Predicted) | V (Predicted) | F (Predicted) | Q (Predicted) | Row Sum (True Count) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **N (Normal)** | **18,045** | 21 | 18 | 11 | 23 | **18,118** |
| **S (Supraventricular)** | 181 | **369** | 4 | 0 | 2 | **556** |
| **V (Ventricular)** | 100 | 4 | **1,323** | 15 | 6 | **1,448** |
| **F (Fusion)** | 39 | 0 | 21 | **102** | 0 | **162** |
| **Q (Unknown / Paced)** | 27 | 1 | 5 | 0 | **1,575** | **1,608** |
| **Column Sum** | 18,392 | 395 | 1,371 | 128 | 1,606 | **Total: 21,892** |
