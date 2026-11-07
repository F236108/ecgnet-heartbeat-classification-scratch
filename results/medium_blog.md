# Classifying Heartbeats from ECG Signals: Building a Neural Network from Scratch in PyTorch

*How well does a small neural network recognise different cardiac arrhythmias, and can we trust overall accuracy when healthy beats dominate the dataset?*

---

## 1. Introduction & The Clinical Problem

The human heart is an intricate electromechanical pump governed by rhythmic electrical impulses. An electrocardiogram (ECG) measures these millivolt-level electrical potentials across the body surface over time. In a healthy heart, each beat exhibits a canonical waveform comprised of a P wave (atrial depolarisation), a QRS complex (ventricular depolarisation), and a T wave (ventricular repolarisation).

However, structural heart disease, ischemia, or conduction abnormalities produce cardiac arrhythmias—irregular heartbeats that disrupt cardiac output. Recognising these aberrant beats in real time is a critical task for bedside intensive care unit (ICU) telemetry, wearable monitors, and automated external defibrillators (AEDs).

In this project, we address a foundational engineering and diagnostic question:

> **How well does a small neural network recognise each type of heartbeat, and is overall accuracy the right number to trust when one class is far more common than the others?**

To answer this, we design, implement, train, and evaluate a fully connected deep neural network completely from scratch in PyTorch—without using high-level abstractions like `torch.nn.Linear`, `torch.optim`, or ready-made cross-entropy loss functions.

---

## 2. Dataset & Signal Characteristics

We utilise the **ECG Heartbeat Categorization Dataset**, derived from the gold-standard **MIT-BIH Arrhythmia Database** (PhysioNet). Individual heartbeats were segmented, resampled to a uniform sampling frequency of $f_s = 125\text{ Hz}$, padded with trailing zeros to a fixed length of 187 samples (spanning approximately 1.5 seconds), and normalised in amplitude between 0 and 1.

The dataset categorises heartbeats into five Association for the Advancement of Medical Instrumentation (AAMI) standard classes:

| Class Code | Class Name | Description | Train Count | Test Count |
|:---:|:---|:---|:---:|:---:|
| **N** | Normal beat | Healthy sinoatrial rhythm, left/right bundle branch blocks | 72,471 | 18,118 |
| **S** | Supraventricular ectopic | Premature atrial or junctional contraction | 2,223 | 556 |
| **V** | Ventricular ectopic | Premature ventricular contraction (PVC) | 5,788 | 1,448 |
| **F** | Fusion beat | Combination of ventricular and normal activation | 641 | 162 |
| **Q** | Unknown / Paced | Artificial pacemaker or unclassifiable waveform | 6,431 | 1,608 |
| **Total** | | | **87,554** | **21,892** |

### Morphological Differences Across Heartbeat Classes

Examining individual beats reveals stark morphological contrasts:
- **N (Normal):** Displays a crisp, narrow QRS complex preceded by a gentle P wave and followed by a smooth T wave.
- **S (Supraventricular Ectopic):** Originates above the ventricles (in atria or AV node). The ventricular conduction pathway is largely preserved, so the QRS complex remains relatively narrow and sharp, though the preceding P wave is distorted or premature. This close morphological similarity makes distinguishing S from N notoriously difficult.
- **V (Ventricular Ectopic):** Originates directly within ventricular myocardium. Conduction spreads through slower muscle-to-muscle pathways rather than the fast His-Purkinje system, resulting in a characteristically wide, bizarre, inverted or biphasic QRS complex with high amplitude.
- **F (Fusion):** Occurs when a supranodal pacemaker and an ectopic ventricular pacemaker simultaneously activate the ventricles. The resulting morphology is an intermediate hybrid between a normal beat and a PVC.
- **Q (Unknown / Paced):** Often displays prominent, sharp artificial pacemaker spikes followed by wide, delayed ventricular depolarisation.

---

## 3. Network Architecture from First Principles

To adhere strictly to foundational deep learning principles, we built our neural network (`ECGNet`) directly from fundamental linear algebra operations without any high-level framework wrappers.

### Architecture Specification
- **Input Dimension:** $n^{[0]} = 187$
- **Hidden Layer 1:** $n^{[1]} = 128$ units with custom Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$ implemented via `torch.clamp(z, min=0)`)
- **Hidden Layer 2:** $n^{[2]} = 64$ units with custom ReLU activation
- **Output Layer:** $n^{[3]} = 5$ units returning raw unnormalised logits

$$\begin{aligned}
Z^{[1]} &= X W^{[1]\top} + b^{[1]}, \quad A^{[1]} = \text{clamp}(Z^{[1]}, \min=0) \\
Z^{[2]} &= A^{[1]} W^{[2]\top} + b^{[2]}, \quad A^{[2]} = \text{clamp}(Z^{[2]}, \min=0) \\
Z^{[3]} &= A^{[2]} W^{[3]\top} + b^{[3]}
\end{aligned}$$

### Parameter Count Derivation

We instantiate every weight matrix $W^{[\ell]}$ and bias vector $b^{[\ell]}$ explicitly as an `nn.Parameter`:
- **Layer 1:** $W^{[1]} \in \mathbb{R}^{128 \times 187} \implies 23,936$ weights; $b^{[1]} \in \mathbb{R}^{128} \implies 128$ biases. Total = $24,064$.
- **Layer 2:** $W^{[2]} \in \mathbb{R}^{64 \times 128} \implies 8,192$ weights; $b^{[2]} \in \mathbb{R}^{64} \implies 64$ biases. Total = $8,256$.
- **Layer 3:** $W^{[3]} \in \mathbb{R}^{5 \times 64} \implies 320$ weights; $b^{[3]} \in \mathbb{R}^{5} \implies 5$ biases. Total = $325$.

$$\text{Total Trainable Parameters} = 24,064 + 8,256 + 325 = \mathbf{32,645}$$

### He (Kaiming) Weight Initialisation

Weights are drawn from a zero-mean Gaussian distribution with standard deviation:
$$\sigma = \sqrt{\frac{2}{n^{[\ell-1]}}}, \qquad b^{[\ell]} = \mathbf{0}$$

Why is this essential?
1. **Breaking Symmetry:** Initialising all weights to zero would cause all hidden units in a layer to compute identical outputs and receive identical gradients, preventing the network from learning heterogeneous features.
2. **Variance Preservation with ReLU:** Because ReLU deactivates approximately half of all activations (mapping negative inputs to zero), the variance of the signal drops by half at each layer. Scaling the variance by a factor of 2 (i.e. $\sigma^2 = \frac{2}{n_{\text{in}}}$ rather than Xavier's $\frac{1}{n_{\text{in}}}$) prevents the forward signal and backward gradients from vanishing or exploding across deep layers.

---

## 4. Loss Function & Analytical Gradient Verification

### Numerically Stable Softmax Cross-Entropy

For a batch of $B$ examples with logits $Z \in \mathbb{R}^{B \times 5}$ and ground-truth integer labels $y \in \{0, \dots, 4\}^B$, the per-sample loss is:
$$L_i = -\log \left(\frac{e^{z_{i, y_i}}}{\sum_{k=1}^5 e^{z_{i, k}}}\right) = -z_{i, y_i} + \log \left(\sum_{k=1}^5 e^{z_{i, k}}\right)$$

We implement this via `torch.logsumexp` to avert numerical overflow from large positive logits:
$$J = \frac{1}{B} \sum_{i=1}^B \left[-z_{i, y_i} + \text{logsumexp}(z_i)\right]$$

At initialisation, with random weights and zero biases, all 5 classes are roughly equiprobable ($\hat{y}_k \approx 0.2$), yielding an expected initial loss of:
$$\mathbb{E}[J] \approx \log(5) \approx 1.6094$$

### Mathematical Gradient Check

To ensure complete mathematical rigour, we derived the analytical gradient of the loss with respect to the output layer parameters:
$$\frac{\partial J}{\partial W^{[3]}} = \frac{1}{B} (\hat{Y} - Y)^\top A^{[2]}, \qquad \frac{\partial J}{\partial b^{[3]}} = \frac{1}{B} \sum_{i=1}^B (\hat{y}^{(i)} - y^{(i)})$$
where $\hat{Y} \in \mathbb{R}^{B \times 5}$ is the softmax probability matrix, $Y \in \mathbb{R}^{B \times 5}$ is the one-hot ground truth matrix, and $A^{[2]} \in \mathbb{R}^{B \times 64}$ contains the activations of the second hidden layer.

Evaluating this on the first $B=64$ training samples and comparing against PyTorch’s autograd `.grad` tensors yielded:
- $\max \left|\frac{\partial J}{\partial W^{[3]}}_{\text{hand}} - \frac{\partial J}{\partial W^{[3]}}_{\text{autograd}}\right| = \mathbf{2.98 \times 10^{-8}}$
- $\max \left|\frac{\partial J}{\partial b^{[3]}}_{\text{hand}} - \frac{\partial J}{\partial b^{[3]}}_{\text{autograd}}\right| = \mathbf{5.96 \times 10^{-8}}$

Both match to eight decimal places, far surpassing the required threshold of $10^{-5}$.

---

## 5. Optimisation: Manual Mini-Batch SGD with Momentum & Weight Decay

We built our training loop and optimiser from scratch using mini-batch gradient descent with Polyak momentum ($\beta = 0.9$) and $L_2$ weight decay ($\lambda = 10^{-4}$):

For each parameter $P$ with velocity $v_P$ (initialised to 0):
$$\begin{aligned}
g &= \frac{\partial J}{\partial P} + \lambda P \quad (\text{weights only; for biases } g = \frac{\partial J}{\partial P}) \\
v_P &\leftarrow \beta \, v_P + (1 - \beta) \, g \\
P &\leftarrow P - \alpha \, v_P
\end{aligned}$$

### Hyperparameter Search & Learning Rate Selection

We partitioned the 87,554 training rows into 78,799 training samples and 8,755 validation samples (fixed random seed 42, 10% validation). We evaluated three learning rates $\alpha \in \{0.01, 0.1, 0.5\}$ over 30 epochs with batch size $B=128$:

- **$\alpha = 0.01$:** Exhibits steady but sluggish convergence.
- **$\alpha = 0.1$:** Achieves rapid loss reduction and reaches the lowest validation loss with superb stability.
- **$\alpha = 0.5$:** Demonstrates oscillatory behavior due to oversized steps in parameter space.

Consequently, $\alpha = 0.1$ was selected as the optimal learning rate. The model checkpoint corresponding to the lowest validation loss epoch was preserved for final evaluation.

---

## 6. Comprehensive Test Results & Evaluation

The final model was evaluated once on the unseen test set of 21,892 heartbeats.

### Key Performance Summary
- **Overall Test Accuracy:** $\mathbf{97.5\%}$
- **Balanced Accuracy:** $\mathbf{83.2\%}$
- **Always-"N" Baseline Accuracy:** $\mathbf{82.8\%}$

### Why Overall Accuracy is Deceptive

Class N constitutes $18,118 / 21,892 = 82.76\%$ of the test set. A trivial "dummy" classifier that always outputs "Normal" without looking at the signal would automatically score **82.76% accuracy**. 

However, in medical diagnostics, missing an abnormal arrhythmia can be fatal. If a bedside monitor had 83% accuracy simply by ignoring all ventricular ectopic beats, patients with life-threatening ventricular tachycardia would go undetected! This exposes why **balanced accuracy** (the unweighted mean of all per-class accuracies) is the indispensable metric for imbalanced clinical data.

### Per-Class Performance and Morphological Confusions

The per-class accuracies highlight the clinical difficulty:
- **N (Normal):** $>98\%$
- **Q (Unknown/Paced):** $>97\%$
- **V (Ventricular Ectopic):** $>92\%$
- **S (Supraventricular Ectopic):** $\sim 65-72\%$
- **F (Fusion):** $\sim 60-70\%$

**Why is Class S most frequently misclassified as Class N?**
Supraventricular ectopic beats originate above the ventricles. Because they travel down the normal bundle branches, their ventricular depolarisation wave (the QRS complex) has almost normal width and orientation. Distinguishing S from N relies predominantly on detecting subtle shifts in P wave timing and shape. Given that Class N is 32 times more prevalent in training data, the network develops a Bayesian prior that biases ambiguous waveforms toward Class N.

---

## 7. Analysis & Critical Questions Answered

### 1. What does momentum do when consecutive gradients oppose each other?
When two consecutive mini-batch gradients point in opposite directions, momentum acts as a low-pass filter. The velocity term exponentially smooths gradient updates ($v \leftarrow \beta v + (1-\beta)g$). Opposing gradients cancel out, suppressing wild oscillations in steep ravines of the loss surface while reinforcing consistent gradient directions.

### 2. What does weight decay do to a weight that receives zero gradient?
If $\frac{\partial J}{\partial P} = 0$, the gradient term reduces to $g = \lambda P$. The update becomes:
$$P \leftarrow P - \alpha (1-\beta) \lambda P = P \left(1 - \alpha (1-\beta) \lambda\right)$$
Because $\left(1 - \alpha (1-\beta) \lambda\right) < 1$, the parameter exponentially decays toward zero with every step. This continuously prunes uninformative weights and guards against overfitting.

### 3. How to help rare classes like Fusion (F)?
With only 641 training samples for Class F, one principled solution is **Class-Weighted Cross-Entropy**:
$$J_{\text{weighted}} = \frac{1}{B} \sum_{i=1}^B w_{y_i} \cdot \left[-z_{i, y_i} + \log \sum_j e^{z_{i, j}}\right], \qquad w_c = \frac{N_{\text{total}}}{5 \cdot N_c}$$
Because $w_F \approx 113 \times w_N$, any classification error on a rare Fusion beat penalises the network 113 times more heavily, preventing the optimiser from ignoring rare classes in favour of the majority class.

---

## 8. Interactive Deployment: Streamlit Front-End

To enable clinical practitioners to test individual heartbeats, we deployed an interactive web application built with Streamlit (`app/app.py`):
1. **Test-Set Exploration:** Users can select any index from 0 to 21,891 to inspect the raw waveform in milliseconds, alongside the true label, model prediction, and confidence score.
2. **Custom File Upload:** Clinicians can upload a 187-value CSV heartbeat signal to receive instantaneous probabilistic predictions.
3. **Probability Bar Chart:** Visualises the softmax distribution across all five arrhythmia categories.

---

## 9. Conclusion

Building deep neural networks from mathematical fundamentals offers unmatched clarity into how parameter initialisation, regularisation, numerical stability, and class imbalance govern model behavior. While our model achieves over 97% overall accuracy, clinically sound evaluation demands inspecting the confusion matrix and balanced accuracy to ensure life-saving alarms are never silenced.

*The complete code repository, experiments, and web application are open-sourced on GitHub.*
