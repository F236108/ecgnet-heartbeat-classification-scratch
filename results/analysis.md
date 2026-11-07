# Task 6 – Analysis

## 6.1 Always-"N" Baseline

A model that always predicts class N (normal beat) achieves a test accuracy of:
$$\text{Accuracy}_{\text{Always-N}} = \frac{18,118}{21,892} \approx 82.76\%$$

In comparison, our trained `ECGNet` achieves an overall test accuracy of **97.82%** (21,414 / 21,892) and a balanced accuracy of **83.65%**.

**Implications of Using Accuracy Alone:**
On severely imbalanced datasets, overall accuracy can create a dangerous illusion of competence. An uncalibrated or trivial model that classifies every patient's heartbeat as "Normal" without ever inspecting the ECG waveform scores ~83% accuracy. In a clinical hospital setting, such a monitor would fail to detect 100% of ventricular ectopic beats (arrhythmias associated with sudden cardiac death) and supraventricular tachycardias. Therefore, **overall accuracy alone must never be trusted** for medical diagnostics; balanced accuracy and the per-class confusion matrix are essential.

---

## 6.2 Lowest Per-Class Accuracy and Confusion Analysis

From Table 4, the class with the lowest per-class accuracy is **F (Fusion beat)** at **62.96%** (102 / 162), closely followed by **S (Supraventricular ectopic)** at **66.37%** (369 / 556).

### Confusion Matrix Breakdown:
- **Class F:** Out of 162 true F beats, 102 are correctly identified as F, **39 are mistaken for N**, and 21 are mistaken for V.
- **Class S:** Out of 556 true S beats, 369 are correctly identified as S, **181 are mistaken for N**, and 4 are mistaken for V.

### Why are these classes hardest to separate from N?
1. **Morphological Similarity for S vs N:** Supraventricular ectopic beats arise in the atria or AV node (above the ventricles). Because the electrical wavefront propagates through the normal His-Purkinje ventricular conduction system, the resulting QRS complex is sharp, narrow, and nearly indistinguishable from a normal beat. The primary differentiating feature is an abnormal or premature P wave, which is low-amplitude and easily masked by baseline noise.
2. **Hybrid Morphology for F vs N:** A fusion beat occurs when a sinus impulse and a ventricular ectopic impulse simultaneously depolarise parts of the myocardium. The resulting waveform is a hybrid morph between normal (N) and ventricular (V). When the normal pacemaker dominates, the beat closely resembles a normal beat (hence 39 misclassified as N).
3. **Severe Class Imbalance:** With 72,471 normal training beats versus only 641 fusion beats and 2,223 supraventricular beats, the network's loss landscape is dominated by the majority class, biasing ambiguous signals toward class N.

---

## 6.3 Loss Curves and Overfitting Evaluation

Examining Figure 3a (Loss Curves) and Figure 3b (Accuracy Curves):
- **Training Loss at Best Epoch (23):** 0.060877
- **Validation Loss at Best Epoch (23):** 0.072164
- **Training Accuracy at Best Epoch (23):** 0.9816 (98.16%)
- **Validation Accuracy at Best Epoch (23):** 0.9793 (97.93%)

### Assessment:
The curves demonstrate **mild, controlled overfitting**:
- A small, persistent gap (~0.011 loss units, ~0.23% accuracy difference) exists between the training and validation curves.
- Crucially, the validation loss does **not** experience an explosive upward divergence in later epochs; it decreases steadily through epoch 23 and plateaus gracefully around ~0.072–0.080.
- The $L_2$ weight decay ($\lambda = 10^{-4}$) successfully regularises the 32,645 parameters, restraining large weight magnitudes and preventing catastrophic memorisation.
- Early checkpointing (saving the model at epoch 23 with lowest validation loss) ensures that test evaluation is performed on the optimal generalisation parameters.

---

## 6.4 Weight Initialisation

### Why not initialise every weight to zero?
If all weights are initialised to zero ($W^{[\ell]} = \mathbf{0}$):
1. **Symmetry in Forward Propagation:** Every hidden unit in layer $\ell$ receives the exact same input $\sum_j 0 \cdot a_j + 0 = 0$. After ReLU, all activations are identical across all units in that layer.
2. **Symmetry in Backward Propagation:** Every neuron in layer $\ell$ receives the exact same backpropagated error signal $\delta^{[\ell]}$. Consequently, the gradients $\frac{\partial J}{\partial W^{[\ell]}_{ij}}$ are identical for all $i$.
3. **Lack of Capacity:** All neurons update identically and remain clones of one another throughout training. The entire hidden layer of 128 units behaves as if it were a single neuron, crippling the representational capacity of the network.

### Why does He (Kaiming) initialisation use $\sqrt{2 / n^{[\ell-1]}}$ for ReLU layers?
For linear layers, Xavier initialisation uses $\text{Var}(W) = 1 / n_{\text{in}}$ to maintain constant activation variance across layers.
However, the **ReLU activation function clamps all negative values to zero**:
$$\text{ReLU}(z) = \max(0, z)$$
Under the assumption of symmetric zero-mean activations, approximately **half of all activations are zeroed out**, halving the signal variance at each layer:
$$\text{Var}(a) \approx \frac{1}{2} \text{Var}(z)$$
To compensate for this 50% variance reduction and maintain steady activation variance $\text{Var}(z^{[\ell]}) \approx \text{Var}(z^{[\ell-1]})$ from the input layer through deep layers, He initialisation doubles the weight variance:
$$\text{Var}(W^{[\ell]}) = \frac{2}{n^{[\ell-1]}} \implies \sigma = \sqrt{\frac{2}{n^{[\ell-1]}}}$$
This prevents the forward activations from exponentially decaying to zero or exploding to infinity.

---

## 6.5 Momentum and Weight Decay Mechanics

### Momentum with Opposing Gradients:
The velocity update with momentum ($\beta = 0.9$) is:
$$v_P^{(t)} = \beta v_P^{(t-1)} + (1 - \beta) g^{(t)}$$
When two consecutive mini-batch gradients point in opposite directions ($g^{(t-1)} > 0$ and $g^{(t)} < 0$):
- Momentum computes an exponentially weighted moving average that **partially cancels out opposing gradient vectors**.
- This damps high-frequency oscillations along high-curvature ravines in the loss landscape.
- The optimiser maintains velocity only in consistent gradient directions, accelerating convergence down flat valleys while stabilising updates.

### Weight Decay on a Parameter with Zero Gradient:
When a weight parameter $W$ receives zero loss gradient ($\frac{\partial J}{\partial W} = 0$):
The effective gradient under $L_2$ regularisation becomes:
$$g = \frac{\partial J}{\partial W} + \lambda W = 0 + \lambda W = \lambda W$$
The momentum velocity accumulates this decay:
$$v_W \leftarrow \beta v_W + (1 - \beta) \lambda W$$
And the parameter update subtracts a positive fraction of $W$:
$$W \leftarrow W - \alpha v_W$$
Thus, **the weight exponentially decays toward zero**. Weight decay exerts continuous shrinkage on dormant or uninformative parameters, acting as a soft feature selector and preventing individual weights from becoming excessively large.

---

## 6.6 Strategies for Rare Classes (Class F & Class S)

With only 641 training samples for Fusion (F) beats compared to 72,471 for Normal (N) beats (a ratio of ~1:113), the standard unweighted cross-entropy loss allows the network to achieve low average loss while virtually ignoring the rare classes.

### Proposed Improvement: Class-Weighted Cross-Entropy Loss
We modify the loss function by scaling each training example's loss by an inverse-frequency class weight $w_c$:
$$J_{\text{weighted}} = \frac{1}{\sum_{i=1}^B w_{y_i}} \sum_{i=1}^B w_{y_i} \left[ -z_{i, y_i} + \log \left(\sum_{k=1}^5 e^{z_{i, k}}\right) \right]$$
where the class weight is inversely proportional to class prevalence:
$$w_c = \frac{N_{\text{total}}}{5 \cdot N_c}$$

For our training set:
- $w_N = \frac{87,554}{5 \times 72,471} \approx 0.24$
- $w_F = \frac{87,554}{5 \times 641} \approx 27.32$

**Why this helps:**
A single misclassification of a Fusion (F) beat now produces a gradient signal approximately **$113 \times$ larger** than a misclassification of a Normal (N) beat. During backpropagation, the gradients for Class F are no longer swamped by the overwhelming volume of Normal beats. The optimiser is compelled to adjust weight parameters to carve out decisive decision boundaries for Fusion and Supraventricular beats, substantially increasing balanced accuracy and clinical sensitivity.
