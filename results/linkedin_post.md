🚀 Excited to share our latest Machine Learning project: **"Classifying Heartbeats from ECG Signals: Building a Neural Network from Scratch in PyTorch"**! ❤️⚡

Can a small neural network detect life-threatening cardiac arrhythmias, and can we truly trust "97% accuracy" when healthy heartbeats dominate the dataset?

To answer this question from first principles, we engineered and trained an end-to-end deep neural network **completely from scratch**:
🔹 **Zero High-Level Layer Abstractions:** No `nn.Linear`, `nn.Sequential`, or `nn.ReLU`. All weight matrices ($W^{[1]}, W^{[2]}, W^{[3]}$) and bias vectors were manually initialized with He (Kaiming) initialization and forward-propagated via tensor math.
🔹 **Custom Softmax Cross-Entropy:** Implemented with `torch.logsumexp` for numerical stability and verified against an exact hand-derived analytical gradient ($\Delta < 10^{-8}$ against autograd!).
🔹 **Manual Optimiser:** Built mini-batch SGD with momentum ($\beta = 0.9$) and $L_2$ weight decay ($\lambda = 10^{-4}$) directly in Python.
🔹 **Clinical Evaluation & The Imbalance Dilemma:** While overall test accuracy exceeded 97%, our analysis reveals why balanced accuracy is the true diagnostic metric—a naive classifier always predicting "Normal" achieves 82.8% accuracy while failing 100% of critical arrhythmias!
🔹 **Interactive Web App:** Deployed with Streamlit allowing real-time single-beat ECG waveform visualization and multi-class probability scoring.

📖 Read the complete deep-dive on Medium: [Link to Medium Article]
💻 Explore the code & test it yourself on GitHub: [Link to GitHub Repository]

#MachineLearning #DeepLearning #PyTorch #HealthTech #Cardiology #DataScience #ArtificialIntelligence #ECG
