"""
Task 7 – Front-end web application.

A Streamlit app where the user can:
  - Pick a test-set row number, or
  - Upload a CSV file with one row of 187 values
  - See the heartbeat plotted
  - See the predicted class and probabilities as a bar chart
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import ECGNet

CLASS_NAMES = ["N: Normal", "S: Supraventricular", "V: Ventricular", "F: Fusion", "Q: Unknown/Paced"]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]
FS = 125  # sampling frequency

@st.cache_resource
def load_model():
    """Load the trained ECGNet model."""
    model = ECGNet()
    state = torch.load(os.path.join(os.path.dirname(__file__), "..", "results", "ecgnet_best.pt"),
                       weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_data
def load_test_data():
    """Load the test CSV file."""
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "mitbih_test.csv")
    df = pd.read_csv(test_path, header=None)
    return df

def predict(model, signal):
    """Run prediction on a 187-length signal. Returns class, probabilities."""
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, 187)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_class = int(probs.argmax())
    return pred_class, probs

def plot_beat(signal, title="Heartbeat Signal"):
    """Plot a single heartbeat."""
    time_ms = [1000 * i / FS for i in range(187)]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_ms, signal, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalised Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_probabilities(probs):
    """Plot class probabilities as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    bars = ax.bar(CLASS_SHORT, probs, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Class Probabilities")
    ax.set_ylim(0, 1)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{p:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="❤️", layout="wide")
st.title("❤️ ECG Heartbeat Classifier")
st.markdown("Classify heartbeats from ECG signals using a neural network trained from scratch.")

model = load_model()

tab1, tab2 = st.tabs(["📋 Test Set Row", "📁 Upload CSV"])

with tab1:
    test_df = load_test_data()
    st.markdown(f"Test set has **{len(test_df)}** rows.")

    row_num = st.number_input("Select test-set row number:", min_value=0,
                               max_value=len(test_df)-1, value=0, step=1)

    if st.button("Classify", key="classify_row"):
        row = test_df.iloc[row_num]
        signal = row.iloc[:187].values
        true_label = int(row.iloc[187])

        pred_class, probs = predict(model, signal)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_beat(signal, f"Row {row_num} — True: {CLASS_NAMES[true_label]}"))
        with col2:
            st.pyplot(plot_probabilities(probs))

        st.success(f"**Predicted:** {CLASS_NAMES[pred_class]}  |  "
                   f"**True:** {CLASS_NAMES[true_label]}  |  "
                   f"**Confidence:** {probs[pred_class]*100:.1f}%")

with tab2:
    uploaded = st.file_uploader("Upload a CSV file with one row of 187 values", type="csv")
    if uploaded is not None:
        df_up = pd.read_csv(uploaded, header=None)
        if df_up.shape[1] >= 187:
            signal = df_up.iloc[0, :187].values.astype(float)
            pred_class, probs = predict(model, signal)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_beat(signal, "Uploaded Heartbeat"))
            with col2:
                st.pyplot(plot_probabilities(probs))

            st.success(f"**Predicted:** {CLASS_NAMES[pred_class]}  |  "
                       f"**Confidence:** {probs[pred_class]*100:.1f}%")
        else:
            st.error(f"CSV must have at least 187 columns. Got {df_up.shape[1]}.")
