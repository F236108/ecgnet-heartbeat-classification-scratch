/**
 * ECG.NET — Client-Side Neural Network Inference & Visualizer
 * Built from scratch in pure JavaScript matching the PyTorch ECGNet model.
 * Includes complete interactive Neural Network Layer Simulation.
 */

const CLASS_INFO = [
  {
    code: "N",
    name: "Normal Beat",
    desc: "Normal sinoatrial rhythm. Regular P-wave, narrow QRS complex, normal ST segment and T-wave."
  },
  {
    code: "S",
    name: "Supraventricular Ectopic",
    desc: "Ectopic beat originating above the ventricles. Narrow QRS complex but premature or inverted P-wave."
  },
  {
    code: "V",
    name: "Ventricular Ectopic",
    desc: "Premature ventricular contraction (PVC). Wide, abnormal, bizarre QRS complex with compensatory pause."
  },
  {
    code: "F",
    name: "Fusion Beat",
    desc: "Hybrid beat from concurrent normal and ventricular activation. Intermediate waveform morphology."
  },
  {
    code: "Q",
    name: "Unknown / Paced",
    desc: "Artificial pacemaker spike or unclassifiable anomalous cardiac rhythm."
  }
];

// State variables
let currentSignal = null;
let currentTrueClass = 0;
let currentForwardResult = null;

// DOM Elements - ECG
const ecgCanvas = document.getElementById("ecgCanvas");
const ecgCtx = ecgCanvas.getContext("2d");
const predictedClassNameEl = document.getElementById("predictedClassName");
const confidenceRateEl = document.getElementById("confidenceRate");
const groundTruthLabelEl = document.getElementById("groundTruthLabel");
const matchTagEl = document.getElementById("matchTag");
const clinicalDescEl = document.getElementById("clinicalDesc");
const probListEl = document.getElementById("probList");

// DOM Elements - Neural Sim
const nnCanvas = document.getElementById("nnCanvas");
const nnCtx = nnCanvas.getContext("2d");
const simTooltip = document.getElementById("simTooltip");
const telInputCount = document.getElementById("telInputCount");
const telH1Active = document.getElementById("telH1Active");
const telH1Sparsity = document.getElementById("telH1Sparsity");
const telH2Active = document.getElementById("telH2Active");
const telH2Sparsity = document.getElementById("telH2Sparsity");
const telOutputClass = document.getElementById("telOutputClass");
const telOutputProb = document.getElementById("telOutputProb");
const triggerPulseBtn = document.getElementById("triggerPulseBtn");
const toggleSynapsesBtn = document.getElementById("toggleSynapsesBtn");
const synapseDensityLabel = document.getElementById("synapseDensityLabel");

// Synapse density settings
const DENSITY_MODES = ["Optimal", "Dense", "Minimal"];
let densityIndex = 0;

// --- Neural Network Forward Pass in pure JS ---
function forwardPass(x) {
  const { W1, b1, W2, b2, W3, b3 } = MODEL_DATA.weights;

  // Layer 1: z1 = x @ W1^T + b1  (187 -> 128)
  const z1 = new Float32Array(128);
  const a1 = new Float32Array(128);
  for (let j = 0; j < 128; j++) {
    let sum = b1[j];
    const wRow = W1[j];
    for (let i = 0; i < 187; i++) {
      sum += x[i] * wRow[i];
    }
    z1[j] = sum;
    a1[j] = Math.max(0, sum); // ReLU
  }

  // Layer 2: z2 = a1 @ W2^T + b2  (128 -> 64)
  const z2 = new Float32Array(64);
  const a2 = new Float32Array(64);
  for (let j = 0; j < 64; j++) {
    let sum = b2[j];
    const wRow = W2[j];
    for (let i = 0; i < 128; i++) {
      sum += a1[i] * wRow[i];
    }
    z2[j] = sum;
    a2[j] = Math.max(0, sum); // ReLU
  }

  // Layer 3: z3 = a2 @ W3^T + b3  (64 -> 5)
  const z3 = new Float32Array(5);
  for (let j = 0; j < 5; j++) {
    let sum = b3[j];
    const wRow = W3[j];
    for (let i = 0; i < 64; i++) {
      sum += a2[i] * wRow[i];
    }
    z3[j] = sum; // Logits
  }

  // Softmax
  let maxLogit = -Infinity;
  for (let j = 0; j < 5; j++) {
    if (z3[j] > maxLogit) maxLogit = z3[j];
  }

  let expSum = 0;
  const probs = new Float32Array(5);
  for (let j = 0; j < 5; j++) {
    probs[j] = Math.exp(z3[j] - maxLogit);
    expSum += probs[j];
  }

  let predClass = 0;
  let maxProb = 0;
  for (let j = 0; j < 5; j++) {
    probs[j] /= expSum;
    if (probs[j] > maxProb) {
      maxProb = probs[j];
      predClass = j;
    }
  }

  return {
    x,
    z1, a1,
    z2, a2,
    z3, probs,
    predClass,
    maxProb
  };
}

// --- ECG Waveform Canvas Rendering ---
function renderECG(signal) {
  const dpr = window.devicePixelRatio || 1;
  const width = ecgCanvas.parentElement.clientWidth;
  const height = ecgCanvas.parentElement.clientHeight;

  ecgCanvas.width = width * dpr;
  ecgCanvas.height = height * dpr;
  ecgCtx.scale(dpr, dpr);

  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  ecgCtx.fillStyle = isDark ? "#060606" : "#0d1117";
  ecgCtx.fillRect(0, 0, width, height);

  // Grid
  const gridColor = isDark ? "rgba(255, 49, 49, 0.08)" : "rgba(255, 255, 255, 0.08)";
  ecgCtx.strokeStyle = gridColor;
  ecgCtx.lineWidth = 1;

  const gridSize = 24;
  ecgCtx.beginPath();
  for (let x = 0; x < width; x += gridSize) {
    ecgCtx.moveTo(x, 0);
    ecgCtx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += gridSize) {
    ecgCtx.moveTo(0, y);
    ecgCtx.lineTo(width, y);
  }
  ecgCtx.stroke();

  if (!signal || signal.length < 187) return;

  const padX = 20;
  const padY = 30;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  // Trace
  ecgCtx.save();
  ecgCtx.shadowColor = "#ff3131";
  ecgCtx.shadowBlur = 12;
  ecgCtx.strokeStyle = "#ff3131";
  ecgCtx.lineWidth = 2.5;
  ecgCtx.lineJoin = "round";
  ecgCtx.lineCap = "round";

  ecgCtx.beginPath();
  for (let i = 0; i < 187; i++) {
    const x = padX + (i / 186) * plotW;
    const y = height - padY - signal[i] * plotH;
    if (i === 0) ecgCtx.moveTo(x, y);
    else ecgCtx.lineTo(x, y);
  }
  ecgCtx.stroke();
  ecgCtx.restore();

  // Labels
  ecgCtx.fillStyle = isDark ? "rgba(255, 255, 255, 0.4)" : "rgba(255, 255, 255, 0.6)";
  ecgCtx.font = "10px JetBrains Mono";
  ecgCtx.fillText("0 ms", padX, height - 8);
  ecgCtx.fillText("748 ms", width / 2 - 20, height - 8);
  ecgCtx.fillText("1496 ms", width - padX - 45, height - 8);
}

// ══════════════════════════════════════════════════════════════
// NEURAL NETWORK SIMULATION ENGINE
// ══════════════════════════════════════════════════════════════
class NeuralSimulator {
  constructor() {
    this.layerSizes = [187, 128, 64, 5];
    this.layerNames = ["Input (187)", "Hidden 1 (128 ReLU)", "Hidden 2 (64 ReLU)", "Output (5 Classes)"];
    this.nodes = []; // [layer][node] -> {x, y, radius, val, rawZ, layer, index}
    this.pulses = []; // electrical impulse particles
    this.animId = null;
    this.hoveredNode = null;
    this.waveProgress = 1.0; // 0 to 1

    this.initMouseEvents();
    this.animate = this.animate.bind(this);
    this.animate();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = nnCanvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height || 440;

    nnCanvas.width = this.width * dpr;
    nnCanvas.height = this.height * dpr;
    this.dpr = dpr;

    this.computeNodePositions();
  }

  computeNodePositions() {
    this.nodes = [];
    const padTop = 60;
    const padBottom = 30;
    const availH = this.height - padTop - padBottom;

    const xOffsets = [0.10, 0.36, 0.64, 0.90];

    for (let l = 0; l < 4; l++) {
      const count = this.layerSizes[l];
      const layerNodes = [];
      const x = this.width * xOffsets[l];

      let r = 2.2;
      if (l === 1) r = 2.6;
      if (l === 2) r = 3.8;
      if (l === 3) r = 16.0;

      for (let i = 0; i < count; i++) {
        let y;
        if (count > 1) {
          y = padTop + (i / (count - 1)) * availH;
        } else {
          y = padTop + availH / 2;
        }

        layerNodes.push({
          x,
          y,
          r,
          val: 0,
          rawZ: 0,
          layer: l,
          index: i
        });
      }
      this.nodes.push(layerNodes);
    }
  }

  updateActivations(fwd) {
    if (!fwd) return;

    // Layer 0: Input (187)
    for (let i = 0; i < 187; i++) {
      this.nodes[0][i].val = fwd.x[i];
      this.nodes[0][i].rawZ = fwd.x[i];
    }

    // Layer 1: Hidden 1 (128)
    let maxA1 = 0.0001;
    for (let i = 0; i < 128; i++) if (fwd.a1[i] > maxA1) maxA1 = fwd.a1[i];
    let h1ActiveCount = 0;
    for (let i = 0; i < 128; i++) {
      this.nodes[1][i].val = fwd.a1[i] / maxA1;
      this.nodes[1][i].rawZ = fwd.z1[i];
      if (fwd.a1[i] > 0) h1ActiveCount++;
    }

    // Layer 2: Hidden 2 (64)
    let maxA2 = 0.0001;
    for (let i = 0; i < 64; i++) if (fwd.a2[i] > maxA2) maxA2 = fwd.a2[i];
    let h2ActiveCount = 0;
    for (let i = 0; i < 64; i++) {
      this.nodes[2][i].val = fwd.a2[i] / maxA2;
      this.nodes[2][i].rawZ = fwd.z2[i];
      if (fwd.a2[i] > 0) h2ActiveCount++;
    }

    // Layer 3: Output (5)
    for (let i = 0; i < 5; i++) {
      this.nodes[3][i].val = fwd.probs[i];
      this.nodes[3][i].rawZ = fwd.z3[i];
    }

    // Update Telemetry Panel
    telInputCount.textContent = "187 / 187";
    telH1Active.textContent = `${h1ActiveCount} / 128`;
    telH1Sparsity.textContent = `Sparsity: ${((1 - h1ActiveCount / 128) * 100).toFixed(1)}%`;

    telH2Active.textContent = `${h2ActiveCount} / 64`;
    telH2Sparsity.textContent = `Sparsity: ${((1 - h2ActiveCount / 64) * 100).toFixed(1)}%`;

    telOutputClass.textContent = `${CLASS_INFO[fwd.predClass].code}: ${CLASS_INFO[fwd.predClass].name}`;
    telOutputProb.textContent = `Confidence: ${(fwd.maxProb * 100).toFixed(1)}%`;

    // Trigger impulse wave
    this.triggerWave();
  }

  triggerWave() {
    this.pulses = [];
    const mode = DENSITY_MODES[densityIndex];
    const particleCount = mode === "Dense" ? 48 : (mode === "Minimal" ? 18 : 32);

    // Stage 1: Input to H1
    for (let k = 0; k < particleCount; k++) {
      const srcIdx = Math.floor(Math.random() * 187);
      const tgtIdx = Math.floor(Math.random() * 128);
      this.pulses.push({
        layerFrom: 0,
        srcIdx,
        tgtIdx,
        progress: -Math.random() * 0.2, // staggered start
        speed: 0.02 + Math.random() * 0.015,
        stage: 1
      });
    }

    this.waveProgress = 0;
  }

  initMouseEvents() {
    nnCanvas.addEventListener("mousemove", (e) => {
      const rect = nnCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let found = null;
      for (let l = 0; l < 4; l++) {
        const nodes = this.nodes[l];
        if (!nodes) continue;
        for (let i = 0; i < nodes.length; i++) {
          const node = nodes[i];
          const hitRadius = Math.max(node.r + 5, 8);
          const dx = mouseX - node.x;
          const dy = mouseY - node.y;
          if (dx * dx + dy * dy <= hitRadius * hitRadius) {
            found = node;
            break;
          }
        }
        if (found) break;
      }

      this.hoveredNode = found;
      if (found) {
        simTooltip.style.opacity = "1";
        simTooltip.style.left = `${found.x}px`;
        simTooltip.style.top = `${found.y}px`;

        let layerTitle = this.layerNames[found.layer];
        let status = "";
        let details = "";

        if (found.layer === 0) {
          details = `ECG Voltage: ${found.val.toFixed(4)}`;
        } else if (found.layer === 1 || found.layer === 2) {
          const isActive = found.val > 0;
          status = isActive ? `<span style="color:#10b981;">[Active ReLU]</span>` : `<span style="color:#6b7280;">[Clamped to 0]</span>`;
          details = `Pre-ReLU z: ${found.rawZ.toFixed(3)}<br />Post-ReLU a: ${(found.val).toFixed(3)} ${status}`;
        } else {
          details = `Class: <strong>${CLASS_INFO[found.index].code} (${CLASS_INFO[found.index].name})</strong><br />Logit z: ${found.rawZ.toFixed(3)}<br />Softmax Prob: ${(found.val * 100).toFixed(2)}%`;
        }

        simTooltip.innerHTML = `
          <div style="font-size:0.7rem; color:var(--accent); font-weight:700; margin-bottom:2px;">
            ${layerTitle} — Node #${found.index}
          </div>
          <div>${details}</div>
        `;
      } else {
        simTooltip.style.opacity = "0";
      }
    });

    nnCanvas.addEventListener("mouseleave", () => {
      this.hoveredNode = null;
      simTooltip.style.opacity = "0";
    });
  }

  animate() {
    this.render();
    this.animId = requestAnimationFrame(this.animate);
  }

  render() {
    if (!this.width || !this.height) return;

    nnCtx.save();
    nnCtx.scale(this.dpr, this.dpr);

    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    nnCtx.fillStyle = isDark ? "#040404" : "#0a0f1d";
    nnCtx.fillRect(0, 0, this.width, this.height);

    // Layer Headers
    nnCtx.fillStyle = isDark ? "rgba(255, 255, 255, 0.45)" : "rgba(255, 255, 255, 0.65)";
    nnCtx.font = "600 11px 'Space Grotesk', sans-serif";
    nnCtx.textAlign = "center";

    const xOffsets = [0.10, 0.36, 0.64, 0.90];
    for (let l = 0; l < 4; l++) {
      nnCtx.fillText(this.layerNames[l].toUpperCase(), this.width * xOffsets[l], 28);
    }

    // Draw Synapses
    this.drawSynapses(isDark);

    // Update and Draw Traveling Impulse Pulses
    this.drawPulses();

    // Draw Neurons
    this.drawNeurons(isDark);

    nnCtx.restore();
  }

  drawSynapses(isDark) {
    const mode = DENSITY_MODES[densityIndex];
    const stride0 = mode === "Dense" ? 6 : (mode === "Minimal" ? 18 : 10);
    const stride1 = mode === "Dense" ? 4 : (mode === "Minimal" ? 12 : 6);
    const stride2 = mode === "Dense" ? 2 : (mode === "Minimal" ? 6 : 3);

    nnCtx.lineWidth = 0.6;

    // Synapses Layer 0 -> Layer 1
    for (let i = 0; i < 187; i += stride0) {
      const src = this.nodes[0][i];
      if (src.val < 0.1) continue;
      for (let j = 0; j < 128; j += stride1) {
        const tgt = this.nodes[1][j];
        if (tgt.val > 0) {
          const alpha = 0.05 + 0.15 * src.val * tgt.val;
          nnCtx.strokeStyle = `rgba(255, 49, 49, ${alpha})`;
          nnCtx.beginPath();
          nnCtx.moveTo(src.x, src.y);
          nnCtx.lineTo(tgt.x, tgt.y);
          nnCtx.stroke();
        }
      }
    }

    // Synapses Layer 1 -> Layer 2
    for (let j = 0; j < 128; j += stride1) {
      const src = this.nodes[1][j];
      if (src.val === 0) continue;
      for (let k = 0; k < 64; k += stride2) {
        const tgt = this.nodes[2][k];
        if (tgt.val > 0) {
          const alpha = 0.07 + 0.20 * src.val * tgt.val;
          nnCtx.strokeStyle = `rgba(255, 49, 49, ${alpha})`;
          nnCtx.beginPath();
          nnCtx.moveTo(src.x, src.y);
          nnCtx.lineTo(tgt.x, tgt.y);
          nnCtx.stroke();
        }
      }
    }

    // Synapses Layer 2 -> Layer 3 (All 64 to 5)
    for (let k = 0; k < 64; k += (mode === "Minimal" ? 4 : 2)) {
      const src = this.nodes[2][k];
      for (let c = 0; c < 5; c++) {
        const tgt = this.nodes[3][c];
        const isWinner = currentForwardResult && currentForwardResult.predClass === c;
        const alpha = isWinner ? 0.25 : 0.04;
        nnCtx.strokeStyle = isWinner ? `rgba(16, 185, 129, ${alpha})` : `rgba(255, 49, 49, ${alpha})`;
        nnCtx.beginPath();
        nnCtx.moveTo(src.x, src.y);
        nnCtx.lineTo(tgt.x, tgt.y);
        nnCtx.stroke();
      }
    }
  }

  drawPulses() {
    for (let p = this.pulses.length - 1; p >= 0; p--) {
      const pulse = this.pulses[p];
      pulse.progress += pulse.speed;

      if (pulse.progress < 0) continue;

      if (pulse.progress >= 1.0) {
        // Cascade to next layer
        if (pulse.stage === 1) {
          pulse.stage = 2;
          pulse.layerFrom = 1;
          pulse.srcIdx = pulse.tgtIdx;
          pulse.tgtIdx = Math.floor(Math.random() * 64);
          pulse.progress = 0;
        } else if (pulse.stage === 2) {
          pulse.stage = 3;
          pulse.layerFrom = 2;
          pulse.srcIdx = pulse.tgtIdx;
          pulse.tgtIdx = currentForwardResult ? currentForwardResult.predClass : 0;
          pulse.progress = 0;
        } else {
          this.pulses.splice(p, 1);
          continue;
        }
      }

      const src = this.nodes[pulse.layerFrom][pulse.srcIdx];
      const nextLayer = pulse.layerFrom + 1;
      const tgt = this.nodes[nextLayer][pulse.tgtIdx];

      if (!src || !tgt) continue;

      const px = src.x + (tgt.x - src.x) * pulse.progress;
      const py = src.y + (tgt.y - src.y) * pulse.progress;

      nnCtx.save();
      nnCtx.shadowColor = "#ffffff";
      nnCtx.shadowBlur = 8;
      nnCtx.fillStyle = "#ffffff";
      nnCtx.beginPath();
      nnCtx.arc(px, py, 2.0, 0, Math.PI * 2);
      nnCtx.fill();
      nnCtx.restore();
    }
  }

  drawNeurons(isDark) {
    for (let l = 0; l < 4; l++) {
      const nodes = this.nodes[l];
      const isOutput = l === 3;

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const isHovered = this.hoveredNode === node;

        if (isOutput) {
          // Output Nodes: Large circles with Class Code
          const isWinner = currentForwardResult && currentForwardResult.predClass === i;
          const prob = node.val;

          nnCtx.save();
          if (isWinner) {
            nnCtx.shadowColor = "#10b981";
            nnCtx.shadowBlur = 18;
            nnCtx.fillStyle = "#10b981";
            nnCtx.strokeStyle = "#ffffff";
            nnCtx.lineWidth = 2.5;
          } else {
            nnCtx.fillStyle = isDark ? "#1f1f1f" : "#2a3447";
            nnCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            nnCtx.lineWidth = 1.5;
          }

          nnCtx.beginPath();
          nnCtx.arc(node.x, node.y, node.r + (isWinner ? 2 : 0), 0, Math.PI * 2);
          nnCtx.fill();
          nnCtx.stroke();
          nnCtx.restore();

          // Class letter inside circle
          nnCtx.fillStyle = isWinner ? "#000000" : "#ffffff";
          nnCtx.font = "bold 13px 'JetBrains Mono', monospace";
          nnCtx.textAlign = "center";
          nnCtx.textBaseline = "middle";
          nnCtx.fillText(CLASS_INFO[i].code, node.x, node.y);

          // Probability percentage label beside circle
          nnCtx.textAlign = "left";
          nnCtx.font = isWinner ? "bold 11px 'JetBrains Mono'" : "10px 'JetBrains Mono'";
          nnCtx.fillStyle = isWinner ? "#10b981" : (isDark ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.7)");
          nnCtx.fillText(`${(prob * 100).toFixed(1)}%`, node.x + node.r + 8, node.y);

        } else if (l === 0) {
          // Input Nodes: Temporal signal nodes
          const intensity = node.val;
          nnCtx.save();
          if (intensity > 0.05) {
            nnCtx.fillStyle = `rgba(59, 130, 246, ${0.3 + 0.7 * intensity})`;
            if (isHovered) {
              nnCtx.shadowColor = "#3b82f6";
              nnCtx.shadowBlur = 10;
              nnCtx.fillStyle = "#ffffff";
            }
          } else {
            nnCtx.fillStyle = isDark ? "#1a1a1a" : "#2a3040";
          }
          nnCtx.beginPath();
          nnCtx.arc(node.x, node.y, isHovered ? node.r * 2 : node.r, 0, Math.PI * 2);
          nnCtx.fill();
          nnCtx.restore();

        } else {
          // Hidden Layers: ReLU Activation Nodes
          const isActive = node.val > 0;
          nnCtx.save();
          if (isActive) {
            const glow = Math.min(1.0, 0.4 + node.val * 0.6);
            nnCtx.fillStyle = `rgba(255, 49, 49, ${glow})`;
            if (node.val > 0.5) {
              nnCtx.shadowColor = "#ff3131";
              nnCtx.shadowBlur = 8;
            }
            if (isHovered) {
              nnCtx.fillStyle = "#ffffff";
              nnCtx.shadowColor = "#ffffff";
              nnCtx.shadowBlur = 12;
            }
          } else {
            // Clamped by ReLU (a = 0)
            nnCtx.fillStyle = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(255, 255, 255, 0.12)";
          }
          nnCtx.beginPath();
          nnCtx.arc(node.x, node.y, isHovered ? node.r * 1.8 : node.r, 0, Math.PI * 2);
          nnCtx.fill();
          nnCtx.restore();
        }
      }
    }
  }
}

// Global Simulator Instance
let sim = null;

// --- Update All App UI ---
function runInference(signal, trueClass = null, labelText = null) {
  currentSignal = signal;
  renderECG(signal);

  const fwd = forwardPass(signal);
  currentForwardResult = fwd;

  // Prediction Name
  predictedClassNameEl.innerHTML = `<span>${CLASS_INFO[fwd.predClass].code}:</span> ${CLASS_INFO[fwd.predClass].name}`;
  confidenceRateEl.textContent = `${(fwd.maxProb * 100).toFixed(1)}%`;

  // Ground truth check
  if (trueClass !== null) {
    groundTruthLabelEl.textContent = `${CLASS_INFO[trueClass].code} (${CLASS_INFO[trueClass].name})`;
    const isCorrect = fwd.predClass === trueClass;
    matchTagEl.textContent = isCorrect ? "Match" : "Mismatch";
    matchTagEl.className = `match-tag ${isCorrect ? "correct" : "mismatch"}`;
  } else {
    groundTruthLabelEl.textContent = labelText || "User Uploaded CSV";
    matchTagEl.textContent = "Unlabeled";
    matchTagEl.className = "match-tag";
  }

  // Clinical Summary
  clinicalDescEl.innerHTML = `<strong>Diagnostic Profile:</strong> ${CLASS_INFO[fwd.predClass].desc}`;

  // Probabilities Bars
  probListEl.innerHTML = "";
  for (let c = 0; c < 5; c++) {
    const p = fwd.probs[c];
    const isPred = c === fwd.predClass;
    const item = document.createElement("div");
    item.className = `prob-item ${isPred ? "is-predicted" : ""}`;
    item.innerHTML = `
      <div class="prob-meta">
        <span class="class-title"><strong>${CLASS_INFO[c].code}</strong> — ${CLASS_INFO[c].name}</span>
        <span class="mono">${(p * 100).toFixed(2)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${(p * 100).toFixed(1)}%"></div>
      </div>
    `;
    probListEl.appendChild(item);
  }

  // Update Neural Simulation Canvas
  if (sim) {
    sim.updateActivations(fwd);
  }
}

// --- Load Preset Sample Beat ---
function loadPreset(classId) {
  const match = MODEL_DATA.samples.find(s => s.class === classId);
  if (match) {
    currentTrueClass = match.class;
    runInference(match.signal, match.class);

    document.querySelectorAll(".preset-chip").forEach(chip => {
      chip.classList.toggle("active", parseInt(chip.dataset.class) === classId);
    });
  }
}

// --- CSV Parser for Custom Upload ---
function handleCSVText(text) {
  const lines = text.trim().split(/[\r\n]+/);
  if (lines.length === 0) return alert("Empty CSV file.");
  
  let values = lines[0].split(/[,\t\s]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
  if (values.length < 187) {
    if (lines.length >= 187) {
      values = lines.slice(0, 187).map(l => parseFloat(l.trim())).filter(v => !isNaN(v));
    }
  }

  if (values.length < 187) {
    alert(`Invalid format: Expected at least 187 ECG values, got ${values.length}.`);
    return;
  }

  const signal = values.slice(0, 187);
  let trueClass = null;
  if (values.length >= 188) {
    const rawLabel = Math.round(values[187]);
    if (rawLabel >= 0 && rawLabel <= 4) {
      trueClass = rawLabel;
    }
  }

  runInference(signal, trueClass, "Custom CSV File");
}

// --- Event Listeners Initialization ---
window.addEventListener("DOMContentLoaded", () => {
  // Initialize Neural Simulator
  sim = new NeuralSimulator();
  sim.resize();

  // Initial inference on Class 0 (Normal)
  loadPreset(0);

  // Preset chips click
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const classId = parseInt(chip.dataset.class);
      loadPreset(classId);
    });
  });

  // Tab controls
  const tabPresets = document.getElementById("tabPresets");
  const tabUpload = document.getElementById("tabUpload");
  const presetsGroup = document.getElementById("presetsGroup");
  const uploadDropzone = document.getElementById("uploadDropzone");

  tabPresets.addEventListener("click", () => {
    tabPresets.classList.add("active");
    tabUpload.classList.remove("active");
    presetsGroup.style.display = "flex";
    uploadDropzone.classList.remove("active-tab");
  });

  tabUpload.addEventListener("click", () => {
    tabUpload.classList.add("active");
    tabPresets.classList.remove("active");
    presetsGroup.style.display = "none";
    uploadDropzone.classList.add("active-tab");
  });

  // File Upload Handlers
  const fileInput = document.getElementById("fileInput");
  uploadDropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => handleCSVText(evt.target.result);
    reader.readAsText(file);
  });

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => handleCSVText(evt.target.result);
    reader.readAsText(file);
  });

  // Re-trigger Pulse Button
  triggerPulseBtn.addEventListener("click", () => {
    if (sim) sim.triggerWave();
  });

  // Synapse Density Toggle Button
  toggleSynapsesBtn.addEventListener("click", () => {
    densityIndex = (densityIndex + 1) % DENSITY_MODES.length;
    synapseDensityLabel.textContent = DENSITY_MODES[densityIndex];
    if (sim) sim.triggerWave();
  });

  // Theme Toggle
  const themeToggle = document.getElementById("themeToggle");
  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const nextTheme = currentTheme === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", nextTheme);
    themeToggle.innerHTML = nextTheme === "light" 
      ? '<i class="fa-solid fa-moon"></i>' 
      : '<i class="fa-solid fa-sun"></i>';
    renderECG(currentSignal);
  });

  // Window Resize
  window.addEventListener("resize", () => {
    if (currentSignal) renderECG(currentSignal);
    if (sim) sim.resize();
  });
});
