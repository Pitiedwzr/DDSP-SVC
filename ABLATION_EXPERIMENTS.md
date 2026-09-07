# DDSP-SVC & Self-Flow Ablation Experiment Suite

This document defines a systematic, reproducible ablation experiment suite for **DDSP-SVC** and its **Rectified Flow (Reflow)** extension. It evaluates mathematical corrections, optimizer behaviors, Self-Flow representation learning mechanisms, backbone architecture improvements, and sampling solver efficiency.

---

## 1. Experimental Methodology & Rigor

To ensure statistical rigor and isolate individual variables, all ablation runs must adhere to the following controls:

1. **Fixed Hardware & Environment**:
   - Single GPU (e.g., NVIDIA RTX 3090 / 4090 or A100).
   - PyTorch 2.x, CUDA 12.x / 13.x, cuDNN deterministic flags enabled for validation.
2. **Fixed Data Split & Preprocessing**:
   - Seed: `42`.
   - Preprocessed features (`units`, `f0`, `voiced`, `volume`, `mel`) computed once and frozen across all runs.
   - Content encoder: ContentVec 768L12TTA2X (`pretrain/contentvec/pytorch_model.bin`).
   - Pitch extractor: RMVPE (`pretrain/rmvpe/model.pt`).
3. **Fixed Optimization Protocol**:
   - Base learning rate: $\eta = 5 \times 10^{-4}$.
   - Batch size: 48 audio clips (2.0 seconds duration per sample).
   - Validation interval: 2,000 steps; force save every 10,000 steps; total training steps: 100,000 steps.
4. **Comprehensive Evaluation Metrics**:
   - **Mel Spectral Distortion (Mel-L1 / Mel-MSE)**: Multi-scale spectral distance on held-out validation set.
   - **F0 Pearson Correlation & F0-RMSE (cents)**: Pitch contour fidelity measured using RMVPE on synthesized audio.
   - **Voiced/Unvoiced (V/UV) Error Rate (%)**: Percentage of frames misclassified as voiced vs. unvoiced.
   - **Speaker Embedding Cosine Similarity (SECS)**: Extracted using a pretrained speaker verification model (e.g. ECAPA-TDNN or WavLM-SV).
   - **Fréchet Audio Distance (FAD)**: Distributional audio quality measured against ground-truth vocals.
   - **Computational Cost**: Real-Time Factor (RTF), Training Throughput (samples/sec), Peak VRAM (GB).

---

## 2. Group 1: Mathematical & Algorithmic Corrections

### Experiment 1.1: Gradient Isolation of DDSP Acoustic Prior (`detach` vs. Joint Backprop)

- **Hypothesis**: The Reflow model conditions on the DDSP mel spectrogram ($M_{\text{ddsp}}$). In the current code, `reflow_cond_mel = ddsp_mel` is not detached, allowing Reflow velocity loss gradients to backpropagate into the DDSP network. Isolating the gradients with `ddsp_mel.detach()` will prevent DDSP representation drift, stabilize training, and significantly improve shallow diffusion ($t_{\text{start}} > 0$).
- **Configuration**:
  - `E1.1-Base`: Current implementation (joint backprop through `ddsp_mel`).
  - `E1.1-Detach`: `reflow_cond_mel = ddsp_mel.detach()` in `reflow/vocoder.py:309`.
- **Primary Metrics**: DDSP Mel MSE, Reflow Validation Loss, Synthesis Mel-L1 at $t_{\text{start}} = 0.0$ and $t_{\text{start}} = 0.4$.

### Experiment 1.2: FP16 Stability & Mel Compression Epsilon

- **Hypothesis**: In `nsf_hifigan/nvSTFT.py`, dynamic range compression applies $\log(\max(x, 10^{-5}))$. The derivative is $1 / 10^{-5} = 100,000$, exceeding the maximum finite value of FP16 ($65,504$), triggering gradient overflows and NaN loss events. Setting a numerically stable epsilon or computing STFT/loss in Float32 eliminates NaN loss spikes and improves gradient fidelity.
- **Configuration**:
  - `E1.2-Base`: `clip_val = 1e-5` in `nvSTFT.py` (Current).
  - `E1.2-SafeClamp`: `clip_val = 1e-4` in `nvSTFT.py` ($1 / 10^{-4} = 10,000 < 65,504$).
  - `E1.2-FP32Loss`: Compute STFT and mel loss in `float32` before casting back to model precision.
- **Primary Metrics**: Number of NaN events / gradient scaler backoffs per 10k steps, final Mel-L1.

### Experiment 1.3: Optimizer Filter on Depthwise Convolutions

- **Hypothesis**: `optimizer/muon.py` includes all 2D+ parameters, improperly capturing `nn.Conv1d(1024, 1024, kernel_size=31, groups=1024)` depthwise filters. Orthogonalizing a $1024 \times 31$ matrix (rank $\le 31$) via Newton-Schulz damages independent spatial filtering. Routing depthwise convolutions to AdamW (matching `aurora.py`) improves convergence.
- **Configuration**:
  - `E1.3-Muon-Bug`: Current Muon implementation (depthwise conv included in Muon).
  - `E1.3-Muon-Fixed`: Muon with depthwise conv filtered out and routed to AdamW.
  - `E1.3-Aurora-Fixed`: Aurora optimizer with corrected aspect-ratio scaling (`max(dim0, dim1)**0.5`).
  - `E1.3-AdamW`: Pure AdamW baseline ($\beta_1=0.9, \beta_2=0.98, \text{lr}=5 \times 10^{-4}$).
- **Primary Metrics**: Training loss convergence rate, validation velocity MSE, spectral high-frequency SNR.

---

## 3. Group 2: Self-Supervised Flow Matching (Self-Flow) Mechanism

### Experiment 2.1: Information Asymmetry Masking Topology (Acoustic Span vs. i.i.d. Frame)

- **Hypothesis**: Audio frames are strongly correlated over phoneme durations ($50\text{--}200\text{ ms} \approx 5\text{--}20\text{ frames}$). Frame-level i.i.d. Bernoulli masking creates high-frequency noise that the network trivializes through linear interpolation of adjacent frames. Continuous span masking forces the model to learn long-range phonetic and harmonic dependencies.
- **Configuration**:
  - `E2.1-None`: Standard Reflow (`use_self_flow: false`).
  - `E2.1-IID`: i.i.d. Frame Masking (`self_flow_mask_ratio: 0.5`) (Current).
  - `E2.1-Span5`: Acoustic Span Masking with mean span length $L=5$ frames ($\sim 60\text{ ms}$).
  - `E2.1-Span12`: Acoustic Span Masking with mean span length $L=12$ frames ($\sim 140\text{ ms}$).
- **Primary Metrics**: Phoneme intelligibility (CER/WER via Whisper on converted audio), SECS, FAD.

### Experiment 2.2: Representation Alignment Scope (Selective vs. All Tokens)

- **Hypothesis**: Calculating cosine similarity loss over all tokens forces the student to mimic the teacher even on clean tokens where representation divergence is minimal. Computing the alignment loss selectively on corrupted/masked tokens provides stronger self-supervised gradients.
- **Configuration**:
  - `E2.2-All`: Compute loss over all tokens: $\mathcal{L}_{\text{rep}} = \frac{1}{T} \sum_{t=1}^T (1 - \cos(s_t, z_t))$ (Current).
  - `E2.2-MaskedOnly`: Compute loss exclusively on masked/perturbed tokens: $\mathcal{L}_{\text{rep}} = \frac{1}{|M|} \sum_{t \in M} (1 - \cos(s_t, z_t))$.
  - `E2.2-Weighted`: Weighted combination: $0.8 \times \mathcal{L}_{\text{masked}} + 0.2 \times \mathcal{L}_{\text{unmasked}}$.
- **Primary Metrics**: Validation velocity loss, representation alignment cosine similarity, FAD.

### Experiment 2.3: Intermediate Layer Depth & Projector Capacity

- **Hypothesis**: The student layer must have sufficient depth to extract acoustic abstractions before alignment, while the teacher layer must be deep enough to encode contextual semantics.
- **Configuration**:
  - `E2.3-L2-L4`: Student Layer 2, Teacher Layer 4 (Current default on 6-layer backbone).
  - `E2.3-L3-L5`: Student Layer 3, Teacher Layer 5 (Deeper student abstraction).
  - `E2.3-Proj-Linear`: Single linear projection ($1024 \to 1024$).
  - `E2.3-Proj-MLP2`: 2-layer MLP with SiLU bottleneck ($1024 \to 1024 \to 1024$, Current).
  - `E2.3-Proj-MLP3`: 3-layer MLP with LayerNorm ($1024 \to 2048 \to 1024$).
- **Primary Metrics**: Cosine distance, representation rank collapse diagnostics, synthesis naturalness.

### Experiment 2.4: Loss Weight $\lambda_{\text{self\_flow}}$ Sensitivity

- **Configuration**:
  - Evaluate $\lambda_{\text{self\_flow}} \in \{0.0, 0.2, 0.5, 0.8, 1.2, 2.0\}$.
- **Primary Metrics**: Optimal trade-off curve between velocity prediction error and speaker/content representation fidelity.

---

## 4. Group 3: Model Architecture & Backbone Modernization

### Experiment 3.1: AdaLN Block Decomposition (Decoupled vs. Monolithic)

- **Hypothesis**: In `LYNXNet2AdaLNBlock`, the depthwise conv and the SwiGLU MLP are fused inside a single residual connection and modulated by a single AdaLN scale $\alpha$. Decoupling into two distinct sub-blocks (Conv Sub-block and FFN Sub-block), each with its own LayerNorm, AdaLN modulation, and residual connection (DiT / Conformer standard), enhances gradient flow and feature expressiveness.
- **Configuration**:
  - `E3.1-Fused`: Current monolithic block (`x -> AdaLN -> ConvGate -> MLP -> + res * alpha`).
  - `E3.1-Decoupled`: Decoupled architecture:
    1. $x_1 = x + \alpha_1 \cdot \text{SpatialGatedConv}(\text{AdaLN}_1(x))$
    2. $x_2 = x_1 + \alpha_2 \cdot \text{SwiGLU\_MLP}(\text{AdaLN}_2(x_1))$
- **Primary Metrics**: Convergence speed (steps to reach Mel MSE $< 0.05$), gradient norm stability.

### Experiment 3.2: Receptive Field & Temporal Attention Integration

- **Hypothesis**: Pure dilated depthwise convolutions cannot model long-range melodic phrasing and phrase-level dynamics in singing voice conversion. Incorporating Rotary Position Embedding (RoPE) Multi-Head Self-Attention layers provides global context.
- **Configuration**:
  - `E3.2-ConvOnly`: 6 layers dilated conv (dilation $1, 2, 4, 8, 1, 2$, Current).
  - `E3.2-DeepConv`: 10 layers dilated conv (dilation $1, 2, 4, 8, 1, 2, 4, 8, 1, 2$).
  - `E3.2-HybridConformer`: 6 layers alternating Depthwise Conv and RoPE Multi-Head Self-Attention (4 heads, head dim 64).
- **Primary Metrics**: Long-phrase F0 contour error, V/UV transition accuracy, inference RTF.

### Experiment 3.3: Spatial Gating Activation Function

- **Hypothesis**: The current backbone uses `x = v * torch.atan(gate)`. While $\arctan$ is bounded $(-\pi/2, \pi/2)$ and smooth at 0, standard gating functions like $\text{SiLU}$ (Swish) or $\text{Gated Linear Units (GLU)}$ maintain non-saturating gradients for positive activations.
- **Configuration**:
  - `E3.3-Atan`: `v * torch.atan(gate)` (Current).
  - `E3.3-SiLU`: `v * F.silu(gate)`.
  - `E3.3-GLU`: `v * torch.sigmoid(gate)` (WaveNet standard).
  - `E3.3-Bilinear`: `v * gate` (Linear gating with SwiGLU FFN).
- **Primary Metrics**: Mel-L1, high-frequency harmonic reconstruction, gradient vanishing in early layers.

---

## 5. Group 4: Inference Solvers & Computational Efficiency

### Experiment 4.1: ODE Numerical Solver Comparison & Step Scaling

- **Hypothesis**: The 1st-order Euler method requires $50\text{--}100$ steps to avoid phase and spectral blur. The 2nd-order Midpoint (RK2) method provides a superior accuracy-speed trade-off, achieving perceptual parity with 4th-order RK4 at significantly fewer function evaluations (NFE).
- **Configuration**:
  - Solvers: `euler`, `rk2` (Midpoint), `rk4`.
  - Steps: $N \in \{5, 10, 15, 25, 50\}$.
  - Total NFEs evaluated:
    - Euler: $N$
    - RK2: $2N$
    - RK4: $4N$
- **Primary Metrics**: Mel-L1 vs. Total NFE curve, perceptual audio quality (MOS / blinded preference), RTF.

### Experiment 4.2: Batched Classifier-Free Guidance (CFG)

- **Hypothesis**: Current inference computes conditional velocity $v_{\text{cond}}$ and unconditional velocity $v_{\text{uncond}}$ sequentially. Stacking them into a single forward pass with batch size $2B$ eliminates half the Python kernel launches and maximizes GPU tensor core occupancy.
- **Configuration**:
  - `E4.2-Sequential`: Sequential evaluation (Current).
  - `E4.2-Batched`: Batched evaluation ($x_{\text{cat}} = [x; x]$, single model forward call).
- **Primary Metrics**: Inference latency (ms), RTF on CPU and CUDA.

### Experiment 4.3: Unexpanded AdaLN Projection

- **Hypothesis**: Expanding condition embeddings to $[B, T, 512]$ before computing `film_proj` wastes FLOPs by repeating the linear layer across $T$ frames. Computing `film_proj` on $[B, 1, 512]$ and broadcasting $[B, 1, 3072]$ saves $O(B \times T \times D)$ computations in every layer.
- **Configuration**:
  - `E4.3-Expanded`: Pre-expansion to $[B, T, 512]$ (Current).
  - `E4.3-Broadcast`: Projection on $[B, 1, 512]$ with spatial broadcasting.
- **Primary Metrics**: Forward pass execution time (ms), GPU VRAM footprint during inference.

---

## 6. Summary Matrix of Proposed Ablation Runs

| ID | Category | Key Modification | Baseline Value | Experimental Value | Primary Metric |
|---|---|---|---|---|---|
| **E1.1** | Math | DDSP Prior Detach | No (`ddsp_mel`) | Yes (`ddsp_mel.detach()`) | Mel-L1 ($t_{\text{start}}=0.4$) |
| **E1.2** | Math | Dynamic Range Compression | `clip_val=1e-5` | `1e-4` / Float32 | NaN Frequency / Scaler Drops |
| **E1.3** | Optimizer | Depthwise Conv Routing | Captured in Muon | Routed to AdamW | Validation Velocity Loss |
| **E2.1** | Self-Flow | Noise Mask Topology | i.i.d. Bernoulli (0.5) | Phonetic Span ($L=5, 12$) | CER / Phone Error |
| **E2.2** | Self-Flow | Representation Target | All Tokens | Masked Tokens Only | Cosine Alignment Score |
| **E2.3** | Self-Flow | Student/Teacher Layers | L2 / L4 | L3 / L5 | Feature Diversity Rank |
| **E3.1** | Backbone | AdaLN Block Structure | Fused Monolithic | Decoupled Conv + FFN | Training Steps to Convergence |
| **E3.2** | Backbone | Temporal Mixing | Pure Dilated Conv | Hybrid RoPE Attention | Long-Phrase F0 RMSE |
| **E3.3** | Backbone | Gating Activation | $\arctan(\cdot)$ | $\text{SiLU} / \text{Sigmoid}$ | High-Freq Spectral Loss |
| **E4.1** | Inference | Numerical Solver | Euler (50 steps) | RK2 / Midpoint (15 steps) | RTF vs. Mel-L1 Pareto Curve |
| **E4.2** | Inference | CFG Dispatch | Sequential | Batched ($2B$) | Inference Latency (ms) |
| **E4.3** | Performance| AdaLN Projection | Pre-expanded $[B, T]$ | Broadcast $[B, 1]$ | Inference Memory & Speed |

---

## 7. Execution Recipe: Configuration Templates

### Baseline Config (`configs/ablation_base.yaml`)
```yaml
model:
  type: 'RectifiedFlow'
  win_length: 2048
  n_layers: 6
  n_chans: 1024
  use_self_flow: false
  use_f0_conditioning: true
  use_aux_f0: true

train:
  optimizer: 'muon'
  lr: 0.0005
  batch_size: 48
  lambda_ddsp: 1.0
  lambda_f0: 0.1
  lambda_self_flow: 0.0
```

### Self-Flow Span-Masking Config (`configs/ablation_self_flow_span.yaml`)
```yaml
model:
  type: 'RectifiedFlow'
  win_length: 2048
  n_layers: 6
  n_chans: 1024
  use_self_flow: true
  self_flow_mask_ratio: 0.5
  self_flow_span_length: 8
  self_flow_student_layer: 2
  self_flow_teacher_layer: 4
  self_flow_projector_dim: 1024

train:
  optimizer: 'muon'
  lr: 0.0005
  lambda_ddsp: 1.0
  lambda_f0: 0.1
  lambda_self_flow: 0.8
```
