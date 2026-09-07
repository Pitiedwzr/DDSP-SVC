# Implementation log

## 2026-09-07 — Architecture, stability, and optimization review

Following an exhaustive audit of mathematical foundations, numerical stability, optimization correctness, and backbone design, six atomic improvements were implemented:

- `98b5923` — **Fix Muon depthwise conv selection and Aurora dimension scaling**:
  - Excluded depthwise 1D convolutions (`groups == in_channels > 1`) and embedding matrices from Muon optimization in `optimizer/muon.py`, safely routing them to AdamW to prevent orthogonalization degeneration.
  - Corrected dimension scaling factor in `optimizer/aurora.py` from aspect ratio `sqrt(M / N)` to scale `sqrt(max(M, N))` according to the Aurora algorithm specification, preventing exploding updates on asymmetric tensors. Added safe bfloat16/float32 fallbacks.
  - Added unit tests in `tests/test_optimizers.py`.
- `37ced20` — **Fix FP16 STFT gradient overflow, clamp norm_spec, and align infer mask**:
  - Clamped nvSTFT minimum magnitude floor to `1e-4` in `float16` (`1e-5` in `float32`) in `nsf_hifigan/nvSTFT.py`, eliminating massive backward gradient spikes ($100{,}000 > 65{,}504$) that trigger FP16 underflow/overflow NaN cascades.
  - Clamped `norm_spec` and `denorm_spec` in `reflow/reflow.py` to prevent spectrogram values from escaping the normalized dynamic range $[-1, 1]$.
  - Aligned padding mask slice in `batch_infer.py` to match exact output time length.
  - Added unit tests in `tests/test_numerical_stability.py`.
- `6d04bd3` — **Add configurable DDSP acoustic prior condition detachment**:
  - Added `detach_ddsp_cond` option to `configs/reflow.yaml`, `reflow/vocoder.py`, and `train_reflow.py`.
  - When enabled (`true` by default), detaches DDSP output before concatenating as Reflow conditioner, isolating acoustic prior learning and preventing DDSP collapse to trivial mel surrogates.
  - Added unit tests in `tests/test_detach_ddsp.py`.
- `b7ad92a` — **Optimize AdaLN projection compute and batch CFG forward passes**:
  - In `reflow/lynxnet2adaln.py`, avoided redundant 3D temporal expansion of `block_cond` before the AdaLN linear projection, reducing linear projection FLOPs by up to $1000\times$ across all layers during inference while preserving exact broadcast semantics.
  - In `reflow/reflow.py`, batched conditional and unconditional forward passes into a single $2B$ batch in `_get_velocity()` for Euler, RK2, and RK4 ODE samplers with classifier-free guidance (CFG).
  - Added unit tests in `tests/test_performance_optimizations.py`.
- `31523ae` — **Add configurable Self-Flow phonetic span masking and selective representation loss**:
  - In `reflow/reflow.py`, implemented contiguous phoneme span masking (`self_flow_span_length`) to avoid trivial single-frame acoustic interpolation.
  - Added `self_flow_loss_on_masked_only` option to focus self-supervised representation loss solely on masked tokens.
  - Exposed options via `configs/reflow.yaml`, `reflow/vocoder.py`, and `train_reflow.py`.
  - Added unit tests in `tests/test_self_flow.py`.
- `5dd0af4` — **Add configurable decoupled backbone architecture and gating activations**:
  - Added `block_type: 'fused'` (default) vs `'decoupled'` in `reflow/lynxnet2adaln.py`, isolating Depthwise-Conv spatial mixing and SwiGLU feedforward channel mixing with independent AdaLN conditioning and residual connections.
  - Added configurable `gating_act`: `'atan'` (default bounded gating), `'silu'`, and `'glu'` (sigmoid gating).
  - Exposed options via `configs/reflow.yaml`, `reflow/vocoder.py`, and `train_reflow.py` with full backward compatibility for existing checkpoints.
  - Added unit tests in `tests/test_backbone_improvements.py`.

A comprehensive ablation experiment plan for evaluating these features was designed and saved in `ABLATION_EXPERIMENTS.md`.

## 2026-08-11 — Whole-project correctness audit

The audit findings were fixed as separate commits:

- `66825d3` — fixed `DotDict` attribute semantics, added the missing
  `accelerate` dependency, made the primary optimizer/scheduler explicit, and
  replaced the finite `OneCycleLR` schedule with warmup plus cosine decay that
  honors `eta_min` and remains at the minimum learning rate after `t_max`.
- `09f0a00` — made saved configurations safe-YAML compatible and removed
  `AveragedModel.n_averaged` before loading EMA weights into the base inference
  model.
- `2f46eaf` — preserved a separate voiced/unvoiced mask while retaining
  interpolated F0 for synthesis. The mask now reaches DDSP pitch conditioning
  and the auxiliary F0 loss in preprocessing, training, validation, CLI, batch,
  and GUI paths.
- `2278bb1` — made preprocessing fail on vocoder/data parameter mismatches,
  missing input audio, invalid worker counts, or any failed file instead of
  leaving an incomplete dataset that fails later during training.
- `512ffa6` — padded the resampled encoder audio rather than the source-rate
  tensor and preserved all batches during unit-frame alignment.
- `9931398` — rejected non-positive inference step counts, made speaker-mix
  parsing safe and validated, honored the GUI speaker-mix toggle, fixed GUI
  output alignment, and replaced executable pickle GUI configurations with
  JSON.
- `9d03da2` — rebuilt ONNX export around the current runtime architecture. The
  exporter now loads current standard/EMA checkpoints, copies exact DDSP
  control weights, includes the shared unit encoder and Reflow speaker
  conditioning, exports voiced/formant inputs, supports single- and
  multi-speaker models, and validates conditioner outputs with ONNX Runtime.
- `d6c311e` — zero-initialized both the weight and bias of the Reflow velocity
  output projection.

Compatibility and migration notes:

- Existing preprocessed datasets still load, but their interpolated F0 cannot
  reconstruct historical voiced/unvoiced boundaries. Re-run `preprocess.py` to
  create the new `voiced/` arrays before enabling explicit F0 conditioning or
  relying on voiced-only auxiliary F0 loss.
- GUI settings are now stored as `config.json`. Unsafe legacy `config.pkl`
  files are deliberately not loaded automatically.
- ONNX export is now invoked with
  `python export_onnx.py -m <checkpoint.pt> -o <output_directory>` and requires
  the newly listed `onnx` and `onnxruntime` dependencies.
- The untracked `patch_lynxnet.py` helper is an obsolete channel-layout rewrite
  experiment. It is not required by the current model or ONNX exporter and was
  intentionally left uncommitted.

Verification added during this audit includes safe checkpoint serialization,
EMA base-model loading, F0/voicing interpolation, short-audio padding,
multi-batch unit alignment, inference validation, safe speaker-mix parsing,
ONNX/runtime state-key compatibility, Self-Flow backpropagation, EMA resume,
scalar/token timestep shapes, and zero output initialization.

Final verification completed on 2026-08-11:

- All tracked Python sources compiled successfully.
- All 13 unit tests passed.
- `git diff --check` passed.
- Full audio inference and end-to-end ONNX export were not run because no
  pretrained checkpoint or test dataset is included in the workspace.

## Earlier correctness and experiment work

Implemented the correctness fixes and made the F0 experiments configurable.

  Changes:

  - Fixed temporal attention by enabling batch_first=True.
  - Fixed nonzero t_start inference to initialize from a noised DDSP mel.
  - Changed CFG dropout to be per-sample and retain content/DDSP conditioning.
  - Bounded DDSP log magnitudes before exp, preventing FP16 overflow.
  - Fully unvoiced clips are now saved instead of silently moved/skipped.
  - Fixed missing configuration defaults for spec_min/spec_max; the custom DotDict prevented normal getattr(...,
    default) behavior.

  - Added configurable auxiliary F0 regularization:
      - model.use_aux_f0: true
      - train.lambda_f0: 0.1

  - Added configurable explicit DDSP pitch conditioning:
      - model.use_f0_conditioning: false
      - Uses normalized log-F0 plus voiced/unvoiced state when enabled.

  - Auxiliary F0 loss ignores unvoiced frames and remains training-only.

  Important compatibility note: enabling use_f0_conditioning adds parameters and therefore requires a new/retrained
  checkpoint. Disabling use_aux_f0 removes the predictor parameters. The provided defaults preserve the current
  checkpoint architecture.

  Validation completed:

  - uv run python -m compileall ...: passed
  - AST parsing: passed
  - git diff --check: passed

  A full tensor/runtime test was not possible because the current uv environment does not have PyTorch installed, though
  Python itself works correctly through uv run. Existing unrelated untracked files were left untouched.
  
• Implemented the Self-Flow training path and EMA fix.

  Key changes:

  - EMA state now restores correctly when resuming training.
  - Added per-frame Dual-Timestep Scheduling.
  - Added EMA-teacher representation alignment.
  - Added configurable student/teacher intermediate layers and projection head.
  - Added student-only DDSP condition masking using a learned mask token.
  - Preserved the existing homogeneous inference path.
  - Skips the teacher pass when lambda_self_flow: 0.
  - Added configuration validation for masking ratios and layer selection.

  Configuration is in /D:/IdeaProjects/DDSP-SVC/configs/reflow.yaml:35. It remains disabled by default for checkpoint
  compatibility:

  model:
    use_self_flow: false
    self_flow_mask_ratio: 0.5
    self_flow_condition_mask_ratio: 0.1
    self_flow_student_layer: 2
    self_flow_teacher_layer: 4
    self_flow_projector_dim: 1024

  train:
    lambda_self_flow: 0.8

  Set use_self_flow: true for a new training run.
  
Review:
Audited the tracked Python/config paths. I found several concrete implementation defects; no files were modified.

  ## Critical findings

  1. Generated checkpoints cannot be loaded because their saved YAML is invalid for safe_load. Training replaces
     args.device with a torch.device at /D:/IdeaProjects/DDSP-SVC/train_reflow.py:35, then serializes it using yaml.dump
     at /D:/IdeaProjects/DDSP-SVC/logger/saver.py:46. This emits a Python-specific !!python/object/apply:torch.device
     tag, which the inference loader rejects at /D:/IdeaProjects/DDSP-SVC/reflow/vocoder.py:28. Reproduced locally.

  2. EMA checkpoints fail inference loading. Every current checkpoint saves AveragedModel.state_dict() at /D:/
     IdeaProjects/DDSP-SVC/logger/saver.py:129. The inference loader strips module. but retains n_averaged, then
     strictly loads it into the base model at /D:/IdeaProjects/DDSP-SVC/reflow/vocoder.py:67. PyTorch rejects that
     unexpected key. Reproduced with an isolated AveragedModel.

  3. The primary training configuration crashes before training. DotDict.__getattr__ returns None for missing keys
     instead of raising AttributeError at /D:/IdeaProjects/DDSP-SVC/logger/utils.py:49, so defaults supplied to
     getattr() never work. Consequently /D:/IdeaProjects/DDSP-SVC/train_reflow.py:78 calls .lower() on None, because /
     D:/IdeaProjects/DDSP-SVC/configs/reflow.yaml:54 defines neither optimizer nor lr_scheduler. The reproduction
     returned optimizer=None and scheduler=None.

  4. A documented clean installation also lacks accelerate. It is imported unconditionally at /D:/IdeaProjects/DDSP-SVC/
     train_reflow.py:11 but absent from /D:/IdeaProjects/DDSP-SVC/requirements.txt:1.

  ## High-impact findings

  5. ONNX export is currently nonfunctional.
      - Its duplicated model architecture does not match Unit2Wav, yet checkpoint loading is strict at /D:/IdeaProjects/
        DDSP-SVC/export_onnx.py:682.

      - The default single-speaker branch uses undefined test_sid at /D:/IdeaProjects/DDSP-SVC/export_onnx.py:738.
      - Velocity export supplies three arguments at /D:/IdeaProjects/DDSP-SVC/export_onnx.py:758, while the current
        backbone requires global_cond at /D:/IdeaProjects/DDSP-SVC/reflow/lynxnet2adaln.py:156.

  6. Voiced/unvoiced conditioning and auxiliary F0 masking do not receive real UV information. Preprocessing
     interpolates every unvoiced gap at /D:/IdeaProjects/DDSP-SVC/preprocess.py:174, while every inference path requests
     uv_interp=True; that subsequently raises zero F0 to f0_min at /D:/IdeaProjects/DDSP-SVC/ddsp/vocoder.py:138.
     Therefore checks such as f0 > 0 in /D:/IdeaProjects/DDSP-SVC/ddsp/unit2control.py:85 and /D:/IdeaProjects/DDSP-SVC/
     reflow/vocoder.py:314 classify essentially every frame as voiced.

  7. Preprocessing can report completion while producing an unusable dataset. A vocoder sample-rate/hop mismatch merely
     disables mel extraction at /D:/IdeaProjects/DDSP-SVC/preprocess.py:61, although the loader unconditionally requires
     both mel files at /D:/IdeaProjects/DDSP-SVC/reflow/data_loaders.py:169. Per-file exceptions are also printed and
     swallowed at /D:/IdeaProjects/DDSP-SVC/preprocess.py:240.

  8. The “cosine” scheduler does not implement its configuration contract. eta_min is read but unused at /D:/
     IdeaProjects/DDSP-SVC/train_reflow.py:126. OneCycleLR instead derives another final LR and raises once training
     exceeds total_steps=t_max; it does not remain at eta_min.

  ## Additional defects

  - Short resampled audio pads the original audio rather than the resampled tensor at /D:/IdeaProjects/DDSP-SVC/ddsp/
    vocoder.py:203. Batch alignment also hard-codes output batch size one at /D:/IdeaProjects/DDSP-SVC/ddsp/
    vocoder.py:211.

  - The GUI computes a conditional speaker-mix variable but passes the original dictionary regardless of whether mixing
    is enabled at /D:/IdeaProjects/DDSP-SVC/gui_reflow.py:123. Its audio-alignment slice is also discarded at /D:/
    IdeaProjects/DDSP-SVC/gui_reflow.py:148.

  - CLI inference accepts infer_step=0, then divides by zero at /D:/IdeaProjects/DDSP-SVC/reflow/reflow.py:200.
  - Speaker-mix text is executed using eval at /D:/IdeaProjects/DDSP-SVC/gui_reflow.py:328, allowing arbitrary Python
    execution.

  ## Verification

  - All tracked Python components compiled successfully.
  - All three existing self-flow/EMA unit tests passed in an isolated dependency environment.
  - The whole workspace compilation additionally found a syntax error in the untracked /D:/IdeaProjects/DDSP-SVC/
    patch_lynxnet.py:54.

  - End-to-end audio execution was not possible without the pretrained checkpoints and dataset.
