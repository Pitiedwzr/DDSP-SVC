# Self-Flow experiment plan

This document keeps Self-Flow, auxiliary F0, and backbone scaling as separate
variables. Use the same data split, seed, optimizer, batch audio duration, and
validation noise seeds for every comparison.

## Primary ablation

| Run | Dual timestep | Representation loss | Condition mask | Auxiliary F0 |
| --- | --- | ---: | ---: | --- |
| Baseline | No | 0.0 | 0.0 | Current setting |
| A | Yes | 0.0 | 0.0 | Same as baseline |
| B | Yes | 0.2 | 0.1 | Same as baseline |
| C | Yes | 0.8 | 0.1 | Same as baseline |
| D | Yes | Best of B/C | 0.25 | Same as baseline |
| E | Best Self-Flow setup | Best Self-Flow setup | Best mask | Off |

Run A measures Dual-Timestep Scheduling without representation alignment. Runs
B and C measure the contribution of the EMA-teacher loss. Run D tests whether
stronger condition masking prevents the Reflow model from copying DDSP mel.
Run E measures interaction with the auxiliary F0 regularizer.

For Run A, set `use_self_flow: true` and `lambda_self_flow: 0.0`. The projection
head and teacher pass are still available, but the training code should skip the
teacher pass when the loss weight is zero for maximum efficiency.

## Backbone scaling

Do not compare backbone sizes until selecting a Self-Flow configuration on the
default backbone. Then run:

| Run | Layers × channels | Student/teacher layers | Purpose |
| --- | --- | --- | --- |
| S0 | 6 × 1024 | 2 / 4 | Current reference |
| S1 | 10 × 1024 | 3 / 7 | Test additional depth/context |
| S2 | 10 × 2048 | 3 / 7 | Test width only after S1 wins clearly |

Dense projections scale approximately with `layers × channels²`. Relative to
6 × 1024, the rough Reflow compute/parameter multipliers are:

- 10 × 1024: about 1.67×
- 10 × 2048: about 6.67×

Self-Flow also adds a no-gradient EMA-teacher Reflow pass. It does not double
activation memory for backward, but it increases wall time and temporary
activation memory. Reduce batch size only if required, and use gradient
accumulation to preserve the effective batch size.

Depth is the preferred first scaling direction: 10 × 1024 provides more stages
for the student/teacher layer separation and a larger temporal receptive field
at much lower cost than doubling width. Move to 10 × 2048 only if S1 shows a
repeatable quality improvement rather than merely a lower training loss.

## Evaluation

At minimum, compare:

- validation Reflow velocity loss;
- mel reconstruction error using fixed validation noise;
- speaker embedding similarity;
- content intelligibility or ASR error;
- F0 correlation/RMSE and voiced/unvoiced error;
- FAD or an audio-embedding distance;
- blinded listening tests for timbre, artifacts, and temporal consistency;
- training throughput, peak VRAM, and inference real-time factor.

Do not select a larger model using training loss alone. A wider model can fit the
flow target better without producing a perceptually meaningful improvement.

## Initial recommended configuration

```yaml
model:
  use_self_flow: true
  self_flow_mask_ratio: 0.5
  self_flow_condition_mask_ratio: 0.1
  self_flow_student_layer: 2
  self_flow_teacher_layer: 4
  self_flow_projector_dim: 1024

train:
  lambda_self_flow: 0.8
```

Keep inference unchanged. Dual timesteps, condition masking, the projection
head, and the teacher are training-only mechanisms.
