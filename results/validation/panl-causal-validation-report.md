# Qwen2.5-7B PANL causal steering validation

## Outcome

The `Likely - Unlikely` direction causally and dose-dependently changes Qwen2.5-7B-
Instruct's categorical confidence on held-out TriviaQA trials. The strongest layer is
15, followed by layer 16. Positive alpha raises confidence and negative alpha lowers
it while the question, generated answer, and confidence prompt remain fixed.

This validates the main causal steering effect in this implementation. It does not
show that PANL is the only token from which the direction can affect confidence.

## Leakage and determinism safeguards

- Direction construction: 25 adjudicated-correct `Unlikely` plus 25 adjudicated-
  correct `Likely` trials, seed 42.
- All 50 construction IDs were automatically excluded from intervention trials.
- The 32-trial layer screen and 150-trial confirmation sets have zero overlap.
- Greedy/eval-mode Qwen2.5-7B-Instruct at pinned revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Exact saved question-answer prefixes were replayed. Answers were never regenerated
  during intervention.
- The alpha-zero comparison is computed in the same right-padded bfloat16 batch as
  intervention conditions. Its maximum absolute measured change was exactly 0.
- All intervention outputs retained a valid categorical confidence token as the
  global next-token argmax.

## All-layer screen

The screen used 32 held-out trials, all 28 layers, and alpha `-5, 0, 5`. The top
positive-minus-negative expected-confidence separations were:

| Layer | Mean separation | Questions with positive separation |
|---:|---:|---:|
| 15 | 0.1155 | 32/32 |
| 16 | 0.0982 | 32/32 |
| 17 | 0.0632 | 31/32 |
| 13 | 0.0541 | 30/32 |
| 14 | 0.0446 | 30/32 |

The screen was used only to select layers 13-17; its trials were not reused in the
confirmation set.

## Independent 150-trial confirmation

Mean change in class-restricted expected confidence relative to the same-batch clean
condition is shown below. Approximate 95% confidence intervals are trial-level mean
intervals.

| Layer | alpha -5 | alpha -2 | alpha +2 | alpha +5 | +5 minus -5 | Positive pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | -0.0331 | -0.0136 | +0.0124 | +0.0296 | 0.0627 +/- 0.0161 | 143/150 |
| 14 | -0.0276 | -0.0108 | +0.0104 | +0.0244 | 0.0520 +/- 0.0138 | 147/150 |
| 15 | -0.0673 | -0.0268 | +0.0257 | +0.0611 | 0.1284 +/- 0.0243 | 149/150 |
| 16 | -0.0607 | -0.0211 | +0.0187 | +0.0420 | 0.1027 +/- 0.0216 | 149/150 |
| 17 | -0.0392 | -0.0156 | +0.0143 | +0.0296 | 0.0688 +/- 0.0162 | 146/150 |

Across the five selected layers, the mean dose curve was:

| alpha | Mean confidence change |
|---:|---:|
| -5 | -0.0456 |
| -2 | -0.0176 |
| 0 | 0.0000 |
| +2 | +0.0163 |
| +5 | +0.0373 |

## Token-location controls

The same PANL-derived vector was injected at other token locations for the same 64-
question subset and the same layers. Values are +5-minus--5 confidence separation.

| Layer | PANL | PANL+1 | Last answer token | Confidence colon |
|---:|---:|---:|---:|---:|
| 13 | +0.0652 | -0.0018 | +0.0533 | +0.0206 |
| 14 | +0.0515 | -0.0003 | +0.0377 | +0.0013 |
| 15 | +0.1446 | +0.0061 | +0.0866 | +0.0143 |
| 16 | +0.1212 | -0.0007 | +0.0515 | -0.0191 |
| 17 | +0.0827 | -0.0004 | +0.0487 | -0.0267 |

Interpretation:

- PANL is the strongest and most consistent tested location.
- PANL+1 is a strong negative control: the effect is approximately zero.
- The last answer token retains a smaller but consistent effect. Thus, the direction
  is not exclusively causal at PANL.
- Applying a PANL-derived vector at the final confidence colon is weak and layer-
  dependent, including sign reversals at layers 16-17.

## Artifacts

- Direction: `work/qwen25-7b-panl-direction-v1.safetensors`
- Screen: `work/qwen25-7b-panl-steering-screen-v1.jsonl`
- Confirmation: `work/qwen25-7b-panl-steering-confirm-v1.jsonl`
- PANL+1 control: `work/qwen25-7b-panl-vector-at-panl-plus-1-control-v1.jsonl`
- Last-answer control: `work/qwen25-7b-panl-vector-at-last-answer-control-v1.jsonl`
- Confidence-colon control:
  `work/qwen25-7b-panl-vector-at-confidence-colon-control-v1.jsonl`

All remote copies are retained in the `qwen-confidence-results` Modal Volume under
`/interventions/`.
