# Qwen PANL confidence-introspection experiment: full results

## Integrity

All integrity checks passed: 6,000 unique measurement rows, 60 complete cells, 100 observations per cell, no parse failures, PANL token 198 throughout, and exactly one intervention application at every positive alpha.

All outcomes below are oriented so larger values mean more confidence, more perceived inconsistency/unusualness, more error detection, or more abstention. Confidence intervals are unadjusted 95% normal-approximation intervals over independent item-level paired quantities; error detection and abstention are secondary outcomes.

## Cell means

### Expected confidence

| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| definite correct | 0.756 | 0.767 | 0.774 | 0.778 | +0.022 | [+0.018, +0.026] |
| definite false | 0.507 | 0.615 | 0.675 | 0.705 | +0.198 | [+0.168, +0.229] |
| ambiguous | 0.570 | 0.591 | 0.603 | 0.612 | +0.043 | [+0.029, +0.056] |

### P(inconsistent)

| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| definite correct | 0.184 | 0.106 | 0.098 | 0.149 | -0.036 | [-0.062, -0.010] |
| definite false | 0.832 | 0.614 | 0.453 | 0.423 | -0.409 | [-0.484, -0.334] |
| ambiguous | 0.611 | 0.451 | 0.402 | 0.427 | -0.183 | [-0.229, -0.138] |

### Unusualness score

| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| definite correct | 0.310 | 0.279 | 0.256 | 0.234 | -0.076 | [-0.117, -0.036] |
| definite false | 0.593 | 0.483 | 0.419 | 0.415 | -0.177 | [-0.230, -0.125] |
| ambiguous | 0.426 | 0.356 | 0.324 | 0.366 | -0.059 | [-0.101, -0.018] |

### P(incorrect)

| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| definite correct | 0.020 | 0.012 | 0.010 | 0.010 | -0.010 | [-0.017, -0.003] |
| definite false | 0.845 | 0.637 | 0.435 | 0.345 | -0.500 | [-0.583, -0.416] |
| ambiguous | 0.600 | 0.322 | 0.216 | 0.228 | -0.373 | [-0.442, -0.303] |

### P(abstain)

| Condition | alpha 0 | alpha 5 | alpha 10 | alpha 15 | paired change 15-0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| definite correct | 0.403 | 0.374 | 0.368 | 0.387 | -0.016 | [-0.039, +0.008] |
| definite false | 0.949 | 0.888 | 0.850 | 0.836 | -0.113 | [-0.163, -0.064] |
| ambiguous | 0.862 | 0.796 | 0.780 | 0.800 | -0.062 | [-0.093, -0.031] |

## Preregistered dose-response contrasts

The table reports how much more or less the outcome changed per +5 alpha in each epistemically problematic condition relative to definite-correct controls.

| Outcome | Contrast | difference per +5 alpha | 95% CI |
|---|---|---:|---:|
| Expected confidence | definite false - definite correct | +0.058 | [+0.048, +0.069] |
| Expected confidence | ambiguous - definite correct | +0.007 | [+0.002, +0.011] |
| P(inconsistent) | definite false - definite correct | -0.127 | [-0.154, -0.100] |
| P(inconsistent) | ambiguous - definite correct | -0.048 | [-0.065, -0.032] |
| Unusualness score | definite false - definite correct | -0.034 | [-0.056, -0.012] |
| Unusualness score | ambiguous - definite correct | +0.004 | [-0.014, +0.023] |
| P(incorrect) | definite false - definite correct | -0.167 | [-0.196, -0.138] |
| P(incorrect) | ambiguous - definite correct | -0.119 | [-0.142, -0.096] |
| P(abstain) | definite false - definite correct | -0.032 | [-0.051, -0.014] |
| P(abstain) | ambiguous - definite correct | -0.015 | [-0.028, -0.002] |

## Kumaran-style hard confidence classes

| Condition | hard midpoint alpha 0 | hard midpoint alpha 15 | paired change | 95% CI | class changed |
|---|---:|---:|---:|---:|---:|
| definite correct | 0.751 | 0.798 | +0.047 | [+0.029, +0.065] | 24/100 |
| definite false | 0.469 | 0.714 | +0.245 | [+0.194, +0.296] | 69/100 |
| ambiguous | 0.551 | 0.600 | +0.049 | [+0.027, +0.071] | 30/100 |

## Binary label-order audit

Binary response order was fixed within item across alpha. The table stratifies the paired alpha-15-minus-alpha-0 change by whether the target response appeared as A or B.

| Outcome | Condition | target=A | target=B |
|---|---|---:|---:|
| P(inconsistent) | definite correct | -0.061 (n=52) | -0.008 (n=48) |
| P(inconsistent) | definite false | -0.210 (n=52) | -0.625 (n=48) |
| P(inconsistent) | ambiguous | -0.162 (n=56) | -0.211 (n=44) |
| P(incorrect) | definite correct | -0.016 (n=53) | -0.002 (n=47) |
| P(incorrect) | definite false | -0.544 (n=36) | -0.475 (n=64) |
| P(incorrect) | ambiguous | -0.440 (n=53) | -0.296 (n=47) |
| P(abstain) | definite correct | -0.013 (n=48) | -0.018 (n=52) |
| P(abstain) | definite false | -0.013 (n=61) | -0.271 (n=39) |
| P(abstain) | ambiguous | +0.000 (n=45) | -0.113 (n=55) |

## Interpretation

The confidence manipulation succeeded. It was strongest for definite-false answers, moderate for ambiguous answers, and small for already-high-confidence correct answers.

The primary provenance-anomaly prediction was not supported. Increasing the high-confidence PANL direction reduced rather than increased forced-choice inconsistency reports. Relative to definite-correct controls, the per-+5-alpha slope difference was -0.127 for definite-false items and -0.048 for ambiguous items. Both confidence intervals excluded zero in the negative direction.

The continuous unusualness measure converged for definite-false answers: its dose response was -0.034 per +5 alpha relative to correct controls. The ambiguous-minus-correct continuous contrast was +0.004, with a confidence interval that included zero; it therefore supplied no evidence for an ambiguity-specific anomaly increase.

Error detection and abstention moved in the same direction: stronger injected confidence made the model less likely to call the answer incorrect and less likely to abstain. Yet at alpha 15, false and ambiguous answers still had higher absolute anomaly, error, and abstention scores than correct controls. The model retained sensitivity to question-answer fit, but the injected confidence suppressed that sensitivity instead of becoming an independently detected anomaly.

The most direct interpretation is that Qwen reads the steered confidence state as evidence about answer quality; it does not show behavioral access to the intervention's provenance. A narrower mechanistic alternative remains: the PANL direction may directly bias several downstream metacognitive reports, not only a unitary subjective confidence state. Control-position and unrelated-direction interventions would distinguish these explanations.

The primary forced-choice anomaly conclusion survived both A/B response orders: false and ambiguous anomaly reports decreased in both strata. The abstention effect was notably label-order sensitive, especially in the target=A stratum, so abstention should be treated as weaker secondary evidence rather than a result of the same strength as anomaly or error detection.
