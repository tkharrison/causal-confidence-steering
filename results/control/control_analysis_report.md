# Post-hoc confidence-direction control extension

## Conservative reanalysis of the original primary outcomes

Changes are exact item-level alarm log-odds at alpha 15 minus alpha 0.

| Measure | correct change | false change | false - correct | ambiguous change | ambiguous - correct |
|---|---:|---:|---:|---:|---:|
| anomaly_forced_choice | -0.410 [-0.589, -0.231] | -4.771 [-5.287, -4.256] | -4.361 [-4.907, -3.816] | -1.641 [-1.932, -1.350] | -1.231 [-1.573, -0.890] |
| error_detection | -1.286 [-1.551, -1.021] | -7.350 [-8.205, -6.495] | -6.064 [-6.959, -5.169] | -4.211 [-4.703, -3.719] | -2.925 [-3.484, -2.366] |

## Negative-alpha sweep on definite-correct answers

Cell values for confidence are expected categorical midpoints. Alarm values are exact candidate log-odds,
calculated as the alarm-token logit minus the reassuring-token logit.

| Measure | alpha -15 | -10 | -5 | 0 | 5 | 10 | 15 | slope per alpha | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| confidence_manipulation_check | 0.584 | 0.704 | 0.741 | 0.756 | 0.767 | 0.774 | 0.778 | +0.005 | [+0.004, +0.006] |
| anomaly_forced_choice | 5.791 | 2.295 | -1.843 | -3.993 | -4.746 | -4.834 | -4.402 | -0.341 | [-0.357, -0.325] |
| error_detection | 7.249 | 0.901 | -5.120 | -7.709 | -8.668 | -8.988 | -8.995 | -0.515 | [-0.541, -0.488] |

## Matched yes/no polarity control on definite-false answers

| Probe | alarm response | alarm probability change | alarm log-odds change | yes log-odds change |
|---|---|---:|---:|---:|
| inconsistency_yes_no | yes | -0.544 [-0.601, -0.488] | -3.868 [-4.213, -3.522] | -3.868 [-4.213, -3.522] |
| consistency_yes_no | no | -0.147 [-0.200, -0.094] | -2.611 [-3.023, -2.200] | +2.611 [+2.200, +3.023] |

The critical literal-response-bias test is the consistency-minus-inconsistency interaction
in yes log-odds: +6.479 [+5.841, +7.116].
