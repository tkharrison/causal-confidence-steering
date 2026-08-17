# Results

`main/full_analysis.json` is the authoritative machine-readable result. It contains:

- integrity checks;
- cell means and standard deviations;
- paired `alpha 15 - alpha 0` changes with 95% confidence intervals;
- itemwise dose slopes and condition contrasts;
- Kumaran-style hard confidence classes; and
- binary label-order robustness checks.

`main/analysis-long.csv` is a tidy 6,000-row scalar export. It does not contain the
full candidate-token logits or every replay field from the raw measurements.

The raw measurement archive is retained locally and identified by:

```text
causal-confidence-steering-v1.0-artifacts.tar.gz
e7b26555193ec9d4d9377411d7d1d7c85cdf83e6b701bb0f9489ff9cb84de310
```

It is not included in the repository or a GitHub Release pending confirmation that
the embedded dataset-derived stimulus text may be redistributed.

The `validation/` directory records input validation, CPU runner validation, causal
steering validation, the dose-selection rationale, and the frozen experimental
runbook.
