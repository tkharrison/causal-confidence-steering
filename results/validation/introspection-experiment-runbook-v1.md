# PANL confidence-introspection experiment runbook

## Frozen primary design

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Stimuli: 100 definite-correct, 100 definite-false, and 100 ambiguous.
- Intervention: layer 15 PANL residual direction at alpha 0, +5, +10, and +15.
- Design: paired within item. Every item is evaluated independently at both alpha
  values; no conversational state is shared.
- Total: 300 stimuli x 4 alpha values x 5 measures = 6,000 measurement rows.

Ambiguous items first receive one deterministic, unsteered forced choice among the
four substantive answers. That answer is frozen and replayed for every intervention
and measure. Definite-correct answers are genuine verified Qwen answers. Definite-
false answers are independently validated plausible substitutions.

Every replay is reconstructed with the same literal prefix used to build the
confidence direction:

```text
You will be shown a question, together with an answer you previously provided before.

Question: <question and four options>
**Answer**: <frozen answer>
```

The final newline is tokenizer ID 198 and is the PANL intervention position. Each
dependent measure is appended after it in a separate forward/generation branch.

## Measures

1. Primary forced-choice anomaly measure: probability assigned to
   `inconsistent` versus `consistent`.
2. Primary continuous anomaly measure: deterministic 0-100 response, analyzed on
   a 0-1 scale.
3. Manipulation check: expected midpoint from the published ten categorical
   confidence-class logits.
4. Secondary error-detection measure: probability assigned to `incorrect`.
5. Secondary behavioral measure: probability assigned to `abstain` rather than
   wager.

Binary response order is deterministically counterbalanced by item and measure and
held fixed across all alpha levels.

## Primary comparisons

- Verify a positive within-item dose-response in categorical confidence across
  alpha 0, +5, +10, and +15.
- For each anomaly outcome, estimate the within-item outcome-by-alpha slope.
- Compare that dose-response slope for definite-false versus definite-correct and
  for ambiguous versus definite-correct.
- Also report condition differences at alpha +15 and paired alpha +15 minus alpha
  0 changes. Error detection and wager/
  abstention are secondary convergent measures.

Negative steering is not part of the primary run. If used, run it under a separate
name as an exploratory control so the frozen primary manifest remains unchanged.

## Validation completed

- Local syntax and pure protocol checks passed.
- Input audit passed: 300 unique IDs and four unique options per item.
- CPU Modal validation passed with the real tokenizer and direction:
  newline `[198]`, ten unique confidence-class tokens, layer-15 direction shape
  `[3584]`, and exactly one hook application.
- Three-item L4 smoke passed: 30/30 unique rows, no continuous parse failures,
  correct alpha-0/alpha-5 hook behavior, and valid global candidate argmaxes.
- Thirty-item confidence-dose pilot passed: all 120 outputs were valid; mean
  expected confidence was 0.605, 0.612, 0.628, and 0.666 at alpha 0, +2, +5,
  and +10. Alpha +10 exceeded alpha +5 on 27/30 items without ceiling collapse.
- Three-item alpha +10 all-measures smoke passed: 30/30 rows, valid response
  tokens, exact hook behavior, and no numeric parse failures.
- Extended alpha 0-40 pilot located the clean upper boundary. Condition means rose
  through alpha +15; alpha +20 introduced aggregate and item-level reversals, and
  alpha +40 decreased confidence on 26/30 items relative to alpha +30. Alpha +15
  exceeded baseline on 27/30 items and is the strongest defensible primary dose.
- Three-item alpha +15 all-measures smoke passed: 30/30 rows, valid response
  tokens, exact hook behavior, and no numeric parse failures.

## Full launch

```bash
.venv/bin/modal run modal_app.py::run_introspection_experiment \
  --run-name panl-introspection-100x3-v1 \
  --alphas 0,5,10,15 \
  --measures all \
  --batch-size 8
```

Reissuing the identical command resumes safely. The Modal GPU container exits when
the function completes, fails, or reaches its timeout.
