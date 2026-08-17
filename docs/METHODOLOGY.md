# Methodology

## Research question

Does Qwen detect a conflict when its confidence is causally increased despite a
false or underdetermined fixed answer, or does the injected confidence itself reduce
the model's metacognitive alarm?

## Calibration and direction construction

Qwen2.5-7B-Instruct generated deterministic answers to deduplicated TriviaQA
questions. Each exact answer was replayed under a ten-class categorical confidence
prompt. Residual-stream activations were captured at the tokenizer-identified
post-answer newline (PANL) across all 28 layers.

Semantic correctness was determined by normalized alias matching followed by a
frozen OpenRouter judge and adjudication of low-confidence disagreements. The final
layer-15 direction was:

```text
mean(PANL residual | correct and Likely)
  - mean(PANL residual | correct and Unlikely)
```

Twenty-five trials were used per class. The direction was scaled to 3% of the mean
residual norm.

## Causal validation

Held-out trials were used for an all-layer `alpha = -5, 0, 5` screen, an independent
confirmation set, and PANL+1, last-answer-token, and confidence-colon controls.
Layers 15-16 showed the strongest reproducible PANL effect. A separate dose pilot
showed monotonic improvement through `alpha = 15` and reversals beginning at larger
doses, motivating layer 15 and `alpha = 15` as the strongest defensible primary dose.

## Experimental stimuli

The final set contained:

- 100 definite questions replayed with Qwen's genuine verified correct answer;
- 100 definite questions replayed with a plausible validated false answer; and
- 100 ambiguous questions replayed with a deterministic forced specific answer.

Correct and false definite conditions were assigned in matched pairs within baseline
confidence class. Ambiguous questions were admitted only after model-recognition and
quality filtering.

## Intervention and measurement

For each item and dose, the exact question-answer prefix was replayed. The layer-15
PANL hidden state was changed once:

```text
h' = h + alpha * v_confidence
```

Five dependent measures were elicited in independent branches with no conversational
state shared across measures: forced inconsistency, continuous unusualness,
categorical confidence, explicit error detection, and wager/abstention. Binary label
order was deterministically counterbalanced by item and fixed across doses.

## Analysis

The final design was 3 conditions x 4 doses x 5 measures x 100 items = 6,000 rows.
The analysis reports cell means, paired dose changes, itemwise dose slopes, slope
contrasts against the definite-correct control, hard categorical confidence, and
response-order audits. Confidence intervals are normal-approximation 95% intervals
over item-level paired quantities and are not adjusted for multiplicity.
