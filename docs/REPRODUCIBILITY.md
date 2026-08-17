# Reproducibility Guide

## Lightweight verification

These checks do not require Qwen weights or a GPU:

```bash
python3 -m compileall -q src scripts modal_app.py tests
python3 -m pytest -q
python3 scripts/validate_experiment_inputs.py \
  --definite data/stimuli/definite-experiment-stimuli-100-v1.jsonl \
  --ambiguous data/stimuli/ambiguous-experiment-stimuli-100-v1.jsonl \
  --output /tmp/input-validation.json
```

## Analysis reproduction

If you have an authorized copy of the frozen raw archive, extract it into
`results/raw`, then run:

```bash
python3 scripts/analyze_full_introspection_results.py --run-dir results/raw
python3 scripts/plot_results.py \
  --analysis results/raw/full_analysis.json \
  --output-dir figures/generated
```

The regenerated `full_analysis.json` should match `results/main/full_analysis.json`
semantically. PNG hashes can vary across operating systems because available fonts
and rasterization differ; plotted values are read directly from the frozen analysis.

## Full GPU reproduction

Obtain or independently reconstruct the two stimulus files, verify them against the
published checksums, and upload them with the direction tensor to the named Modal results volume,
then run `validate_introspection_runner`. Only after it passes should the full
experiment command in the root README be launched.

The model revision, direction checksum, stimulus checksums, prompt hashes, layer,
doses, measures, and seed are all frozen into the run manifest. The runner refuses to
resume into a directory whose signature does not match.

## Control-extension analysis

After obtaining authorized local copies of the two control-run directories, run:

```bash
python3 scripts/analyze_control_extension.py \
  --primary-dir results/raw/panl-introspection-100x3-v1 \
  --negative-dir results/raw/panl-controls-negative-correct-100-v1 \
  --polarity-dir results/raw/panl-controls-polarity-false-100-v1 \
  --output-dir results/control/generated
```

The committed aggregate output is in `results/control/`. Raw control rows contain
the frozen stimulus text and therefore follow the same redistribution restriction as
the primary raw measurements.

## Expected integrity checks

- 6,000 unique rows
- 60 cells, each containing 100 observations
- one run signature matching the manifest
- PANL token ID 198 throughout
- no hook at alpha zero and exactly one hook at every nonzero alpha
- valid candidate-token argmaxes
- no failed continuous-score parses
- binary probabilities summing to one
- fixed A/B semantic mappings within item across doses
