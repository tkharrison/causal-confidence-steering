# Public Repository Manifest

This manifest records why each category is present in the curated repository.

## Included

### Core experiment

- `src/model.py`, `dataset.py`, `config.py`, and `activations.py`: deterministic
  Qwen answer generation, exact replay, confidence logits, PANL identification, and
  residual capture.
- `src/confidence_direction.py` and `interventions.py`: construction and validation
  of the `Likely - Unlikely` direction.
- `src/introspection_protocol.py` and `introspection_experiment.py`: frozen prompts,
  counterbalancing, single-use PANL intervention, checkpointing, and scoring.
- `modal_app.py`: final compute-platform adapter used for the completed run.

### Stimulus construction

- AmbigQA conversion, model qualification, OpenRouter prefiltering, synthetic
  generation, manual curation, user-question import, and final pool assembly are
  retained because each contributed to or selected stimuli in the final 100-item
  ambiguous condition.
- Semantic grading, distractor generation/validation, definite qualification, and
  matched assignment are retained because they generated the final correct/false
  definite conditions and the confidence direction's correctness labels.

### Analysis

- `analyze_full_introspection_results.py` is the authoritative final analysis.
- `analyze_control_extension.py` is the authoritative post-hoc signed-sweep,
  polarity-pair, and exact log-odds analysis.
- `analyze_introspection_experiment.py` is retained as the original tidy-export and
  early paired-analysis utility used immediately after collection.
- `plot_results.py` contains the figure code extracted from the manuscript builder.
  It is independent of document generation.

### Frozen artifacts

- The direction tensor and metadata, run manifest, compact numeric analysis,
  aggregate control analysis, validation reports, final figures, and exact stimulus
  checksums are included.
- Exact dataset-derived stimulus text and raw measurements are withheld pending
  confirmation of redistribution permission.

## Deliberately excluded

- `scripts/build_paper.py` and all rendered Word/PDF artifacts
- the historical Lightning implementation and 400 MB activation archive
- provider-migration and Lightning/Modal equivalence code
- virtual environments, caches, downloaded model weights, and temporary renders
- incomplete smoke runs, superseded dose pilots, and obsolete 3,000-item outputs
- discarded or failed candidate pools and judge smoke tests
- local API responses not required to reproduce the frozen inputs or reported results

## Publication blockers

1. Confirm redistribution terms before publishing derived TriviaQA or AmbigQA text.
2. Add an archival DOI to `CITATION.cff` if one is created.
3. Upload the prepared artifact archive only after redistribution permission is clear.
