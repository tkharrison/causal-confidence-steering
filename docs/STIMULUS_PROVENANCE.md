# Stimulus Provenance

## Definite conditions

1. Begin with the final 4,000 deterministic TriviaQA collection.
2. Retain semantically correct, high-confidence, concise Qwen answers not used for
   direction construction or held-out steering validation.
3. Generate three plausible false options for each question through the frozen
   OpenRouter generator.
4. Validate that each distractor is false, plausible, type-matched, distinct, and
   non-duplicative.
5. Ask Qwen to choose among the four options and retain questions where it reliably
   selects its genuine correct answer.
6. Match items within baseline confidence class and randomly assign one member of
   each pair to genuine-correct and the other to plausible-false replay.

Relevant scripts are `prepare_definite_candidates.py`,
`generate_trivia_distractors.py`, `validate_trivia_distractors.py`, the
`qualify_definite_options` Modal entrypoint, and `build_experiment_stimuli.py`.

## Ambiguous condition

The final pool combines three source families:

- AmbigQA questions with at least four concise disambiguated interpretations;
- synthetic questions generated to instantiate the same missing-information
  structure; and
- author-supplied questions that were corrected, deduplicated, and independently
  screened.

Candidate options were type-checked and screened for four plausible interpretations,
no obvious default reading, and genuine dependence on missing information. Qwen then
completed counterbalanced recognition checks. Only questions passing the final
recognition criterion were eligible. Manual curation removed factual defects,
semantic duplicates, tautological interpretations, and questions whose four answers
did not genuinely resolve the ambiguity.

The first curated core contributed 34 items. A second author-supplied pool yielded 73
passing items; 66 were selected using seed `20260816` to reach exactly 100.

Relevant scripts are `prepare_ambigqa.py`, `prefilter_ambigqa.py`,
`embellish_ambigqa.py`, `generate_synthetic_ambiguity.py`,
`curate_ambigqa_survivors.py`, `build_final_ambiguous_pool.py`,
`import_user_ambiguous_questions.py`, `assemble_ambiguous_pool.py`, and
`build_experiment_stimuli.py`.

## Frozen outputs

The exact experimental inputs are two frozen JSONL files identified by the SHA-256
checksums in `data/README.md`. They are retained locally but not redistributed in this
public repository while source-dataset permissions are clarified. Authorized users
should verify their local copies against those checksums; otherwise, reconstructing
the curation process constitutes an independent stimulus replication.
