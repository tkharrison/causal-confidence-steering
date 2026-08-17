# Frozen inputs

## Stimuli

The local file `stimuli/definite-experiment-stimuli-100-v1.jsonl` contains 200 definite items:
100 genuine semantically verified Qwen answers and 100 plausible independently
validated false substitutions. Assignment was paired within confidence class using
seed `20260816`.

The local file `stimuli/ambiguous-experiment-stimuli-100-v1.jsonl` contains 100 questions for which
Qwen repeatedly selected the option indicating that the question lacked enough
information. Before steering, Qwen was forced to choose one substantive answer; that
choice was then frozen across every dose and dependent measure.

SHA-256:

```text
a326a26501dbaa913f61d717eb8309d893855295857e4d63c633ae461450902f  definite-experiment-stimuli-100-v1.jsonl
3b1d39f656b91db324948db8bbb8b6390282a949fc88a91809617354bef2fee1  ambiguous-experiment-stimuli-100-v1.jsonl
```

## Confidence direction

`direction/qwen25-7b-panl-direction-v1.safetensors` stores the final direction under
key `panl.layer_15`. It was constructed from 25 correct `Likely` and 25 correct
`Unlikely` TriviaQA trials, then scaled to 3% of the mean PANL residual norm.

```text
1c5f38df83591bbd919f69be4971bd6166d468394146bcf81377596ea144be7c  qwen25-7b-panl-direction-v1.safetensors
```

The accompanying JSON records selected trial IDs and construction parameters.

## Licensing note

The exact stimulus JSONL files are intentionally excluded from this public repository
because some text is derived from TriviaQA and AmbigQA and redistribution permission
has not been confirmed. The checksums above identify the frozen inputs without
relicensing or redistributing third-party-derived text. The MIT license covers the
software only.
