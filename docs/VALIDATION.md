# Release Validation

Validated locally on 2026-08-16 before the first Git commit:

- all Python files parsed and compiled successfully;
- 18/18 lightweight tests passed;
- the single-use intervention test applied the layer hook exactly once;
- the frozen stimulus audit found 300 unique items, four unique options per item,
  and the expected 100/100/100 condition balance;
- the raw release artifact regenerated `full_analysis.json` byte-for-byte;
- the regenerated design and dose-response PNGs matched the frozen figures
  byte-for-byte;
- the final analysis passed all 6,000-row integrity checks;
- no credential-shaped strings or user-specific absolute paths were found in files
  selected for Git;
- no selected Git file exceeds 5 MB; the staged repository payload is approximately
  4.1 MB.

Frozen figure hashes:

```text
776faedcc50a1d5b4177224cadcc6af26cd085ef66989d8a547739aee770bf6e  figure-1-design.png
a20828fa83778d3132a04d5d97e66d3bc1d7a2aa8f2f40b401fcfd4c4c253925  figure-2-dose-response.png
```

Release archive:

```text
e7b26555193ec9d4d9377411d7d1d7c85cdf83e6b701bb0f9489ff9cb84de310  causal-confidence-steering-v1.0-artifacts.tar.gz
```

## Post-hoc control extension

Validated and completed on 2026-08-17:

- six control-aware protocol tests passed locally;
- Modal CPU validation passed with the real tokenizer, direction, and frozen inputs;
- one-item L4 smokes produced exactly 9/9 negative-sweep rows and 4/4 polarity rows;
- the full runs produced exactly 900 and 400 unique rows;
- alpha zero had no hook and every nonzero alpha had exactly one hook application;
- all restricted candidate argmaxes were valid;
- the combined analysis matched the same 100 correct and 100 false stimulus IDs used
  in the primary run; and
- the aggregate public analysis contains no question or answer text.

Raw local control checksums:

```text
07c08886773822946236b428f8d2a999ea21c8bf74269856cde0d7788d5789db  negative-correct measurements
421d222169d1b056e49253a6e475cab5285b874a880ac640199758540b244750  polarity-false measurements
```
