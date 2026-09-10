# Benchmark

_Numbers below are filled in by `pda benchmark`, which writes
`data/output/benchmark.json`. Publish whatever it returns._

## Headline

| | Micro F1 | Macro F1 | Abstention accuracy |
|---|---|---|---|
| LLM (bulk model) | — | — | — |
| LLM (with adjudication) | — | — | — |
| Keyword baseline | — | — | — |

Gold set: — labelled paragraphs, — enriched / — random.
Self-consistency (Cohen's kappa, 100 re-labelled rows): —

Report the random-stratum precision separately from the enriched-stratum
figure. The enriched number is the more flattering one and it is not the
number that describes performance on a real filing.

## Per theme

| Theme | Support | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|---|

Themes with support under 15 should be reported but not interpreted. Say so
rather than quietly dropping them.

## Failure taxonomy

Work through every error by hand and sort it into one of these. The
distribution is the interesting result, not the headline F1.

**1. Taxonomy boundary errors.** Two themes whose definitions genuinely
overlap. Shows up as a symmetric confusion pair. Fixable by tightening the
exclusion criteria and re-labelling — and the fix belongs in the writeup as a
before-and-after.

**2. Keyword anchoring.** The model labels a paragraph because a term appears,
not because a claim is made. Most common on `interchange_regulation`, where
the word "interchange" appears in almost every filing.

**3. Missing lexical anchor.** The paragraph makes the claim without using the
vocabulary. These are the ones the keyword baseline cannot catch at all, and
recall on them is the honest argument for using a model.

**4. Scope confusion.** The paragraph describes a competitor's position, a
historical arrangement, or a hypothetical, and the model labels it as the
filer's own situation.

**5. Fragment errors.** Chunking split a paragraph mid-argument and neither
half stands alone. A parsing failure, not a model failure — count it
separately or the model gets blamed for the pipeline.

## What changed after the fixes

| Iteration | Change | Micro F1 | Δ |
|---|---|---|---|
| v1.0.0 | baseline taxonomy | — | — |

Show at least one fix that moved the number and one that did not. A table
where every change helps is a table someone stopped reporting honestly.
