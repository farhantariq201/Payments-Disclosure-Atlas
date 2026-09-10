# Methodology

## 1. Universe

Fourteen US domestic filers in payments and merchant acquiring, in three
cohorts (see `config/peers.yaml`). Cohort assignment is a judgement call and
is stated up front rather than derived, because pretending it fell out of the
data would be worse than owning it. Foreign private issuers are excluded on
filing-format grounds, not because they are unimportant — Adyen and dLocal are
central to the industry and their absence is a real limitation of the study.

## 2. Documents

10-K filings for fiscal years 2020 through 2025, fetched from EDGAR by CIK.
Item 1, Item 1A and Item 7 are in scope: the strategic framing lives in the
risk factors and MD&A, and Item 1 is included because competitive language
often appears in the business description first.

Sections the parser cannot locate are reported, not silently skipped. Any
company with a missing Item 1A in a given year has to be checked by hand
before its counts are used, because a missing section looks exactly like a
company that stopped talking about everything at once.

## 3. Chunking and novelty

Paragraphs shorter than 200 characters are dropped (headers, page furniture,
table fragments). Paragraphs longer than 4,000 characters are split on
sentence boundaries.

Each paragraph is compared to the prior year's paragraphs in the same item.
Comparison is 5-gram Jaccard over normalised text with numbers masked, so a
paragraph carried forward with refreshed figures scores as unchanged.
Similarity below 0.6 marks the paragraph as new language. The threshold is
arbitrary and should be sensitivity-tested; 0.5 and 0.7 both produce defensible
results and the finding should not depend on the choice.

## 4. Taxonomy

Twelve themes, each with a definition, inclusion and exclusion criteria, and a
worked positive and negative example. The exclusions do the real work: the
boundary between "we are threatened by software platforms" and "we sell to
software platforms" is the entire study, and without an explicit exclusion the
classifier collapses them.

The taxonomy version is part of the classifier cache key. Editing a definition
invalidates every label produced under the old one, which is the correct
behaviour and the reason the version exists.

## 5. Classification

Bulk pass with a cheap model, batched six paragraphs per call. Chunks where
the model and the keyword baseline disagree are re-run through a stronger
model, on the reasoning that agreement is cheap evidence and disagreement is
where the errors live.

The prompt explicitly instructs abstention: most paragraphs carry no theme.
Without that instruction the model finds a theme in everything and precision
collapses — this was visible in early runs and is the single highest-leverage
line in the prompt.

## 6. Gold set

400 paragraphs, stratified by cohort x fiscal year x item, seeded for
reproducibility.

Half the sample is drawn from paragraphs the keyword baseline flags. Pure
random sampling yields roughly 90% themeless paragraphs, which leaves the rarer
themes with almost no positives and confidence intervals wide enough to be
useless. The stratum is recorded on every row and the benchmark reports
enriched and random slices separately. **Quoting only the enriched number
would materially overstate real-world precision.**

The worksheet is blind by default: the baseline's guesses are hidden. Seeing
them first means labelling by agreeing with regex rather than by reading, and
the benchmark then measures how well the model imitates the baseline.

A deliberate "no theme applies" is recorded as `NONE` in the notes column, so
unlabelled rows can be told apart from labelled-empty ones. Counting blanks as
negatives would inflate precision.

## 7. Metrics

Per-theme precision, recall and F1, plus micro and macro F1. Bootstrap 95%
confidence intervals over 1,000 resamples, because a theme with twelve
positives does not support a bare point estimate.

Abstention accuracy — the share of genuinely themeless paragraphs correctly
left unlabelled — is reported separately. It is the metric that catches an
over-eager classifier, and it is the one most easily hidden by a good-looking
F1.

Cohen's kappa on a re-labelled 100-row slice measures self-consistency. If you
agree with yourself only 70% of the time, no model can score above that
ceiling and the gold set needs tightening before the model does.

## 8. Analysis

- **first_mention_year** — earliest fiscal year a theme appears, censored for
  companies whose first filing is after the study start.
- **salience** — share of in-scope paragraphs carrying the theme that year,
  which normalises for the fact that some filers write four times as much.
- **escalation** — year-over-year change in salience restricted to new
  language.
- **adoption curve** — share of *covered* companies mentioning the theme by
  year and cohort. The denominator excludes companies not yet public.

## 9. Threats to validity

1. Fourteen companies is not a sample that supports causal claims.
2. Disclosure lags strategy by an unknown and probably uneven amount.
3. Risk factor drafting reflects counsel's appetite as much as management's
   view, and that varies by company in ways this study cannot observe.
4. Cohort assignment is judgement, and the headline result would change if
   Block or PayPal were classified as incumbents.
5. One labeller. Kappa bounds the damage but does not remove it.
