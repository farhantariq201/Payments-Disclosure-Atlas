# Payments Disclosure Atlas

A study of how one strategic idea moved through an industry's public
disclosures, and a benchmark measuring how reliably the classification behind
it works.

**The question.** Merchant acquiring used to be sold by bank referrals and
direct sales forces. Over the last five years the distribution shifted to
software: a restaurant gets payment processing from its point-of-sale vendor
because it comes bundled, not because a processor's rep called. Incumbents did
not all notice at the same time. This repo pulls five years of 10-K filings
from fourteen payments companies, tags every paragraph against a hand-built
theme taxonomy, and dates the moment each company began describing integrated
software vendors as a competitive threat rather than a distribution partner.

**The second question, which is the point.** Any pipeline that has a language
model reading filings will get things wrong. So this repo also ships a
stratified gold set labelled by hand, a keyword baseline to measure against,
and a benchmark that reports per-theme F1 with bootstrap confidence intervals.
The accuracy number is published whatever it turns out to be.

---

## What's here

```
config/taxonomy.yaml     12 themes: definitions, boundaries, worked examples
config/peers.yaml        14 US filers across 3 cohorts
src/pda/edgar.py         SEC client: CIK resolution, filings, XBRL facts
src/pda/sections.py      10-K item splitter that survives tables of contents
src/pda/chunks.py        paragraph chunking + year-over-year novelty detection
src/pda/baseline.py      keyword classifier (the thing the LLM must beat)
src/pda/classify.py      LLM labelling: batched, cached, two-tier routing
src/pda/goldset.py       stratified sampling + the labelling worksheet
src/pda/metrics.py       multi-label P/R/F1, bootstrap CIs, Cohen's kappa
src/pda/timeline.py      adoption curves, first-mention lag, escalation
src/pda/memo.py          company memos + the cross-cut findings skeleton
```

72 tests, no network required to run them.

## The peer set

| Cohort | Companies |
|---|---|
| Legacy processors & acquirers | FI, FIS, GPN, CPAY, WEX, EEFT |
| Software-led & integrated | TOST, FOUR, XYZ, PYPL, MQ |
| Cross-border & vertical specialists | RELY, FLYW, PAYO |

Foreign private issuers (dLocal, Lightspeed, Adyen) are excluded: they file
20-F/40-F with a different item structure, and mixing them in would mean the
section parser is doing two different jobs badly instead of one job well.

## Running it

```bash
pip install -e ".[dev]"
cp .env.example .env        # add your SEC User-Agent and API key
pytest -q

pda fetch                   # ~70 filings + XBRL facts from EDGAR
pda parse                   # sections -> chunks -> novelty flags
pda goldset --n 400         # emits the CSV you label by hand
#   ... label it ...
pda classify                # LLM pass over all chunks, cached
pda benchmark               # score LLM and keyword baseline against your gold
pda analyze                 # adoption curves, first-mention lag
pda memo                    # company memos + cross-cut findings
```

## Five decisions worth explaining

**Table-of-contents defence.** Every 10-K names "Item 1A. Risk Factors" at
least twice: once in the contents, once at the section, often again in a
cross-reference. Taking the first match gives you a forty-word stub. The
parser keeps the occurrence that opens the longest run of text, and reports
sections it could not locate instead of returning empty ones.

**Novelty detection.** Risk factors are heavily copy-pasted year to year. A
company that wrote a paragraph in 2020 and never touched it looks identical to
one actively escalating, unless you check. Paragraphs are compared to the
prior year by 5-gram Jaccard with numbers masked, so a figure refresh does not
register as new language. Only genuinely new or rewritten text counts as
escalation.

**Point-in-time financials.** XBRL company facts contain each fiscal year many
times over: as originally filed, as a comparative, and as restated. This repo
keeps the earliest-filed value, because that is the number the market had.
Mixing in restatements is lookahead bias.

**Censoring for IPO dates.** Six of the fourteen companies listed mid-window.
Silence in 2020 from a company that IPO'd in 2021 is absence of a filing, not
absence of a view. Those companies are flagged as censored and excluded from
the peer median, otherwise every incumbent gets a manufactured head start.

**A baseline to lose to.** "Our classifier hits 0.78 F1" means nothing alone.
The keyword baseline runs on the same held-out set so the LLM number has
something to be compared against, and on some themes the baseline wins, which
is worth knowing before spending money on tokens.

## Where this is weak

Fourteen companies cannot support a causal claim about outcomes. The
first-mention-versus-growth table is suggestive at best, and a real test would
need a wider universe, controls for size and segment mix, and a sharper
definition of the treatment than "wrote a paragraph about it."

Disclosure lags strategy. Risk factor drafting is partly a legal exercise and
partly institutional habit, so a company may have understood the shift years
before naming it. The study measures when companies said things, not when they
knew them, and the writeup should not blur that.

The gold set is labelled by one person. Cohen's kappa on a re-labelled slice
gives a ceiling for how much the accuracy figures can be trusted, but a second
labeller would be better.

## Findings

See `docs/BENCHMARK.md` for classifier accuracy and the failure taxonomy, and
`data/output/crosscut.md` for the adoption analysis once the pipeline has run.
