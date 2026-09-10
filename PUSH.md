# Getting this onto GitHub

From inside this folder:

```bash
git init
git add .
git commit -m "Payments disclosure atlas: pipeline, taxonomy and benchmark harness"

# Create the repo (GitHub CLI). If you don't have gh, make an empty repo on
# github.com first and use the remote-add line below instead.
gh repo create payments-disclosure-atlas --public --source=. --push

# Without gh:
# git remote add origin https://github.com/<your-username>/payments-disclosure-atlas.git
# git branch -M main
# git push -u origin main
```

## First run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # fill in SEC_USER_AGENT and ANTHROPIC_API_KEY
pytest -q                 # 78 tests, no network needed

pda fetch                 # ~10 minutes against EDGAR
pda parse
```

`pda fetch` is the first step that touches the network. Watch the output: if a
ticker fails to resolve, the company has probably renamed or delisted and the
peer set needs updating. If `pda parse` reports sections it could not locate,
open those filings by hand before trusting any counts for that company.

## What to do before you show anyone

1. Run `fetch` and `parse`, and work through the section-parsing warnings.
2. Run `goldset` and label the 400 rows. This is the tedious afternoon, and it
   is also the part nobody else does.
3. Run `classify`, then `benchmark`, and put the real numbers in
   `docs/BENCHMARK.md`.
4. Work every error into the failure taxonomy, fix one or two, and record the
   before-and-after.
5. Only then write the deck.
