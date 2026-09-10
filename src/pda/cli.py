"""Command line interface.

Pipeline order:

    pda fetch      download filings and XBRL facts     (network, ~10 min)
    pda parse      sections -> chunks -> novelty flags  (local, fast)
    pda goldset    sample and emit the labelling sheet  (local)
                   ... you label the sheet by hand ...
    pda classify   run the LLM over all chunks          (network, costs money)
    pda benchmark  score predictions against your gold  (local)
    pda analyze    build the theme panel and timelines  (local)
    pda memo       render company and cross-cut memos   (local)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import baseline, metrics
from .chunks import Chunk, chunk_section, mark_novelty
from .classify import BULK_MODEL, Classifier, disagreements
from .config import DATA_DIR, load_peers, load_taxonomy
from .edgar import (
    NET_INCOME_TAGS,
    OPERATING_INCOME_TAGS,
    REVENUE_TAGS,
    EdgarClient,
    first_available_tag,
)
from .goldset import labelled_only, read_worksheet, stratified_sample, write_worksheet
from .memo import CompanyMemo, render_crosscut
from .sections import extract_sections
from .timeline import build_panel, first_mentions, join_financials

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

RAW = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
OUTPUT = DATA_DIR / "output"


def _client() -> EdgarClient:
    agent = os.environ.get("SEC_USER_AGENT", "")
    return EdgarClient(user_agent=agent, cache_dir=RAW / "http")


def _load_chunks() -> list[Chunk]:
    path = INTERIM / "chunks.jsonl"
    if not path.exists():
        raise typer.BadParameter("No chunks found. Run `pda parse` first.")
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines() if line]


@app.command()
def fetch(
    ticker: str = typer.Option("", help="Fetch one ticker instead of the peer set")
) -> None:
    """Download 10-K documents and XBRL company facts from EDGAR."""
    peers = load_peers()
    client = _client()
    RAW.mkdir(parents=True, exist_ok=True)

    targets = [p for p in peers.peers if not ticker or p.ticker == ticker.upper()]
    manifest: list[dict] = []

    for peer in targets:
        try:
            cik = client.resolve_cik(peer.ticker)
        except Exception as error:  # noqa: BLE001
            console.print(f"[red]{peer.ticker}: {error}[/red]")
            continue

        filings = client.list_filings(
            cik,
            peer.ticker,
            forms=peers.study.forms,
            start_year=peers.study.start_fiscal_year,
            end_year=peers.study.end_fiscal_year,
        )
        console.print(f"{peer.ticker}: CIK {cik}, {len(filings)} filings")

        for filing in filings:
            destination = RAW / "filings" / f"{peer.ticker}-{filing.fiscal_year}.html"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_text(client.fetch_document(filing), encoding="utf-8")
            manifest.append(
                {
                    "ticker": peer.ticker,
                    "cik": cik,
                    "fiscal_year": filing.fiscal_year,
                    "accession": filing.accession,
                    "filing_date": filing.filing_date,
                    "path": str(destination.relative_to(DATA_DIR)),
                }
            )

        facts_path = RAW / "facts" / f"{peer.ticker}.json"
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        if not facts_path.exists():
            facts_path.write_text(json.dumps(client.company_facts(cik)))

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.print(f"[green]Wrote manifest with {len(manifest)} filings[/green]")


@app.command()
def parse() -> None:
    """Split filings into sections, chunk them, and flag new language."""
    taxonomy = load_taxonomy()
    manifest = json.loads((RAW / "manifest.json").read_text())
    INTERIM.mkdir(parents=True, exist_ok=True)

    by_ticker: dict[str, list[dict]] = {}
    for record in manifest:
        by_ticker.setdefault(record["ticker"], []).append(record)

    all_chunks: list[Chunk] = []
    problems: list[str] = []

    for ticker, records in sorted(by_ticker.items()):
        prior: list[Chunk] = []
        for record in sorted(records, key=lambda r: r["fiscal_year"]):
            html = (DATA_DIR / record["path"]).read_text(encoding="utf-8", errors="replace")
            sections = extract_sections(html, taxonomy.sections_in_scope)
            missing = set(taxonomy.sections_in_scope) - set(sections)
            if missing:
                problems.append(
                    f"{ticker} FY{record['fiscal_year']}: missing {sorted(missing)}"
                )

            current: list[Chunk] = []
            for item, section in sections.items():
                current += chunk_section(
                    section.text,
                    ticker=ticker,
                    fiscal_year=record["fiscal_year"],
                    item=item,
                    filing_accession=record["accession"],
                    filing_date=record["filing_date"],
                )
            current = mark_novelty(current, prior)
            all_chunks += current
            prior = current

    with (INTERIM / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")

    console.print(f"[green]{len(all_chunks)} chunks from {len(manifest)} filings[/green]")
    if problems:
        console.print("[yellow]Sections that could not be located:[/yellow]")
        for problem in problems:
            console.print(f"  {problem}")
        console.print(
            "[yellow]Check these by hand before trusting any counts for "
            "those companies.[/yellow]"
        )


@app.command()
def goldset(
    n: int = typer.Option(400, help="Target gold set size"),
    enriched_share: float = typer.Option(0.5, help="Share drawn from keyword hits"),
    show_baseline: bool = typer.Option(False, help="Un-blind the baseline guesses"),
) -> None:
    """Emit the stratified CSV worksheet you label by hand."""
    peers, taxonomy = load_peers(), load_taxonomy()
    rows = stratified_sample(
        _load_chunks(), peers, taxonomy, target_n=n, enriched_share=enriched_share
    )
    path = OUTPUT / "goldset_worksheet.csv"
    write_worksheet(rows, path, blind=not show_baseline)
    console.print(f"[green]Wrote {len(rows)} rows to {path}[/green]")
    console.print(
        "Label the `gold_labels` column with pipe-separated theme ids. Write "
        "NONE in `notes` when a paragraph deliberately has no theme, so blank "
        "rows can be told apart from unlabelled ones."
    )


@app.command()
def classify(
    limit: int = typer.Option(0, help="Only classify the first N chunks"),
    model: str = typer.Option(BULK_MODEL, help="Model id for the bulk pass"),
) -> None:
    """Run the LLM classifier over all chunks, with caching."""
    taxonomy = load_taxonomy()
    chunks = _load_chunks()
    if limit:
        chunks = chunks[:limit]

    classifier = Classifier(taxonomy=taxonomy, cache_dir=INTERIM / "llm_cache")
    predictions = classifier.classify(
        [c.chunk_id for c in chunks], [c.text for c in chunks], model=model
    )

    conflicts = disagreements([c.text for c in chunks], predictions, taxonomy)
    console.print(
        f"{len(conflicts)} of {len(chunks)} chunks disagree with the keyword "
        "baseline. Re-run these through the adjudicator model with "
        "`--model $PDA_ADJUDICATOR_MODEL` for the benchmark slice."
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (INTERIM / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict()) + "\n")
    console.print(f"[green]Wrote {len(predictions)} predictions[/green]")


@app.command()
def benchmark(
    worksheet: Path = typer.Option(
        OUTPUT / "goldset_worksheet.csv", help="Your labelled worksheet"
    )
) -> None:
    """Score the LLM and the keyword baseline against your gold labels."""
    taxonomy = load_taxonomy()
    rows = labelled_only(read_worksheet(worksheet, taxonomy))
    if not rows:
        raise typer.BadParameter("No labelled rows found in the worksheet.")

    predictions = {}
    prediction_file = INTERIM / "predictions.jsonl"
    if prediction_file.exists():
        for line in prediction_file.read_text().splitlines():
            if line:
                record = json.loads(line)
                predictions[record["chunk_id"]] = record["labels"]

    gold = [row.gold_labels for row in rows]
    llm = [predictions.get(row.chunk_id, []) for row in rows]
    keyword = [baseline.classify_chunk(row.text, taxonomy) for row in rows]

    llm_result = metrics.evaluate(gold, llm, taxonomy.ids)
    keyword_result = metrics.evaluate(gold, keyword, taxonomy.ids)

    table = Table(title=f"Benchmark on {len(rows)} labelled chunks")
    for column in ("Theme", "Support", "LLM F1", "Keyword F1", "LLM 95% CI"):
        table.add_column(column)
    for llm_score, key_score in zip(
        llm_result.per_theme, keyword_result.per_theme, strict=True
    ):
        table.add_row(
            llm_score.theme,
            str(llm_score.support),
            f"{llm_score.f1:.3f}",
            f"{key_score.f1:.3f}",
            f"[{llm_score.f1_low:.2f}, {llm_score.f1_high:.2f}]",
        )
    console.print(table)
    console.print(
        f"micro F1 {llm_result.micro_f1:.3f} "
        f"[{llm_result.micro_f1_low:.2f}, {llm_result.micro_f1_high:.2f}] "
        f"vs keyword {keyword_result.micro_f1:.3f}"
    )
    console.print(
        f"abstention accuracy {llm_result.abstain_accuracy:.3f} "
        "(share of genuinely themeless paragraphs left unlabelled)"
    )

    confusions = metrics.confusion_pairs(gold, llm, taxonomy.ids)
    if confusions:
        console.print("\nMost common confusions (gold -> predicted):")
        for missed, spurious, count in confusions:
            console.print(f"  {missed} -> {spurious}: {count}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "benchmark.json").write_text(
        json.dumps(
            {
                "n": llm_result.n,
                "taxonomy_version": taxonomy.version,
                "llm": {
                    "micro_f1": llm_result.micro_f1,
                    "macro_f1": llm_result.macro_f1,
                    "ci": [llm_result.micro_f1_low, llm_result.micro_f1_high],
                    "per_theme": llm_result.as_rows(),
                },
                "keyword_baseline": {
                    "micro_f1": keyword_result.micro_f1,
                    "macro_f1": keyword_result.macro_f1,
                },
                "confusions": confusions,
            },
            indent=2,
        )
    )


@app.command()
def analyze(
    theme: str = typer.Option("isv_channel_displacement", help="Headline theme")
) -> None:
    """Build the theme panel, first-mention table and financial join."""
    peers, taxonomy = load_peers(), load_taxonomy()
    chunks = _load_chunks()

    labels: dict[str, list[str]] = {}
    for line in (INTERIM / "predictions.jsonl").read_text().splitlines():
        if line:
            record = json.loads(line)
            labels[record["chunk_id"]] = record["labels"]

    panel = build_panel(chunks, labels, peers, taxonomy)
    first = first_mentions(panel, peers)

    revenue: dict[str, dict[int, float]] = {}
    for peer in peers.peers:
        path = RAW / "facts" / f"{peer.ticker}.json"
        if not path.exists():
            continue
        _, points = first_available_tag(json.loads(path.read_text()), REVENUE_TAGS)
        revenue[peer.ticker] = {p.fiscal_year: p.value for p in points}

    joined = join_financials(
        first,
        revenue,
        theme,
        start_year=peers.study.start_fiscal_year,
        end_year=peers.study.end_fiscal_year,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "panel.json").write_text(
        json.dumps([r.__dict__ for r in panel], indent=2)
    )
    (OUTPUT / "first_mentions.json").write_text(
        json.dumps([r.__dict__ for r in first], indent=2)
    )
    (OUTPUT / "theme_vs_financials.json").write_text(json.dumps(joined, indent=2))

    table = Table(title=f"First mention: {taxonomy.get(theme).name}")
    for column in ("Ticker", "Cohort", "Year", "Lag", "Revenue CAGR"):
        table.add_column(column)
    for record in joined:
        table.add_row(
            record["ticker"],
            record["cohort"],
            str(record["first_mention_year"]),
            str(record["lag_vs_peers"]),
            f"{record['revenue_cagr']:.1%}" if record["revenue_cagr"] else "-",
        )
    console.print(table)


@app.command()
def memo(
    theme: str = typer.Option("isv_channel_displacement", help="Headline theme")
) -> None:
    """Render one memo per company plus the cross-cut findings skeleton."""
    peers, taxonomy = load_peers(), load_taxonomy()
    chunks = _load_chunks()

    labels: dict[str, list[str]] = {}
    for line in (INTERIM / "predictions.jsonl").read_text().splitlines():
        if line:
            record = json.loads(line)
            labels[record["chunk_id"]] = record["labels"]

    panel = build_panel(chunks, labels, peers, taxonomy)
    first = first_mentions(panel, peers)
    memo_dir = OUTPUT / "memos"
    memo_dir.mkdir(parents=True, exist_ok=True)

    for peer in peers.peers:
        rows = [r for r in panel if r.ticker == peer.ticker]
        if not rows:
            continue
        years = sorted({r.fiscal_year for r in rows})

        facts_path = RAW / "facts" / f"{peer.ticker}.json"
        financials: dict[str, dict[int, float]] = {}
        if facts_path.exists():
            facts = json.loads(facts_path.read_text())
            for label, tags in (
                ("Revenue", REVENUE_TAGS),
                ("Operating income", OPERATING_INCOME_TAGS),
                ("Net income", NET_INCOME_TAGS),
            ):
                _, points = first_available_tag(facts, tags)
                financials[label] = {p.fiscal_year: p.value for p in points}

        quotes = []
        for chunk in chunks:
            if chunk.ticker != peer.ticker or not chunk.is_new_language:
                continue
            for label in labels.get(chunk.chunk_id, []):
                quotes.append((label, chunk.fiscal_year, chunk.text))
        quotes = sorted(quotes, key=lambda q: -q[1])[:5]

        document = CompanyMemo(
            ticker=peer.ticker,
            name=peer.name,
            cohort=peer.cohort,
            fiscal_years=years,
            financials=financials,
            theme_rows=rows,
            quotes=quotes,
        ).render(taxonomy)
        (memo_dir / f"{peer.ticker}.md").write_text(document, encoding="utf-8")

    crosscut = render_crosscut(panel, first, peers, taxonomy, theme)
    (OUTPUT / "crosscut.md").write_text(crosscut, encoding="utf-8")
    console.print(f"[green]Memos written to {memo_dir}[/green]")


if __name__ == "__main__":
    app()
