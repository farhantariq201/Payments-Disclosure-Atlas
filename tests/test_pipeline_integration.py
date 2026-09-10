"""End-to-end run over the fixture filings: HTML in, memo out, no network."""

from pathlib import Path

from pda.baseline import classify_chunk
from pda.chunks import chunk_section, mark_novelty
from pda.config import load_peers, load_taxonomy
from pda.memo import CompanyMemo, render_crosscut
from pda.sections import extract_sections
from pda.timeline import build_panel, first_mentions

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).resolve().parents[1] / "config"


def _run():
    peers = load_peers(CONFIG / "peers.yaml")
    taxonomy = load_taxonomy(CONFIG / "taxonomy.yaml")

    all_chunks, prior = [], []
    for year, filename in ((2023, "mini_10k_fy2023.html"), (2024, "mini_10k_fy2024.html")):
        html = (FIXTURES / filename).read_text()
        sections = extract_sections(html, taxonomy.sections_in_scope)
        current = []
        for item, section in sections.items():
            current += chunk_section(
                section.text, ticker="FISV", fiscal_year=year, item=item,
                filing_accession=f"acc-{year}", filing_date=f"{year+1}-02-20",
            )
        current = mark_novelty(current, prior)
        all_chunks += current
        prior = current

    # Stand in for the LLM with the keyword baseline so the test stays offline.
    labels = {c.chunk_id: classify_chunk(c.text, taxonomy) for c in all_chunks}
    panel = build_panel(all_chunks, labels, peers, taxonomy)
    return all_chunks, labels, panel, peers, taxonomy


def test_pipeline_produces_chunks_across_both_years_and_all_items():
    chunks, _, _, _, taxonomy = _run()
    assert {c.fiscal_year for c in chunks} == {2023, 2024}
    assert set(taxonomy.sections_in_scope).issubset({c.item for c in chunks})


def test_pipeline_detects_the_theme_planted_in_the_fixture():
    _, labels, _, _, _ = _run()
    found = {theme for row in labels.values() for theme in row}
    assert "isv_channel_displacement" in found
    assert "take_rate_compression" in found


def test_new_theme_in_fy2024_is_flagged_as_new_language():
    chunks, labels, _, _, _ = _run()
    stablecoin = [
        c for c in chunks
        if "stablecoin_crypto_rails" in labels.get(c.chunk_id, [])
    ]
    assert stablecoin
    assert all(c.fiscal_year == 2024 for c in stablecoin)
    assert all(c.is_new_language for c in stablecoin)


def test_carried_forward_theme_is_present_but_not_new_in_year_two():
    chunks, labels, _, _, _ = _run()
    isv_2024 = [
        c for c in chunks
        if c.fiscal_year == 2024
        and "isv_channel_displacement" in labels.get(c.chunk_id, [])
    ]
    assert isv_2024
    assert not any(c.is_new_language for c in isv_2024)


def test_first_mention_reflects_the_earlier_year():
    _, _, panel, peers, _ = _run()
    records = {
        f.theme: f for f in first_mentions(panel, peers) if f.ticker == "FISV"
    }
    assert records["isv_channel_displacement"].year == 2023
    assert records["stablecoin_crypto_rails"].year == 2024


def test_memos_render_end_to_end():
    chunks, labels, panel, peers, taxonomy = _run()
    company = CompanyMemo(
        ticker="FISV", name="Fiserv, Inc.", cohort="incumbent",
        fiscal_years=[2023, 2024],
        financials={"Revenue": {2023: 1.0e9, 2024: 1.1e9}},
        theme_rows=[r for r in panel if r.ticker == "FISV"],
        quotes=[
            (theme, c.fiscal_year, c.text)
            for c in chunks for theme in labels.get(c.chunk_id, [])
        ][:3],
    ).render(taxonomy)
    assert "Theme salience by year" in company
    assert len(company) > 800

    crosscut = render_crosscut(
        panel, first_mentions(panel, peers), peers, taxonomy,
        "isv_channel_displacement",
    )
    assert "Who named it first" in crosscut
