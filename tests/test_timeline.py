from pathlib import Path

from pda.chunks import Chunk
from pda.config import load_peers, load_taxonomy
from pda.timeline import (
    adoption_curve,
    build_panel,
    escalation_table,
    first_mentions,
    join_financials,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
THEME = "isv_channel_displacement"


def _chunk(ticker, year, ordinal, new=False):
    return Chunk(
        chunk_id=f"{ticker}-{year}-item1a-{ordinal:04d}",
        ticker=ticker,
        fiscal_year=year,
        item="item1a",
        ordinal=ordinal,
        text="text",
        text_hash=f"{ticker}{year}{ordinal}",
        filing_accession="a",
        filing_date=f"{year + 1}-02-01",
        is_new_language=new,
    )


def _scenario():
    """FI names the theme in 2020; GPN not until 2023; TOST (2021 IPO) in 2021."""
    chunks, labels = [], {}
    plan = {
        ("FISV", 2020): True, ("FISV", 2021): True, ("FISV", 2022): True, ("FISV", 2023): True,
        ("GPN", 2020): False, ("GPN", 2021): False, ("GPN", 2022): False, ("GPN", 2023): True,
        ("TOST", 2021): True, ("TOST", 2022): True, ("TOST", 2023): True,
    }
    for (ticker, year), mentions in plan.items():
        for ordinal in range(4):
            chunk = _chunk(ticker, year, ordinal, new=(ordinal == 0))
            chunks.append(chunk)
            labels[chunk.chunk_id] = [THEME] if (mentions and ordinal == 0) else []
    return chunks, labels


def _panel():
    peers, taxonomy = load_peers(CONFIG / "peers.yaml"), load_taxonomy(CONFIG / "taxonomy.yaml")
    chunks, labels = _scenario()
    return build_panel(chunks, labels, peers, taxonomy), peers


def test_panel_covers_every_theme_year_combination():
    panel, _ = _panel()
    taxonomy = load_taxonomy(CONFIG / "taxonomy.yaml")
    fi_2020 = [r for r in panel if r.ticker == "FISV" and r.fiscal_year == 2020]
    assert len(fi_2020) == len(taxonomy.ids)


def test_salience_normalises_for_filing_length():
    panel, _ = _panel()
    row = next(r for r in panel if r.ticker == "FISV" and r.fiscal_year == 2020 and r.theme == THEME)
    assert row.mentions == 1 and row.total_chunks == 4
    assert row.salience == 0.25


def test_first_mention_identifies_the_leader_and_the_laggard():
    panel, peers = _panel()
    records = {f.ticker: f for f in first_mentions(panel, peers) if f.theme == THEME}
    assert records["FISV"].year == 2020
    assert records["GPN"].year == 2023


def test_ipo_year_first_mention_is_flagged_as_censored():
    """TOST's first filing is FY2021, so a 2021 first mention is not evidence
    that it led. Treating it as a leader would be an artefact of listing."""
    panel, peers = _panel()
    records = {f.ticker: f for f in first_mentions(panel, peers) if f.theme == THEME}
    assert records["TOST"].year == 2021
    assert records["TOST"].censored is True
    assert records["FISV"].censored is False


def test_peer_median_excludes_censored_companies():
    panel, peers = _panel()
    records = {f.ticker: f for f in first_mentions(panel, peers) if f.theme == THEME}
    # Median over full-history companies only: FI 2020, GPN 2023 -> 2021.5
    assert records["FISV"].lag_vs_peers == -1.5
    assert records["GPN"].lag_vs_peers == 1.5


def test_company_that_never_mentions_a_theme_has_no_year():
    panel, peers = _panel()
    record = next(
        f for f in first_mentions(panel, peers)
        if f.ticker == "GPN" and f.theme == "stablecoin_crypto_rails"
    )
    assert record.year is None and record.lag_vs_peers is None


def test_adoption_curve_denominator_excludes_pre_ipo_years():
    panel, peers = _panel()
    curve = {row["fiscal_year"]: row for row in adoption_curve(panel, THEME, peers)}
    # TOST is not public in 2020, so software_led has no covered companies.
    assert curve[2020]["software_led_n"] == 0
    assert curve[2021]["software_led_n"] == 1
    assert curve[2021]["software_led"] == 1.0


def test_adoption_curve_tracks_the_incumbent_shift():
    panel, peers = _panel()
    curve = {row["fiscal_year"]: row for row in adoption_curve(panel, THEME, peers)}
    assert curve[2020]["incumbent"] == 0.5   # FI yes, GPN no
    assert curve[2023]["incumbent"] == 1.0   # both


def test_escalation_table_needs_a_prior_year():
    panel, _ = _panel()
    rows = escalation_table(panel, THEME)
    assert all(r["fiscal_year"] > 2020 for r in rows)
    assert all("delta" in r for r in rows)


def test_financial_join_computes_cagr_and_keeps_censoring():
    panel, peers = _panel()
    first = first_mentions(panel, peers)
    revenue = {"FISV": {2020: 100.0, 2025: 200.0}, "GPN": {2020: 100.0, 2025: 100.0}}
    joined = {r["ticker"]: r for r in join_financials(
        first, revenue, THEME, start_year=2020, end_year=2025
    )}
    assert joined["FISV"]["revenue_cagr"] == round(2 ** (1 / 5) - 1, 4)
    assert joined["GPN"]["revenue_cagr"] == 0.0
    assert joined["TOST"]["censored"] is True


def test_financial_join_handles_missing_series():
    panel, peers = _panel()
    joined = join_financials(
        first_mentions(panel, peers), {}, THEME, start_year=2020, end_year=2025
    )
    assert all(r["revenue_cagr"] is None for r in joined)
