import csv
from pathlib import Path

import pytest

from pda.baseline import classify_chunk
from pda.chunks import Chunk
from pda.config import load_peers, load_taxonomy
from pda.goldset import GOLD_FIELDS, read_worksheet, stratified_sample, write_worksheet

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _taxonomy():
    return load_taxonomy(CONFIG / "taxonomy.yaml")


def _peers():
    return load_peers(CONFIG / "peers.yaml")


def test_taxonomy_ids_are_unique():
    ids = _taxonomy().ids
    assert len(ids) == len(set(ids))


def test_every_theme_has_a_definition_and_keywords():
    for theme in _taxonomy().themes:
        assert theme.definition.strip()
        assert theme.keywords


def test_peer_tickers_are_unique_and_cohorts_resolve():
    peers = _peers()
    assert len(peers.tickers) == len(set(peers.tickers))
    for ticker in peers.tickers:
        assert peers.cohort_of(ticker)


def test_partial_history_censoring_reads_back():
    peers = _peers()
    assert peers.first_covered_year("TOST") == 2021
    assert peers.first_covered_year("FI") == peers.study.start_fiscal_year


def test_invented_labels_are_stripped():
    taxonomy = _taxonomy()
    cleaned = taxonomy.validate_labels(["isv_channel_displacement", "made_up_theme"])
    assert cleaned == ["isv_channel_displacement"]


def test_baseline_matches_a_real_keyword():
    text = (
        "Merchants increasingly obtain processing bundled with vertical "
        "software, intensifying competition for our direct channel."
    )
    assert "isv_channel_displacement" in classify_chunk(text, _taxonomy())


def test_baseline_does_not_match_substrings():
    """'yield' must not fire on 'yielding' inside an unrelated sentence."""
    assert classify_chunk("The negotiation was unyielding.", _taxonomy()) == []


def test_baseline_abstains_on_boilerplate():
    text = (
        "The accompanying consolidated financial statements have been "
        "prepared in accordance with generally accepted accounting principles."
    )
    assert classify_chunk(text, _taxonomy()) == []


def _fake_chunks(n=60):
    out = []
    for i in range(n):
        ticker = ["FI", "TOST", "RELY"][i % 3]
        text = (
            "We face competition from integrated software vendors bundling "
            "payments into vertical software platforms. " * 3
            if i % 4 == 0
            else "General corporate boilerplate about our operations. " * 6
        )
        out.append(
            Chunk(
                chunk_id=f"{ticker}-{2021 + i % 3}-item1a-{i:04d}",
                ticker=ticker,
                fiscal_year=2021 + i % 3,
                item="item1a",
                ordinal=i,
                text=text,
                text_hash=f"h{i}",
                filing_accession="a",
                filing_date="2024-02-01",
            )
        )
    return out


def test_sampling_is_reproducible_and_balanced():
    peers, taxonomy = _peers(), _taxonomy()
    first = stratified_sample(_fake_chunks(), peers, taxonomy, target_n=20, seed=7)
    second = stratified_sample(_fake_chunks(), peers, taxonomy, target_n=20, seed=7)
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    assert len({r.ticker for r in first}) > 1


def test_sample_contains_both_strata():
    peers, taxonomy = _peers(), _taxonomy()
    rows = stratified_sample(_fake_chunks(), peers, taxonomy, target_n=20, seed=7)
    assert {r.stratum for r in rows} == {"enriched", "random"}


def test_worksheet_is_blind_by_default(tmp_path):
    peers, taxonomy = _peers(), _taxonomy()
    rows = stratified_sample(_fake_chunks(), peers, taxonomy, target_n=12, seed=7)
    path = tmp_path / "gold.csv"
    write_worksheet(rows, path)
    header = path.read_text().splitlines()[0]
    assert header.split(",") == GOLD_FIELDS

    parsed = read_worksheet(path, taxonomy)
    assert parsed, "worksheet round-tripped to nothing"
    # Blinding must hide the baseline's guesses even though the sampler
    # computed them -- otherwise you label by agreeing with regex.
    assert all(not r.baseline_labels for r in parsed)
    assert any(r.text for r in parsed)


def test_worksheet_rejects_typo_labels(tmp_path):
    peers, taxonomy = _peers(), _taxonomy()
    rows = stratified_sample(_fake_chunks(), peers, taxonomy, target_n=6, seed=7)
    path = tmp_path / "gold.csv"
    write_worksheet(rows, path)

    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    records[0]["gold_labels"] = "isv_channel_displacment"  # deliberate typo
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    with pytest.raises(ValueError):
        read_worksheet(path, taxonomy)


def test_baseline_matches_plural_forms():
    """Filings overwhelmingly use the plural; a singular-only keyword list
    silently under-counts the most important theme in the study."""
    text = "We compete with integrated software vendors across our verticals."
    assert "isv_channel_displacement" in classify_chunk(text, _taxonomy())
