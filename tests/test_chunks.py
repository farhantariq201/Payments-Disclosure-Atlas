from pathlib import Path

from pda.chunks import chunk_section, jaccard, mark_novelty, text_hash
from pda.sections import extract_sections

FIXTURES = Path(__file__).parent / "fixtures"


def _chunks(filename: str, year: int):
    html = (FIXTURES / filename).read_text()
    sections = extract_sections(html, ["item1a"])
    return chunk_section(
        sections["item1a"].text, ticker="EXPY", fiscal_year=year, item="item1a"
    )


def test_chunk_ids_are_stable_and_unique():
    chunks = _chunks("mini_10k_fy2023.html", 2023)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith("EXPY-2023-item1a-") for c in chunks)


def test_hash_ignores_number_changes():
    """A paragraph carried forward with updated figures is not new language."""
    a = "Revenue was $1,000 million in the period and volume grew accordingly."
    b = "Revenue was $1,250 million in the period and volume grew accordingly."
    assert text_hash(a) == text_hash(b)


def test_hash_detects_real_edits():
    a = "We face competition from integrated software vendors in our channel."
    b = "We face competition from banks and traditional acquirers in our channel."
    assert text_hash(a) != text_hash(b)


def test_novelty_flags_only_the_changed_paragraph():
    prior = _chunks("mini_10k_fy2023.html", 2023)
    current = mark_novelty(_chunks("mini_10k_fy2024.html", 2024), prior)
    new = [c for c in current if c.is_new_language]
    assert len(new) == 1
    assert "stablecoin" in new[0].text.lower()


def test_carried_forward_paragraphs_are_not_new():
    prior = _chunks("mini_10k_fy2023.html", 2023)
    current = mark_novelty(_chunks("mini_10k_fy2024.html", 2024), prior)
    carried = [c for c in current if not c.is_new_language]
    assert carried
    assert all(c.similarity_to_prior >= 0.6 for c in carried)


def test_first_year_has_no_prior_so_everything_is_new():
    current = mark_novelty(_chunks("mini_10k_fy2023.html", 2023), [])
    assert all(c.is_new_language for c in current)


def test_jaccard_bounds():
    assert jaccard("", "") == 0.0
    text = "the company processes payments for merchants across many verticals"
    assert jaccard(text, text) == 1.0
    assert 0.0 <= jaccard(text, "entirely different subject matter here") < 0.5


def test_short_fragments_are_dropped():
    chunks = chunk_section("Too short.", ticker="X", fiscal_year=2023, item="item1a")
    assert chunks == []
