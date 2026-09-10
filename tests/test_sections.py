from pathlib import Path

from pda.sections import extract_sections, html_to_text, split_items

FIXTURES = Path(__file__).parent / "fixtures"


def _html() -> str:
    return (FIXTURES / "mini_10k_fy2023.html").read_text()


def test_toc_entries_do_not_win_over_real_sections():
    """The table of contents lists every item. The real Item 1A must win."""
    sections = extract_sections(_html(), ["item1a"])
    assert "item1a" in sections
    body = sections["item1a"].text
    assert "integrated software vendors" in body
    # A TOC match would produce a stub of a few dozen characters.
    assert len(body) > 2000


def test_cross_reference_does_not_win():
    """Item 7 contains 'See Item 1A. Risk Factors' -- must not become Item 1A."""
    sections = extract_sections(_html(), ["item1a", "item7"])
    assert "sponsor bank" in sections["item1a"].text
    assert "Revenue increased" in sections["item7"].text


def test_sections_do_not_bleed_into_each_other():
    sections = extract_sections(_html(), ["item1", "item1a", "item7"])
    assert "payment facilitator" in sections["item1"].text
    assert "payment facilitator" not in sections["item1a"].text
    assert "integrated software vendors" not in sections["item7"].text


def test_short_sections_are_dropped_not_returned_empty():
    """Item 1B is a one-liner. Callers must see it as absent, not empty."""
    sections = extract_sections(_html(), ["item1b"], min_chars=5000)
    assert "item1b" not in sections


def test_missing_item_is_absent_rather_than_silent():
    sections = extract_sections("<html><body><p>nothing here</p></body></html>")
    assert sections == {}


def test_html_to_text_preserves_block_boundaries():
    text = html_to_text("<p>first para</p><p>second para</p>")
    assert "first para" in text and "second para" in text
    assert "first parasecond" not in text


def test_entities_decoded():
    assert "&nbsp;" not in html_to_text("<p>a&nbsp;b</p>")
    assert "AT&T" in html_to_text("<p>AT&amp;T</p>")


def test_split_items_is_deterministic():
    text = html_to_text(_html())
    assert split_items(text).keys() == split_items(text).keys()
