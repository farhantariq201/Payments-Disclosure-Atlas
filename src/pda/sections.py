"""Split a 10-K into its Items.

The hard part is not finding "Item 1A. Risk Factors" — it is finding the
*right* one. Every 10-K mentions each item heading at least twice: once in the
table of contents and once at the actual section. Many mention them a third
time in cross-references ("see Item 1A above"). Naively taking the first match
gives you the table of contents and a section body of roughly forty words.

The heuristic here: find every candidate heading, then keep the occurrence
that opens the largest block of text before the next heading. Table of
contents entries sit a few dozen characters apart; real sections are thousands
of characters long. This is simple, it is testable, and it fails loudly rather
than silently returning a stub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape as _unescape

# Ordered so that "item 7a" is tried before "item 7" when scanning.
ITEM_PATTERNS: list[tuple[str, str]] = [
    ("item1", r"item\s*1\s*[\.\:\-–—]?\s*business"),
    ("item1a", r"item\s*1a\s*[\.\:\-–—]?\s*risk\s*factors"),
    ("item1b", r"item\s*1b\s*[\.\:\-–—]?\s*unresolved"),
    ("item2", r"item\s*2\s*[\.\:\-–—]?\s*properties"),
    ("item3", r"item\s*3\s*[\.\:\-–—]?\s*legal\s*proceedings"),
    ("item5", r"item\s*5\s*[\.\:\-–—]?\s*market\s*for"),
    ("item7", r"item\s*7\s*[\.\:\-–—,]?\s*management\W{0,3}s?\W{0,3}discussion"),
    ("item7a", r"item\s*7a\s*[\.\:\-–—]?\s*quantitative"),
    ("item8", r"item\s*8\s*[\.\:\-–—]?\s*financial\s*statements"),
    ("item9a", r"item\s*9a\s*[\.\:\-–—]?\s*controls"),
]

# Minimum body length for a match to be treated as a real section rather than
# a table-of-contents line. Risk factor sections run to tens of thousands of
# characters; even a terse Item 2 runs to several hundred.
MIN_SECTION_CHARS = 1500

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_RE = re.compile(
    r"</?(p|div|tr|table|br|h[1-6]|li|ul|ol|section)\b[^>]*>", re.IGNORECASE
)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")

_PUNCTUATION = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
}


@dataclass(frozen=True)
class Section:
    item: str
    text: str
    start: int
    end: int

    @property
    def char_count(self) -> int:
        return len(self.text)


def html_to_text(html: str) -> str:
    """Flatten filing HTML to text, preserving block boundaries as newlines."""
    out = _SCRIPT_RE.sub(" ", html)
    out = _BLOCK_RE.sub("\n", out)
    out = _TAG_RE.sub(" ", out)
    # Decode entities properly. Filers encode the apostrophe in
    # "Management's Discussion" as &#8217; -- deleting numeric entities
    # instead of decoding them silently breaks Item 7 detection.
    out = _unescape(out)
    for source, replacement in _PUNCTUATION.items():
        out = out.replace(source, replacement)
    out = _WS_RE.sub(" ", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    return _NL_RE.sub("\n\n", out).strip()


def _candidate_positions(text: str) -> list[tuple[int, str, int]]:
    """All (start, item, end_of_heading) matches across the document."""
    lowered = text.lower()
    found: list[tuple[int, str, int]] = []
    for item, pattern in ITEM_PATTERNS:
        for match in re.finditer(pattern, lowered):
            found.append((match.start(), item, match.end()))
    return sorted(found)


def split_items(text: str, *, min_chars: int = MIN_SECTION_CHARS) -> dict[str, Section]:
    """Return the best occurrence of each item, keyed by item id.

    "Best" means the occurrence with the longest run of text before the next
    heading of any kind, which is what distinguishes a real section from a
    contents line or a cross-reference.
    """
    candidates = _candidate_positions(text)
    if not candidates:
        return {}

    boundaries = [pos for pos, _, _ in candidates] + [len(text)]

    scored: dict[str, tuple[int, Section]] = {}
    for index, (_start, item, heading_end) in enumerate(candidates):
        next_boundary = boundaries[index + 1]
        body = text[heading_end:next_boundary].strip()
        length = len(body)
        current = scored.get(item)
        if current is None or length > current[0]:
            scored[item] = (
                length,
                Section(item=item, text=body, start=heading_end, end=next_boundary),
            )

    return {
        item: section
        for item, (length, section) in scored.items()
        if length >= min_chars
    }


def extract_sections(
    html: str, wanted: list[str] | None = None, *, min_chars: int = MIN_SECTION_CHARS
) -> dict[str, Section]:
    """Full pipeline: raw filing HTML to the sections you asked for.

    Missing items come back absent rather than empty, so callers must decide
    what to do about a filing whose Item 1A could not be located instead of
    quietly analysing nothing.
    """
    text = html_to_text(html)
    sections = split_items(text, min_chars=min_chars)
    if wanted is None:
        return sections
    return {item: sections[item] for item in wanted if item in sections}
