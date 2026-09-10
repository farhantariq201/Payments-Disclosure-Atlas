"""Keyword baseline classifier.

This exists so the LLM's accuracy number means something. "Our classifier hits
0.78 F1" is unfalsifiable on its own. "Our classifier hits 0.78 F1 against a
keyword baseline at 0.44, on the same held-out set" is a result. It also
occasionally wins on a theme, which is worth knowing before spending money on
tokens.
"""

from __future__ import annotations

import re
from functools import cache

from .config import Taxonomy, load_taxonomy


@cache
def _compiled(taxonomy_version: str, theme_id: str, keywords: tuple[str, ...]):
    if not keywords:
        return None
    alternatives = "|".join(re.escape(keyword) for keyword in keywords)
    # Optional plural suffix: filings say "integrated software vendors" far more
    # often than the singular. The lookbehind still prevents matching inside a
    # longer word ("yield" must not fire on "unyielding").
    return re.compile(rf"(?<![a-z0-9])({alternatives})(?:es|s)?(?![a-z0-9])", re.IGNORECASE)


def classify_chunk(text: str, taxonomy: Taxonomy | None = None) -> list[str]:
    tax = taxonomy or load_taxonomy()
    hits: list[str] = []
    for theme in tax.themes:
        pattern = _compiled(tax.version, theme.id, tuple(theme.keywords))
        if pattern and pattern.search(text):
            hits.append(theme.id)
    return sorted(hits)


def classify_many(
    texts: list[str], taxonomy: Taxonomy | None = None
) -> list[list[str]]:
    tax = taxonomy or load_taxonomy()
    return [classify_chunk(text, tax) for text in texts]
