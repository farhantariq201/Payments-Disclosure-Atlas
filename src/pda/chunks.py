"""Turn sections into labelable paragraph chunks.

Includes year-over-year novelty detection, which matters more than it sounds.
Risk factor sections are heavily copy-pasted between years: a company can
carry the same paragraph forward for four years untouched. If you count raw
mentions, a company that wrote a paragraph in 2020 and never revisited it
looks identical to one actively escalating a theme. Flagging which paragraphs
are new or materially rewritten separates "still says this" from "just started
saying this", and the second is the interesting signal.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 4000

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z(])")
_NORMALISE = re.compile(r"[^a-z0-9 ]+")
_NUMERIC = re.compile(r"\b\d[\d,\.]*\b")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    fiscal_year: int
    item: str
    ordinal: int
    text: str
    text_hash: str
    filing_accession: str
    filing_date: str
    is_new_language: bool = False
    similarity_to_prior: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, and blank out numbers.

    Numbers are removed before comparison so that a paragraph carried forward
    with only its dollar figures updated still reads as unchanged language.
    """
    lowered = _NUMERIC.sub(" 0 ", text.lower())
    return " ".join(_NORMALISE.sub(" ", lowered).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()[:16]


def _shingles(text: str, size: int = 5) -> set[str]:
    tokens = _normalise(text).split()
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: str, b: str) -> float:
    left, right = _shingles(a), _shingles(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _split_paragraphs(text: str) -> list[str]:
    raw = [block.strip() for block in text.split("\n") if block.strip()]
    merged: list[str] = []
    buffer = ""
    for block in raw:
        candidate = f"{buffer} {block}".strip() if buffer else block
        if len(candidate) < MIN_CHUNK_CHARS:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""
    if buffer and merged:
        merged[-1] = f"{merged[-1]} {buffer}".strip()
    elif buffer:
        merged.append(buffer)
    return merged


def _hard_wrap(paragraph: str) -> list[str]:
    """Split paragraphs that exceed the size cap on sentence boundaries."""
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENT_SPLIT.split(paragraph):
        if len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_section(
    text: str,
    *,
    ticker: str,
    fiscal_year: int,
    item: str,
    filing_accession: str = "",
    filing_date: str = "",
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for paragraph in _split_paragraphs(text):
        for piece in _hard_wrap(paragraph):
            if len(piece) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{ticker}-{fiscal_year}-{item}-{ordinal:04d}",
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    item=item,
                    ordinal=ordinal,
                    text=piece,
                    text_hash=text_hash(piece),
                    filing_accession=filing_accession,
                    filing_date=filing_date,
                )
            )
            ordinal += 1
    return chunks


def mark_novelty(
    current: Iterable[Chunk],
    prior: Iterable[Chunk],
    *,
    threshold: float = 0.6,
) -> list[Chunk]:
    """Flag chunks that are new or materially rewritten versus the prior year.

    Exact-hash matches are resolved first and cheaply. Everything else is
    compared by 5-gram Jaccard against the prior year's chunks in the same
    item, which catches paragraphs that were edited rather than replaced.
    """
    prior_list = list(prior)
    prior_hashes = {chunk.text_hash for chunk in prior_list}
    by_item: dict[str, list[Chunk]] = {}
    for chunk in prior_list:
        by_item.setdefault(chunk.item, []).append(chunk)

    out: list[Chunk] = []
    for chunk in current:
        if chunk.text_hash in prior_hashes:
            out.append(
                Chunk(**{**chunk.to_dict(), "is_new_language": False, "similarity_to_prior": 1.0})
            )
            continue

        best = 0.0
        for candidate in by_item.get(chunk.item, []):
            best = max(best, jaccard(chunk.text, candidate.text))
            if best >= threshold:
                break

        out.append(
            Chunk(
                **{
                    **chunk.to_dict(),
                    "is_new_language": best < threshold,
                    "similarity_to_prior": round(best, 4),
                }
            )
        )
    return out
