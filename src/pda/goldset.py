"""Gold-set construction: stratified sampling and the labelling worksheet.

Sampling is stratified by cohort x fiscal year x item and seeded, so the gold
set is reproducible and does not over-represent whichever company writes the
longest risk factors. Fiserv's Item 1A alone would otherwise dominate a random
draw.

Enrichment: pure random sampling of 10-K paragraphs yields a gold set that is
roughly 90% "no theme", which produces a benchmark with almost no positives
for the rarer themes and error bars wide enough to drive a truck through. A
share of the sample is therefore drawn from chunks the keyword baseline flags.
That share is recorded on every row so the benchmark can report enriched and
random strata separately -- reporting only the enriched number would overstate
real-world precision, and the writeup has to say so.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .baseline import classify_chunk
from .chunks import Chunk
from .config import PeerSet, Taxonomy

GOLD_FIELDS = [
    "chunk_id",
    "ticker",
    "cohort",
    "fiscal_year",
    "item",
    "stratum",
    "is_new_language",
    "baseline_labels",
    "gold_labels",
    "notes",
    "text",
]


@dataclass(frozen=True)
class GoldRow:
    chunk_id: str
    ticker: str
    cohort: str
    fiscal_year: int
    item: str
    stratum: str
    is_new_language: bool
    baseline_labels: list[str]
    gold_labels: list[str]
    notes: str
    text: str


def stratified_sample(
    chunks: Sequence[Chunk],
    peers: PeerSet,
    taxonomy: Taxonomy,
    *,
    target_n: int = 400,
    enriched_share: float = 0.5,
    seed: int = 20260101,
) -> list[GoldRow]:
    rng = random.Random(seed)

    enriched: list[Chunk] = []
    plain: list[Chunk] = []
    for chunk in chunks:
        (enriched if classify_chunk(chunk.text, taxonomy) else plain).append(chunk)

    def draw(pool: list[Chunk], quota: int) -> list[Chunk]:
        buckets: dict[tuple, list[Chunk]] = defaultdict(list)
        for chunk in pool:
            key = (peers.cohort_of(chunk.ticker), chunk.fiscal_year, chunk.item)
            buckets[key].append(chunk)
        if not buckets:
            return []
        keys = sorted(buckets)
        for key in keys:
            rng.shuffle(buckets[key])

        picked: list[Chunk] = []
        cursor = 0
        while len(picked) < quota and any(buckets[k] for k in keys):
            key = keys[cursor % len(keys)]
            if buckets[key]:
                picked.append(buckets[key].pop())
            cursor += 1
        return picked

    n_enriched = int(round(target_n * enriched_share))
    selected = [(c, "enriched") for c in draw(enriched, n_enriched)]
    selected += [(c, "random") for c in draw(plain, target_n - len(selected))]

    rows = [
        GoldRow(
            chunk_id=chunk.chunk_id,
            ticker=chunk.ticker,
            cohort=peers.cohort_of(chunk.ticker),
            fiscal_year=chunk.fiscal_year,
            item=chunk.item,
            stratum=stratum,
            is_new_language=chunk.is_new_language,
            baseline_labels=classify_chunk(chunk.text, taxonomy),
            gold_labels=[],
            notes="",
            text=chunk.text,
        )
        for chunk, stratum in selected
    ]
    rng.shuffle(rows)  # label in random order to blunt anchoring
    return rows


def write_worksheet(rows: Sequence[GoldRow], path: Path, *, blind: bool = True) -> None:
    """Write the CSV you actually label in.

    `blind` blanks the baseline's guesses. Seeing them first is the fastest
    way to contaminate your own gold set: you end up agreeing with the
    keywords instead of reading the paragraph, and the benchmark then measures
    how well the model imitates regex.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "chunk_id": row.chunk_id,
                    "ticker": row.ticker,
                    "cohort": row.cohort,
                    "fiscal_year": row.fiscal_year,
                    "item": row.item,
                    "stratum": row.stratum,
                    "is_new_language": int(row.is_new_language),
                    "baseline_labels": "" if blind else "|".join(row.baseline_labels),
                    "gold_labels": "",
                    "notes": "",
                    "text": row.text,
                }
            )


def read_worksheet(path: Path, taxonomy: Taxonomy) -> list[GoldRow]:
    rows: list[GoldRow] = []
    unknown: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            raw = [
                label.strip()
                for label in (record.get("gold_labels") or "").split("|")
                if label.strip()
            ]
            unknown |= {label for label in raw if label not in set(taxonomy.ids)}
            rows.append(
                GoldRow(
                    chunk_id=record["chunk_id"],
                    ticker=record["ticker"],
                    cohort=record.get("cohort", ""),
                    fiscal_year=int(record["fiscal_year"]),
                    item=record["item"],
                    stratum=record.get("stratum", "random"),
                    is_new_language=bool(int(record.get("is_new_language") or 0)),
                    baseline_labels=[
                        label
                        for label in (record.get("baseline_labels") or "").split("|")
                        if label
                    ],
                    gold_labels=taxonomy.validate_labels(raw),
                    notes=record.get("notes", ""),
                    text=record["text"],
                )
            )
    if unknown:
        raise ValueError(
            "Worksheet contains theme ids not in the taxonomy: "
            + ", ".join(sorted(unknown))
            + ". Fix the typo or add the theme and bump the taxonomy version."
        )
    return rows


def labelled_only(rows: Sequence[GoldRow]) -> list[GoldRow]:
    """Rows you have actually worked through.

    A blank `gold_labels` cell is ambiguous: it means either 'no theme applies'
    or 'not labelled yet'. The convention is that NONE is written explicitly
    for a deliberate empty label, so blanks can be excluded rather than
    silently counted as negatives -- which would inflate precision.
    """
    out: list[GoldRow] = []
    for row in rows:
        marker = (row.notes or "").strip().upper()
        if row.gold_labels or marker == "NONE":
            out.append(row)
    return out
