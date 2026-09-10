"""Theme adoption analysis -- where the finding comes from.

The output of this module is the argument. Everything upstream exists to make
these tables trustworthy.

Three measures per company per theme:

*   first_mention_year -- the earliest fiscal year the theme appears. Censored
    for companies that IPO'd mid-window, because no filing is not the same as
    no mention, and treating a 2021 IPO as "silent in 2020" would manufacture
    a lead for every incumbent.
*   salience -- share of that year's in-scope paragraphs carrying the theme.
    Normalises for the fact that some filers write four times as much as
    others.
*   escalation -- year-over-year change in salience, restricted to paragraphs
    flagged as new language, which separates active escalation from text
    carried forward untouched.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from .chunks import Chunk
from .config import PeerSet, Taxonomy

CENSORED = None


@dataclass(frozen=True)
class ThemeYear:
    ticker: str
    cohort: str
    theme: str
    fiscal_year: int
    mentions: int
    new_mentions: int
    total_chunks: int

    @property
    def salience(self) -> float:
        return self.mentions / self.total_chunks if self.total_chunks else 0.0

    @property
    def new_salience(self) -> float:
        return self.new_mentions / self.total_chunks if self.total_chunks else 0.0


@dataclass(frozen=True)
class FirstMention:
    ticker: str
    cohort: str
    theme: str
    year: int | None
    censored: bool
    lag_vs_peers: float | None


def build_panel(
    chunks: Sequence[Chunk],
    labels: dict[str, Sequence[str]],
    peers: PeerSet,
    taxonomy: Taxonomy,
) -> list[ThemeYear]:
    """One row per ticker x theme x fiscal year."""
    totals: dict[tuple[str, int], int] = defaultdict(int)
    mentions: dict[tuple[str, int, str], int] = defaultdict(int)
    new_mentions: dict[tuple[str, int, str], int] = defaultdict(int)

    for chunk in chunks:
        totals[(chunk.ticker, chunk.fiscal_year)] += 1
        for theme in labels.get(chunk.chunk_id, []):
            mentions[(chunk.ticker, chunk.fiscal_year, theme)] += 1
            if chunk.is_new_language:
                new_mentions[(chunk.ticker, chunk.fiscal_year, theme)] += 1

    rows: list[ThemeYear] = []
    for (ticker, year), total in sorted(totals.items()):
        for theme in taxonomy.ids:
            rows.append(
                ThemeYear(
                    ticker=ticker,
                    cohort=peers.cohort_of(ticker),
                    theme=theme,
                    fiscal_year=year,
                    mentions=mentions.get((ticker, year, theme), 0),
                    new_mentions=new_mentions.get((ticker, year, theme), 0),
                    total_chunks=total,
                )
            )
    return rows


def first_mentions(
    panel: Sequence[ThemeYear],
    peers: PeerSet,
    *,
    min_mentions: int = 1,
) -> list[FirstMention]:
    """Earliest year each company crosses the mention threshold for a theme.

    `censored` marks companies whose first year of coverage is later than the
    study start. A censored company with a first mention in its debut year
    cannot be called a follower, because we never saw it be silent.
    """
    earliest: dict[tuple[str, str], int] = {}
    for row in panel:
        if row.mentions >= min_mentions:
            key = (row.ticker, row.theme)
            if key not in earliest or row.fiscal_year < earliest[key]:
                earliest[key] = row.fiscal_year

    tickers = sorted({row.ticker for row in panel})
    themes = sorted({row.theme for row in panel})

    by_theme: dict[str, list[int]] = defaultdict(list)
    for (ticker, theme), year in earliest.items():
        if peers.first_covered_year(ticker) <= peers.study.start_fiscal_year:
            by_theme[theme].append(year)

    out: list[FirstMention] = []
    for ticker in tickers:
        coverage_start = peers.first_covered_year(ticker)
        censored = coverage_start > peers.study.start_fiscal_year
        for theme in themes:
            year = earliest.get((ticker, theme))
            reference = by_theme.get(theme)
            lag = (
                round(year - median(reference), 2)
                if year is not None and reference
                else None
            )
            out.append(
                FirstMention(
                    ticker=ticker,
                    cohort=peers.cohort_of(ticker),
                    theme=theme,
                    year=year,
                    censored=censored and (year == coverage_start if year else False),
                    lag_vs_peers=lag,
                )
            )
    return out


def adoption_curve(
    panel: Sequence[ThemeYear], theme: str, peers: PeerSet
) -> list[dict]:
    """Share of covered companies mentioning a theme, by year and cohort.

    The denominator is companies with a filing that year, not the full peer
    set, so a 2021 IPO does not drag the 2020 adoption rate down.
    """
    years = sorted({row.fiscal_year for row in panel})
    cohorts = sorted({row.cohort for row in panel})

    out: list[dict] = []
    for year in years:
        record: dict = {"fiscal_year": year}
        for cohort in cohorts:
            rows = [
                r
                for r in panel
                if r.theme == theme
                and r.fiscal_year == year
                and r.cohort == cohort
                and peers.first_covered_year(r.ticker) <= year
            ]
            covered = len(rows)
            mentioning = sum(1 for r in rows if r.mentions > 0)
            record[cohort] = round(mentioning / covered, 3) if covered else None
            record[f"{cohort}_n"] = covered
        out.append(record)
    return out


def escalation_table(
    panel: Sequence[ThemeYear], theme: str, *, top: int = 15
) -> list[dict]:
    """Largest year-over-year jumps in new-language salience for one theme."""
    by_key = {(r.ticker, r.fiscal_year): r for r in panel if r.theme == theme}
    rows: list[dict] = []
    for (ticker, year), row in by_key.items():
        prior = by_key.get((ticker, year - 1))
        if prior is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "cohort": row.cohort,
                "fiscal_year": year,
                "salience": round(row.salience, 4),
                "prior_salience": round(prior.salience, 4),
                "delta": round(row.salience - prior.salience, 4),
                "new_language_mentions": row.new_mentions,
            }
        )
    return sorted(rows, key=lambda r: -r["delta"])[:top]


def join_financials(
    first: Sequence[FirstMention],
    metrics: dict[str, dict[int, float]],
    theme: str,
    *,
    start_year: int,
    end_year: int,
) -> list[dict]:
    """Line first-mention timing up against a financial series.

    This produces the correlation the deck argues from, and it is the weakest
    link in the whole study: fourteen companies is not a sample that supports
    a causal claim. The writeup must present this as suggestive and say what
    would be needed to test it properly.
    """
    out: list[dict] = []
    for record in first:
        if record.theme != theme or record.year is None:
            continue
        series = metrics.get(record.ticker, {})
        begin, finish = series.get(start_year), series.get(end_year)
        cagr = None
        if begin and finish and begin > 0 and end_year > start_year:
            cagr = round((finish / begin) ** (1 / (end_year - start_year)) - 1, 4)
        out.append(
            {
                "ticker": record.ticker,
                "cohort": record.cohort,
                "first_mention_year": record.year,
                "censored": record.censored,
                "lag_vs_peers": record.lag_vs_peers,
                "revenue_cagr": cagr,
            }
        )
    return sorted(out, key=lambda r: (r["first_mention_year"], r["ticker"]))
