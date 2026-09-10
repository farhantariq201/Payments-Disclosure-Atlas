"""Markdown memo rendering: one per company, plus the cross-cut findings deck
skeleton.

Deliberately Markdown rather than PDF. The memo is an input to a deck you
write yourself, and the analysis is the contribution -- a generated PDF with a
logo on it would add polish to something that has not earned it yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import PeerSet, Taxonomy
from .timeline import FirstMention, ThemeYear, adoption_curve, escalation_table


def _table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append(
            "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        )
    return "\n".join(out)


@dataclass
class CompanyMemo:
    ticker: str
    name: str
    cohort: str
    fiscal_years: list[int]
    financials: dict[str, dict[int, float]]
    theme_rows: list[ThemeYear]
    quotes: list[tuple[str, int, str]]

    def render(self, taxonomy: Taxonomy, top_themes: int = 6) -> str:
        parts = [f"# {self.name} ({self.ticker})", f"_Cohort: {self.cohort}_", ""]

        parts.append("## Reported financials")
        parts.append(
            "Figures are as first filed, not as later restated. Source is the "
            "company's own XBRL data on EDGAR.\n"
        )
        metric_names = list(self.financials.keys())
        rows = []
        for year in self.fiscal_years:
            row = [year]
            for metric in metric_names:
                value = self.financials.get(metric, {}).get(year)
                row.append(f"{value/1e6:,.0f}" if value is not None else None)
            rows.append(row)
        parts.append(_table(["FY"] + [f"{m} ($m)" for m in metric_names], rows))
        parts.append("")

        parts.append("## Theme salience by year")
        parts.append(
            "Salience is the share of in-scope paragraphs carrying the theme. "
            "New-language counts exclude paragraphs carried forward from the "
            "prior year with only cosmetic edits.\n"
        )
        totals = {}
        for row in self.theme_rows:
            totals[row.theme] = totals.get(row.theme, 0) + row.mentions
        ranked = [t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])][
            :top_themes
        ]

        table_rows = []
        for theme in ranked:
            row = [taxonomy.get(theme).name]
            for year in self.fiscal_years:
                match = next(
                    (
                        r
                        for r in self.theme_rows
                        if r.theme == theme and r.fiscal_year == year
                    ),
                    None,
                )
                if match is None:
                    row.append(None)
                else:
                    marker = "*" if match.new_mentions else ""
                    row.append(f"{match.salience:.3f}{marker}")
            table_rows.append(row)
        parts.append(_table(["Theme"] + [str(y) for y in self.fiscal_years], table_rows))
        parts.append("\n`*` indicates new or materially rewritten language that year.\n")

        if self.quotes:
            parts.append("## Language worth reading")
            for theme, year, text in self.quotes:
                excerpt = text if len(text) <= 400 else text[:397] + "..."
                parts.append(
                    f"**{taxonomy.get(theme).name}, FY{year}**\n\n> {excerpt}\n"
                )

        parts.append("## Open questions for primary research")
        parts.append(
            "- What does the salience jump above correspond to operationally?\n"
            "- Does management describe this differently on earnings calls "
            "than in the 10-K?\n"
            "- Which of these themes did competitors name first, and did it "
            "show up in results?\n"
        )
        return "\n".join(parts)


def render_crosscut(
    panel: Sequence[ThemeYear],
    first: Sequence[FirstMention],
    peers: PeerSet,
    taxonomy: Taxonomy,
    headline_theme: str,
) -> str:
    theme_name = taxonomy.get(headline_theme).name
    parts = [
        f"# Cross-company findings: {theme_name}",
        "",
        "## 1. Adoption curve",
        "Share of companies with a filing that year that mention the theme. "
        "Denominators exclude companies not yet public.",
        "",
    ]
    curve = adoption_curve(panel, headline_theme, peers)
    cohorts = [c.id for c in peers.cohorts]
    headers = ["FY"] + [f"{c} (n)" for c in cohorts]
    rows = [
        [record["fiscal_year"]]
        + [
            f"{record.get(c)} ({record.get(f'{c}_n')})"
            if record.get(c) is not None
            else "-"
            for c in cohorts
        ]
        for record in curve
    ]
    parts.append(_table(headers, rows))

    parts += ["", "## 2. Who named it first", ""]
    relevant = [f for f in first if f.theme == headline_theme]
    relevant.sort(key=lambda f: (f.year is None, f.year or 9999, f.ticker))
    parts.append(
        _table(
            ["Ticker", "Cohort", "First mention", "Lag vs peer median", "Censored"],
            [
                [
                    f.ticker,
                    f.cohort,
                    f.year if f.year else "never",
                    f.lag_vs_peers,
                    "yes" if f.censored else "",
                ]
                for f in relevant
            ],
        )
    )
    parts.append(
        "\nCensored means the company's first filing is later than the study "
        "start, so an early first mention cannot be read as leadership.\n"
    )

    parts += ["## 3. Largest escalations", ""]
    escalations = escalation_table(panel, headline_theme)
    parts.append(
        _table(
            ["Ticker", "FY", "Salience", "Prior", "Delta", "New-language mentions"],
            [
                [
                    r["ticker"],
                    r["fiscal_year"],
                    r["salience"],
                    r["prior_salience"],
                    r["delta"],
                    r["new_language_mentions"],
                ]
                for r in escalations
            ],
        )
    )

    parts += [
        "",
        "## 4. Findings",
        "",
        "_Write these yourself. Each one needs a number from the tables above, "
        "a mechanism, and a named company. Anything you cannot attach all "
        "three to is an observation, not a finding, and does not go in the "
        "deck._",
        "",
        "## 5. What would have to be true for this to be wrong",
        "",
        "- Disclosure language may lag internal strategy by years, so silence "
        "is weak evidence of inattention.",
        "- Risk factor drafting is partly a legal exercise; counsel's appetite "
        "varies by company and confounds the timing.",
        "- Fourteen companies cannot support a causal claim about outcomes.",
        "",
    ]
    return "\n".join(parts)
