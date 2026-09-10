"""SEC EDGAR access.

Three things this module is careful about, because they are the three things
that quietly invalidate filings research:

1.  Rate limiting. SEC asks for no more than 10 requests/second and a
    User-Agent that identifies you. Exceeding it gets your IP blocked.
2.  Point-in-time discipline. Every financial datapoint carries the date it
    was *filed*, not just the period it describes. Restatements mean the
    FY2022 revenue figure available in March 2023 is not always the FY2022
    revenue figure available today, and any analysis that mixes them is
    looking at information nobody had at the time.
3.  CIK resolution. Tickers change (FLEETCOR to Corpay, SQ to XYZ). CIKs do
    not. Everything downstream keys on CIK.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_ARCHIVE_DOC = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
)


class EdgarError(RuntimeError):
    pass


@dataclass(frozen=True)
class Filing:
    cik: int
    ticker: str
    form: str
    accession: str
    filing_date: str          # YYYY-MM-DD, the point-in-time anchor
    report_date: str          # period end
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        """Fiscal year of the period covered, taken from the report date.

        Not the filing year: a 10-K for FY2024 filed in Feb 2025 is FY2024.
        """
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return SEC_ARCHIVE_DOC.format(
            cik=self.cik,
            accession_nodash=self.accession.replace("-", ""),
            document=self.primary_document,
        )


@dataclass(frozen=True)
class MetricPoint:
    tag: str
    unit: str
    fiscal_year: int
    fiscal_period: str
    period_end: str
    value: float
    filed: str                # point-in-time: when this number became public
    accession: str
    form: str


class RateLimiter:
    """Simple monotonic-clock spacing. SEC's published ceiling is 10 req/s."""

    def __init__(self, requests_per_second: float = 8.0) -> None:
        self._min_interval = 1.0 / requests_per_second
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


@dataclass
class EdgarClient:
    """Thin EDGAR wrapper.

    `transport` exists so tests can inject httpx.MockTransport and exercise
    the parsing without touching the network.
    """

    user_agent: str
    cache_dir: Path | None = None
    requests_per_second: float = 8.0
    transport: httpx.BaseTransport | None = None
    _limiter: RateLimiter = field(init=False)
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        if not self.user_agent or "@" not in self.user_agent:
            raise EdgarError(
                "SEC requires a User-Agent containing a contact email, e.g. "
                "'Farhan Tariq farhantariq@ucsb.edu'. Set SEC_USER_AGENT."
            )
        self._limiter = RateLimiter(self.requests_per_second)
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            transport=self.transport,
            follow_redirects=True,
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ http

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe = url.replace("https://", "").replace("/", "_").replace("?", "_")
        return self.cache_dir / safe[:200]

    def get_text(self, url: str, *, use_cache: bool = True) -> str:
        cached = self._cache_path(url)
        if use_cache and cached and cached.exists():
            return cached.read_text(encoding="utf-8", errors="replace")

        self._limiter.wait()
        for attempt in range(4):
            response = self._client.get(url)
            if response.status_code == 200:
                text = response.text
                if cached:
                    cached.write_text(text, encoding="utf-8")
                return text
            if response.status_code in (429, 503):
                time.sleep(2**attempt)
                continue
            raise EdgarError(f"{response.status_code} fetching {url}")
        raise EdgarError(f"Gave up after retries fetching {url}")

    def get_json(self, url: str, *, use_cache: bool = True) -> Any:
        return json.loads(self.get_text(url, use_cache=use_cache))

    # ------------------------------------------------------------------- api

    def resolve_cik(self, ticker: str) -> int:
        """Look up CIK from SEC's own ticker map. Never hardcode CIKs."""
        payload = self.get_json(SEC_TICKER_MAP)
        wanted = ticker.upper()
        for row in payload.values():
            if str(row.get("ticker", "")).upper() == wanted:
                return int(row["cik_str"])
        raise EdgarError(
            f"Ticker {ticker} not found in SEC ticker map. If the company "
            "renamed or delisted, look up its CIK on EDGAR and pin it."
        )

    def list_filings(
        self,
        cik: int,
        ticker: str,
        *,
        forms: Iterable[str] = ("10-K",),
        start_year: int = 2019,
        end_year: int = 2100,
    ) -> list[Filing]:
        payload = self.get_json(SEC_SUBMISSIONS.format(cik=cik))
        recent = payload.get("filings", {}).get("recent", {})
        wanted_forms = {f.upper() for f in forms}

        rows = zip(  # noqa: B905 - EDGAR pads these arrays inconsistently
            recent.get("form", []),
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("primaryDocument", []),
        )

        out: list[Filing] = []
        for form, accession, filing_date, report_date, document in rows:
            if form.upper() not in wanted_forms:
                continue
            if not report_date:
                continue
            fiscal_year = int(report_date[:4])
            if not (start_year <= fiscal_year <= end_year):
                continue
            out.append(
                Filing(
                    cik=cik,
                    ticker=ticker.upper(),
                    form=form,
                    accession=accession,
                    filing_date=filing_date,
                    report_date=report_date,
                    primary_document=document,
                )
            )
        return sorted(out, key=lambda f: f.report_date)

    def fetch_document(self, filing: Filing) -> str:
        return self.get_text(filing.url)

    def company_facts(self, cik: int) -> dict:
        return self.get_json(SEC_COMPANY_FACTS.format(cik=cik))


def extract_annual_metric(
    facts: dict,
    tag: str,
    *,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
) -> list[MetricPoint]:
    """Pull annual (FY) datapoints for one XBRL tag, deduplicated.

    A single fiscal year appears many times in companyfacts: once in the
    original 10-K, again as a comparative in the next two years' filings, and
    again after any restatement. We keep the earliest-filed value for each
    period end, which is the number the market actually had at the time.
    """
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return []
    entries = node.get("units", {}).get(unit, [])

    best: dict[str, MetricPoint] = {}
    for entry in entries:
        if entry.get("fp") != "FY" or entry.get("form") not in ("10-K", "10-K/A"):
            continue
        end = entry.get("end")
        filed = entry.get("filed")
        if not end or not filed:
            continue
        point = MetricPoint(
            tag=tag,
            unit=unit,
            fiscal_year=int(entry.get("fy") or end[:4]),
            fiscal_period="FY",
            period_end=end,
            value=float(entry["val"]),
            filed=filed,
            accession=entry.get("accn", ""),
            form=entry.get("form", ""),
        )
        existing = best.get(end)
        if existing is None or point.filed < existing.filed:
            best[end] = point

    return sorted(best.values(), key=lambda p: p.period_end)


def first_available_tag(
    facts: dict, candidates: Iterable[str], *, unit: str = "USD"
) -> tuple[str, list[MetricPoint]]:
    """Try several XBRL tags and return the first that has data.

    Filers are inconsistent: some use Revenues, some RevenueFromContractWith
    CustomerExcludingAssessedTax, some both. Silently picking one produces
    gaps that look like business changes.
    """
    for tag in candidates:
        points = extract_annual_metric(facts, tag, unit=unit)
        if points:
            return tag, points
    return "", []


REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)

OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)
NET_INCOME_TAGS = ("NetIncomeLoss",)
