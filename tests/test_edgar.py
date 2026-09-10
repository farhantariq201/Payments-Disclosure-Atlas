import json
from pathlib import Path

import httpx
import pytest

from pda.edgar import (
    REVENUE_TAGS,
    EdgarClient,
    EdgarError,
    extract_annual_metric,
    first_available_tag,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_client() -> EdgarClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers" in url:
            return httpx.Response(200, text=(FIXTURES / "company_tickers.json").read_text())
        if "submissions" in url:
            return httpx.Response(200, text=(FIXTURES / "submissions.json").read_text())
        if "companyfacts" in url:
            return httpx.Response(200, text=(FIXTURES / "companyfacts.json").read_text())
        return httpx.Response(200, text="<html><body>doc</body></html>")

    return EdgarClient(
        user_agent="Test User test@example.com",
        transport=httpx.MockTransport(handler),
        requests_per_second=1000.0,
    )


def test_user_agent_without_email_is_rejected():
    """SEC blocks anonymous traffic; failing loudly beats getting banned."""
    with pytest.raises(EdgarError):
        EdgarClient(user_agent="just-a-name")


def test_resolve_cik_from_sec_ticker_map():
    assert _mock_client().resolve_cik("expy") == 1234567


def test_unknown_ticker_raises_rather_than_returning_none():
    with pytest.raises(EdgarError):
        _mock_client().resolve_cik("NOPE")


def test_list_filings_filters_form_and_year():
    filings = _mock_client().list_filings(
        1234567, "EXPY", forms=("10-K",), start_year=2022, end_year=2025
    )
    assert [f.fiscal_year for f in filings] == [2023, 2024]
    assert all(f.form == "10-K" for f in filings)


def test_fiscal_year_comes_from_report_date_not_filing_date():
    """A FY2024 10-K filed in Feb 2025 is FY2024."""
    filings = _mock_client().list_filings(1234567, "EXPY", start_year=2024)
    filing = next(f for f in filings if f.filing_date == "2025-02-20")
    assert filing.fiscal_year == 2024


def test_filing_url_strips_dashes_from_accession():
    filings = _mock_client().list_filings(1234567, "EXPY", start_year=2024)
    assert "000123456725000010" in filings[0].url


def test_restated_figures_resolve_to_the_originally_filed_value():
    """FY2022 revenue was 1000 as filed and 1050 as later restated.

    The number the market had in Feb 2023 is 1000. Using 1050 in a study of
    what companies disclosed when is lookahead bias.
    """
    facts = json.loads((FIXTURES / "companyfacts.json").read_text())
    points = extract_annual_metric(facts, "Revenues")
    fy2022 = next(p for p in points if p.period_end == "2022-12-31")
    assert fy2022.value == 1000.0
    assert fy2022.filed == "2023-02-20"


def test_quarterly_and_non_10k_rows_are_excluded():
    facts = json.loads((FIXTURES / "companyfacts.json").read_text())
    points = extract_annual_metric(facts, "Revenues")
    assert all(p.fiscal_period == "FY" for p in points)
    assert all(p.period_end != "2023-06-30" for p in points)


def test_first_available_tag_falls_through_to_a_populated_tag():
    facts = json.loads((FIXTURES / "companyfacts.json").read_text())
    tag, points = first_available_tag(facts, REVENUE_TAGS)
    assert tag == "Revenues"
    assert points


def test_first_available_tag_returns_empty_when_nothing_matches():
    tag, points = first_available_tag({"facts": {}}, ("NoSuchTag",))
    assert tag == "" and points == []
