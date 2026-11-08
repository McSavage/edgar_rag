"""Period classification, fiscal labels and derived quarters (no network or DB needed)."""

from datetime import date

from edgar_rag.facts import classify_period, derive_quarters, fiscal_labels, normalize_unit
from edgar_rag.narrative import split_long_paragraphs, toc_sections


def test_classify_period_calendar_year():
    assert classify_period(None, date(2025, 12, 31), 12) == "instant"
    assert classify_period(date(2025, 10, 1), date(2025, 12, 31), 12) == "Q"
    assert classify_period(date(2025, 1, 1), date(2025, 6, 30), 12) == "6M"
    assert classify_period(date(2025, 1, 1), date(2025, 9, 30), 12) == "9M"
    assert classify_period(date(2025, 1, 1), date(2025, 12, 31), 12) == "FY"


def test_trailing_twelve_months_is_not_a_fiscal_year():
    # Amazon tags TTM cash flows, e.g. Apr 2024 - Mar 2025.
    assert classify_period(date(2024, 4, 1), date(2025, 3, 31), 12) == "TTM"


def test_classify_period_june_fiscal_year():
    assert classify_period(date(2024, 7, 1), date(2024, 12, 31), 6) == "6M"
    assert classify_period(date(2024, 7, 1), date(2025, 6, 30), 6) == "FY"


def test_fiscal_labels():
    assert fiscal_labels(date(2024, 12, 31), 6) == (2025, 2)   # MSFT FY25 Q2
    assert fiscal_labels(date(2025, 8, 31), 5) == (2026, 1)    # ORCL FY26 Q1
    assert fiscal_labels(date(2025, 12, 31), 12) == (2025, 4)
    assert fiscal_labels(date(2026, 1, 2), 12) == (2025, 4)    # 52/53-week year ending early January


def test_normalize_unit():
    assert normalize_unit("U_USD", "iso4217:USD", "us-gaap_Revenues", 69_632_000_000) == (69_632.0, "USD_millions")
    assert normalize_unit("U_UnitedStatesOfAmericaDollarsShare", "iso4217:USD",
                          "us-gaap_EarningsPerShareDiluted", 3.23) == (3.23, "USD_per_share")
    assert normalize_unit("U_shares", "shares", "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic", 7e9)[1] == "shares"


def _fact(start, end, value, period):
    return {"accession": "a", "ticker": "AMZN", "filing_type": "10-Q", "filing_date": end,
            "statement_type": "cashflow", "concept": "capex", "label": "Purchases of property and equipment",
            "standard_concept": None, "unit": "USD_millions", "value": value,
            "period_start": start, "period_end": end, "fiscal_period": period}


def test_derive_quarters_from_ytd_cash_flows():
    # Amazon 2025 capex as reported: Q1 discrete, then 6M / 9M YTD, then FY.
    rows = [
        _fact(date(2025, 1, 1), date(2025, 3, 31), 25019, "Q"),
        _fact(date(2025, 1, 1), date(2025, 6, 30), 57202, "6M"),
        _fact(date(2025, 1, 1), date(2025, 9, 30), 92297, "9M"),
        _fact(date(2025, 1, 1), date(2025, 12, 31), 131819, "FY"),
    ]
    derived = {(d["period_start"], d["period_end"]): d["value"] for d in derive_quarters(rows, 12)}
    assert derived == {
        (date(2025, 4, 1), date(2025, 6, 30)): 32183,
        (date(2025, 7, 1), date(2025, 9, 30)): 35095,
        (date(2025, 10, 1), date(2025, 12, 31)): 39522,
    }
    assert all(d["is_derived"] and d["fiscal_period"] == "Q" for d in derive_quarters(rows, 12))


def test_reported_quarter_is_not_overwritten():
    rows = [
        _fact(date(2025, 1, 1), date(2025, 3, 31), 100, "Q"),
        _fact(date(2025, 4, 1), date(2025, 6, 30), 120, "Q"),
        _fact(date(2025, 1, 1), date(2025, 6, 30), 220, "6M"),
    ]
    assert derive_quarters(rows, 12) == []


def test_toc_sections_and_paragraph_splitting():
    md = ("| PROSPECTUS SUMMARY........ | 1 |\n| RISK FACTORS........ | 27 |\n| BUSINESS........ | 130 |\n\n"
          "1 PROSPECTUS SUMMARY This summary highlights things. "
          "27 RISK FACTORS Investing is risky. "
          "130 BUSINESS We build rockets.")
    secs = toc_sections(md)
    assert [s["name"] for s in secs] == ["Prospectus Summary", "Risk Factors", "Business"]
    assert secs[1]["content"].strip().startswith("Investing is risky.")
    long_para = " ".join(["This is a sentence of moderate length."] * 60)
    assert all(len(p) <= 1200 for p in split_long_paragraphs(long_para).split("\n\n"))
