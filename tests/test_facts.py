"""Period classification, fiscal labels, derived quarters, series, segment and lease facts (no network or DB needed)."""

from datetime import date

import pandas as pd

from edgar_rag.facts import (build_series, classify_period, derive_quarters, fiscal_labels, leases_not_commenced,
                             norm_label, normalize_unit, segment_name)
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


def test_segment_quarters_are_derived_per_segment():
    # AWS and North America capex must not be subtracted from each other.
    def seg(segment, start, end, value, period):
        return {**_fact(start, end, value, period), "concept": "edgar_rag_SegmentCapitalExpenditures",
                "statement_type": "segment", "segment": segment}
    rows = [
        seg("AWS", date(2026, 1, 1), date(2026, 6, 30), 90120, "6M"),
        seg("AWS", date(2026, 1, 1), date(2026, 9, 30), 147236, "9M"),
        seg("North America", date(2026, 1, 1), date(2026, 6, 30), 23265, "6M"),
        seg("North America", date(2026, 1, 1), date(2026, 9, 30), 37795, "9M"),
    ]
    derived = {d["segment"]: d["value"] for d in derive_quarters(rows, 12)}
    assert derived == {"AWS": 57116, "North America": 14530}


def test_segment_name():
    assert segment_name("Consolidation Items: Operating Segments, Segments: AWS") == "AWS"
    assert segment_name("Statement, Business Segments: Intelligent Cloud") == "Intelligent Cloud"
    # Aliases keep one series when a company relabels the same segment.
    assert segment_name("Segments: Corporate Segment and Other Operating Segment") == "Corporate and Other"
    assert segment_name("Segments: DESC") == "Dominion Energy South Carolina"
    assert segment_name(None) is None


def _xbrl(*facts):
    """Minimal edgartools-style facts frame: (concept, label, dimension label, value[, instant])."""
    rows = [{"concept": c, "label": label, "full_dimension_label": dims, "numeric_value": v,
             "period_instant": rest[0] if rest else "2026-06-30"} for c, label, dims, v, *rest in facts]
    return pd.DataFrame(rows)


PO = "us-gaap:UnrecordedUnconditionalPurchaseObligationBalanceSheetAmount"
AS_OF = date(2026, 6, 30)


def test_leases_not_commenced_dedicated_concepts_are_summed():
    # Google 10-K: commencing next year + later years.
    df = _xbrl(("goog:LesseeLeaseNotYetCommencedCurrentAmount", "", None, 5.8e9),
               ("goog:LesseeLeaseNotYetCommencedNoncurrentAmount", "", None, 52.7e9),
               ("goog:FinanceLeaseLeaseNotYetCommencedPrepayment", "", None, 1.1e9))
    assert leases_not_commenced(df, AS_OF)[0] == 58.5e9


def test_leases_not_commenced_microsoft_lease_type_breakdown():
    split = _xbrl((PO, "Finance Lease", "Lease Contractual Term: Finance Lease", 108.4e9),
                  (PO, "Operating Lease", "Lease Contractual Term: Operating Lease", 8.6e9))
    assert leases_not_commenced(split, AS_OF)[0] == 117.0e9
    combined_tagged_twice = _xbrl((PO, "Finance Lease", "Lease Contractual Term: Finance Lease", 329.1e9),
                                  (PO, "Operating Lease", "Lease Contractual Term: Operating Lease", 329.1e9))
    assert leases_not_commenced(combined_tagged_twice, AS_OF)[0] == 329.1e9


def test_leases_not_commenced_ignores_purchase_obligations_and_noise():
    df = _xbrl((PO, "Total", None, 13.3e9),                                       # real purchase obligations
               (PO, "Short-Term Lease Agreement", "Category: Short-Term Lease Agreement", 2e9),
               (PO, "Subsequent Event", "Subsequent Event Type: Subsequent Event", 9e9),
               (PO, "Operating Lease, Lease Not yet Commenced", "Category: Operating Lease, Lease Not yet Commenced", 85.2e9),
               (PO, "Lease commitments", None, 1e9, "2025-10-31"))                # other date
    assert leases_not_commenced(df, AS_OF)[0] == 85.2e9
    assert leases_not_commenced(_xbrl((PO, "Total", None, 13.3e9)), AS_OF) is None


def test_leases_not_commenced_labelled_total_beats_breakdown():
    df = _xbrl((PO, "Total undiscounted lease payments", None, 75.6e9),
               (PO, "Operating Lease, Lease Not yet Commenced", "Category: Operating Lease, Lease Not yet Commenced", 70e9))
    assert leases_not_commenced(df, AS_OF)[0] == 75.6e9


def _stmt(acc, filed, concept, label, end, value, st="cashflow", start=None, unit="USD_millions"):
    return {"accession": acc, "filing_date": filed, "statement_type": st, "concept": concept, "label": label,
            "unit": unit, "period_start": start, "period_end": end, "value": value}


def test_series_merges_retagged_concept():
    # META: fb_ tag until 2021, meta_ tag from 2022; the 2022 10-K re-reports 2021 under the new tag.
    label = "Repurchases of Class A common stock in accrued expenses and other current liabilities"
    old, new = "fb_RepurchaseOfCommonStock", "meta_RepurchaseOfCommonStock"
    rows = [_stmt("A", date(2021, 2, 1), old, label, date(2020, 12, 31), 40, start=date(2020, 1, 1)),
            _stmt("B", date(2022, 2, 1), old, label, date(2021, 12, 31), 340, start=date(2021, 1, 1)),
            _stmt("C", date(2023, 2, 1), new, label, date(2021, 12, 31), 340, start=date(2021, 1, 1)),
            _stmt("C", date(2023, 2, 1), new, label, date(2022, 12, 31), 30, start=date(2022, 1, 1))]
    s = build_series(rows)
    assert s[("cashflow", old, label)] == (new, label)
    assert s[("cashflow", new, label)] == (new, label)


def test_series_keeps_side_by_side_items_apart_and_qualifies_labels():
    # Dominion: basic and diluted EPS share a label and usually a value, but both appear in every filing.
    label = "Net income attributable to Dominion Energy"
    rows = []
    for acc, filed, end in (("A", date(2024, 2, 1), date(2023, 12, 31)), ("B", date(2025, 2, 1), date(2024, 12, 31))):
        for c in ("us-gaap_EarningsPerShareBasic", "us-gaap_EarningsPerShareDiluted"):
            rows.append(_stmt(acc, filed, c, label, end, 2.44, st="income_statement",
                              start=date(end.year, 1, 1), unit="USD_per_share"))
        rows.append(_stmt(acc, filed, "us-gaap_NetIncomeLoss", label, end, 2000, st="income_statement",
                          start=date(end.year, 1, 1)))
    s = build_series(rows)
    assert s[("income_statement", "us-gaap_EarningsPerShareBasic", label)] == ("us-gaap_EarningsPerShareBasic", f"{label} (basic)")
    assert s[("income_statement", "us-gaap_EarningsPerShareDiluted", label)][1] == f"{label} (diluted)"
    assert s[("income_statement", "us-gaap_NetIncomeLoss", label)] == ("us-gaap_NetIncomeLoss", label)  # other unit


def test_series_needs_matching_values_and_label():
    # Same label, never side by side, but no common value: not enough evidence (Dominion "Other":
    # long-term investments one year, other long-term debt another).
    rows = [_stmt("A", date(2021, 2, 1), "us-gaap_OtherLongTermInvestments", "Other", date(2020, 12, 31), 5, st="balance_sheet"),
            _stmt("B", date(2022, 2, 1), "us-gaap_OtherLongTermDebtNoncurrent", "Other", date(2021, 12, 31), 7, st="balance_sheet")]
    s = build_series(rows)
    assert s[("balance_sheet", "us-gaap_OtherLongTermInvestments", "Other")] == (
        "us-gaap_OtherLongTermInvestments", "Other (investments)")
    # A generic tag reused for another line item only moves the rows with the matching label.
    generic, specific = "us-gaap_ProceedsFromSaleOfProductiveAssets", "d_ProceedsFromEastOhioTransaction"
    rows = [_stmt("A", date(2023, 2, 1), generic, "Proceeds from sales of assets", date(2022, 12, 31), 90, start=date(2022, 1, 1)),
            _stmt("B", date(2024, 2, 1), generic, "Proceeds from East Ohio Transaction", date(2023, 12, 31), 6600, start=date(2023, 1, 1)),
            _stmt("C", date(2025, 2, 1), specific, "Proceeds from East Ohio Transaction", date(2023, 12, 31), 6600, start=date(2023, 1, 1))]
    s = build_series(rows)
    assert s[("cashflow", generic, "Proceeds from East Ohio Transaction")][0] == specific
    assert s[("cashflow", generic, "Proceeds from sales of assets")][0] == generic


def test_norm_label():
    assert norm_label("Net income (loss) attributable to Dominion Energy") == "net income attributable to dominion energy"
    assert norm_label("Total stockholders\u2019 equity") == norm_label("Total stockholders' equity")
