"""XBRL financial fact extraction.

Previous versions read values off the rendered statement tables, which expose a
single date per column. In 10-Qs edgartools sometimes renders the 3-month column
and sometimes the year-to-date one, and nothing recorded which, so quarterly
series silently mixed quarters with 6- and 9-month cumulative totals (e.g.
MSFT "Q2 FY25 revenue" = 135,217 instead of 69,632).

Now: the rendered statements are used only to decide *which concepts* belong to
the balance sheet / income statement / cash flow statement (and their company
labels); values come from the raw XBRL facts, which carry exact period start and
end dates and units. Missing discrete quarters (cash-flow statements are YTD-only
in 10-Qs, and no 10-Q exists for Q4) are then derived as YTD differences.
"""

from datetime import date, timedelta

import pandas as pd

STATEMENTS = (
    ("balance_sheet", "balance_sheet"),
    ("income_statement", "income_statement"),
    ("cashflow", "cashflow_statement"),
)
FLOW_STATEMENTS = ("income_statement", "cashflow")


# ── period classification ────────────────────────────────────

def _effective_month(d):
    # 52/53-week fiscal calendars can end a few days into the next month (e.g. Jan 2).
    shifted = d - timedelta(days=7) if d.day <= 7 else d
    return shifted.year, shifted.month


def _starts_fiscal_year(start, fye_month):
    # Fiscal years start the day after FYE; allow 52/53-week starts in the last days of the prior month.
    shifted = start + timedelta(days=7) if start.day >= 24 else start
    return shifted.month == fye_month % 12 + 1


def classify_period(start, end, fye_month):
    """instant | Q | 6M | 9M | FY | TTM | other.

    6M/9M/FY must start on the fiscal-year boundary, so trailing-twelve-month
    figures (Amazon tags TTM cash flows) aren't mistaken for fiscal years.
    """
    if start is None:
        return "instant"
    days = (end - start).days + 1
    if 80 <= days <= 100:
        return "Q"
    from_fy_start = _starts_fiscal_year(start, fye_month)
    if 350 <= days <= 380:
        return "FY" if from_fy_start else "TTM"
    if from_fy_start and 170 <= days <= 195:
        return "6M"
    if from_fy_start and 260 <= days <= 290:
        return "9M"
    return "other"


def fiscal_labels(period_end, fye_month):
    """(fiscal_year, fiscal_quarter) of the quarter ending on period_end."""
    year, month = _effective_month(period_end)
    fiscal_year = year + 1 if month > fye_month else year
    fiscal_quarter = ((month - fye_month - 1) % 12) // 3 + 1
    return fiscal_year, fiscal_quarter


# ── units ────────────────────────────────────────────────────

def normalize_unit(unit_ref, measure, concept, value):
    """Return (value, unit) with monetary values scaled to USD millions."""
    ref = (unit_ref or "").lower()
    meas = (measure or "").lower()
    concept_l = concept.lower()
    if "pershare" in concept_l or ("share" in ref and ("usd" in ref or "dollar" in ref)):
        return value, "USD_per_share"
    if meas == "iso4217:usd" or ref == "u_usd" or ref == "usd":
        return value / 1_000_000, "USD_millions"
    if meas == "shares" or ref in ("u_shares", "shares"):
        return value, "shares"
    if meas == "pure" or ref in ("u_pure", "pure"):
        return value, "pure"
    return value, measure or unit_ref or "unknown"


# ── extraction ───────────────────────────────────────────────

def _statement_concepts(statements):
    """{concept: (statement_type, label, standard_concept)} for the three primary statements."""
    concepts = {}
    for statement_type, method in STATEMENTS:
        try:
            df = getattr(statements, method)().to_dataframe()
        except Exception as e:
            print(f"      ✗ {statement_type}: {e}")
            continue
        rows = df[~df["abstract"].astype(bool) & ~df["dimension"].astype(bool)]
        for _, row in rows.iterrows():
            std = row.get("standard_concept")
            concepts.setdefault((statement_type, row["concept"]),
                                (row["label"], None if pd.isna(std) else std))
    return concepts


def extract_facts(filing, ticker, fye_month):
    """All non-dimensional facts for primary-statement concepts, with exact periods.

    Returns (facts, status) where status is 'ok', 'none' (filing has no XBRL) or 'failed'.
    """
    try:
        xbrl = filing.xbrl()
    except Exception as e:
        print(f"      ✗ XBRL unavailable: {e}")
        return [], "failed"
    if xbrl is None:
        return [], "none"

    concepts = _statement_concepts(xbrl.statements)
    if not concepts:
        return [], "failed"

    facts_df = xbrl.facts.to_dataframe()
    facts_df = facts_df[~facts_df["is_dimensioned"].astype(bool) & facts_df["numeric_value"].notna()]
    # Raw facts use "us-gaap:Foo"; rendered statements use "us-gaap_Foo".
    facts_df = facts_df.assign(concept_key=facts_df["concept"].str.replace(":", "_", n=1))
    by_concept = {k: g for k, g in facts_df.groupby("concept_key")}
    units = xbrl.units or {}

    filing_type = filing.form
    filing_date = filing.filing_date
    accession = filing.accession_number
    out, seen = [], set()

    for (statement_type, concept), (label, standard_concept) in concepts.items():
        group = by_concept.get(concept)
        if group is None:
            continue
        for _, f in group.iterrows():
            if f["period_type"] == "instant":
                instant = f.get("period_instant")
                end = pd.to_datetime(f["period_end"] if pd.isna(instant) else instant).date()
                start, days = None, None
            else:
                start = pd.to_datetime(f["period_start"]).date()
                end = pd.to_datetime(f["period_end"]).date()
                days = (end - start).days + 1
            measure = (units.get(f["unit_ref"]) or {}).get("measure")
            value, unit = normalize_unit(f["unit_ref"], measure, concept, float(f["numeric_value"]))

            key = (statement_type, concept, unit, start, end)
            if key in seen:  # the same fact is repeated across presentation roles
                continue
            seen.add(key)

            fiscal_period = classify_period(start, end, fye_month)
            fiscal_year, fiscal_quarter = fiscal_labels(end, fye_month)
            out.append({
                "accession": accession, "ticker": ticker,
                "filing_type": filing_type, "filing_date": filing_date,
                "statement_type": statement_type, "concept": concept, "label": label,
                "standard_concept": standard_concept, "value": value, "unit": unit,
                "period_type": "instant" if start is None else "duration",
                "period_start": start, "period_end": end, "duration_days": days,
                "fiscal_period": fiscal_period, "fiscal_year": fiscal_year,
                "fiscal_quarter": None if fiscal_period in ("FY", "TTM", "other") else fiscal_quarter,
                "is_derived": False,
            })
    return out, "ok"


FACT_COLUMNS = ["accession", "ticker", "filing_type", "filing_date", "statement_type", "concept",
                "label", "standard_concept", "value", "unit", "period_type", "period_start",
                "period_end", "duration_days", "fiscal_period", "fiscal_year", "fiscal_quarter",
                "is_derived"]


# ── derived quarters ─────────────────────────────────────────

_YTD_STEPS = {"6M": "Q", "9M": "6M", "FY": "9M"}  # period -> the YTD period one quarter shorter


def derive_quarters(clean_rows, fye_month):
    """Compute missing discrete quarters as differences of year-to-date flows.

    clean_rows: deduplicated duration facts (latest value per concept/period) for one
    company, monetary flows only. A 6M/9M/FY value minus the reported value for the
    same fiscal year one quarter shorter (same period_start) gives that quarter.
    Only emitted where no reported 3-month value exists for that quarter.
    """
    by_key = {}
    reported_quarters = set()
    for r in clean_rows:
        by_key[(r["concept"], r["unit"], r["period_start"], r["fiscal_period"])] = r
        if r["fiscal_period"] == "Q":
            reported_quarters.add((r["concept"], r["unit"], r["period_end"]))

    derived = []
    for r in clean_rows:
        shorter = _YTD_STEPS.get(r["fiscal_period"])
        if not shorter:
            continue
        prev = by_key.get((r["concept"], r["unit"], r["period_start"], shorter))
        if prev is None or (r["concept"], r["unit"], r["period_end"]) in reported_quarters:
            continue
        start = prev["period_end"] + timedelta(days=1)
        days = (r["period_end"] - start).days + 1
        if classify_period(start, r["period_end"], fye_month) != "Q":
            continue
        fiscal_year, fiscal_quarter = fiscal_labels(r["period_end"], fye_month)
        derived.append({
            **{k: r[k] for k in ("accession", "ticker", "filing_type", "filing_date", "statement_type",
                                 "concept", "label", "standard_concept", "unit")},
            "value": float(r["value"]) - float(prev["value"]),
            "period_type": "duration", "period_start": start, "period_end": r["period_end"],
            "duration_days": days, "fiscal_period": "Q",
            "fiscal_year": fiscal_year, "fiscal_quarter": fiscal_quarter, "is_derived": True,
        })
    return derived
