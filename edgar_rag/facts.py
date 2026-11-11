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

Beyond the three statements, a few note disclosures matter for AI-infrastructure
spending and are extracted with uniform labels (statement_type in parentheses):
- segment facts (segment): revenue, operating income, capex, D&A and assets per
  business segment, e.g. AWS capex. Read from dimensional facts on the business
  segments axis.
- lease facts (leases): right-of-use assets obtained for new leases (leased
  capacity that never shows up as cash capex), and future payments on leases
  signed but not yet commenced (the forward data-center pipeline). The latter
  is tagged differently by every company, hence the per-company rules.
- backlog (backlog), gross property and equipment by asset class (ppe) and non-lease
  purchase commitments (commitments): see the section further down.
"""

import re
from datetime import date, timedelta

import pandas as pd

STATEMENTS = (
    ("balance_sheet", "balance_sheet"),
    ("income_statement", "income_statement"),
    ("cashflow", "cashflow_statement"),
)
FLOW_STATEMENTS = ("income_statement", "cashflow")
# Statement types whose duration facts get missing quarters derived.
FLOW_FACT_STATEMENTS = ("income_statement", "cashflow", "segment", "leases")


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


def _period(f):
    """(start, end) of a raw fact; start is None for instants."""
    if f["period_type"] == "instant":
        instant = f.get("period_instant")
        return None, pd.to_datetime(f["period_end"] if pd.isna(instant) else instant).date()
    return pd.to_datetime(f["period_start"]).date(), pd.to_datetime(f["period_end"]).date()


class _Rows:
    """Accumulates fact rows for one filing, dropping exact repeats (facts recur across roles)."""

    def __init__(self, filing, ticker, fye_month, units):
        self.base = {"accession": filing.accession_number, "ticker": ticker,
                     "filing_type": filing.form, "filing_date": filing.filing_date}
        self.fye_month, self.units = fye_month, units
        self.rows, self.seen = [], set()

    def add(self, f, statement_type, concept, label, standard_concept=None, segment=None, value=None, unit=None):
        start, end = _period(f)
        if unit is None:
            measure = (self.units.get(f["unit_ref"]) or {}).get("measure")
            value, unit = normalize_unit(f["unit_ref"], measure, concept, float(f["numeric_value"]))
        key = (statement_type, concept, unit, segment, start, end)
        if key in self.seen:
            return False
        self.seen.add(key)
        fiscal_period = classify_period(start, end, self.fye_month)
        fiscal_year, fiscal_quarter = fiscal_labels(end, self.fye_month)
        self.rows.append({
            **self.base, "statement_type": statement_type, "concept": concept, "label": label,
            "standard_concept": standard_concept, "segment": segment, "value": value, "unit": unit,
            "period_type": "instant" if start is None else "duration",
            "period_start": start, "period_end": end,
            "duration_days": None if start is None else (end - start).days + 1,
            "fiscal_period": fiscal_period, "fiscal_year": fiscal_year,
            "fiscal_quarter": None if fiscal_period in ("FY", "TTM", "other") else fiscal_quarter,
            "is_derived": False,
        })
        return True


def _key(concept):
    # Raw facts use "us-gaap:Foo"; rendered statements (and the database) use "us-gaap_Foo".
    return concept.replace(":", "_", 1)


def extract_facts(filing, ticker, fye_month):
    """Primary-statement facts plus segment and lease note facts, with exact periods.

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

    all_df = xbrl.facts.to_dataframe()
    all_df = all_df[all_df["numeric_value"].notna()]
    plain = all_df[~all_df["is_dimensioned"].astype(bool)]
    by_concept = {_key(k): g for k, g in plain.groupby("concept")}
    rows = _Rows(filing, ticker, fye_month, xbrl.units or {})

    for (statement_type, concept), (label, standard_concept) in concepts.items():
        for _, f in by_concept.get(concept, pd.DataFrame()).iterrows():
            rows.add(f, statement_type, concept, label, standard_concept)

    statement_concepts = {concept for _, concept in concepts}
    _add_segment_facts(rows, all_df)
    _add_lease_flow_facts(rows, by_concept, statement_concepts)
    _add_lease_commitments(rows, all_df, ticker, balance_date(filing, all_df))
    _add_backlog_facts(rows, all_df, plain)
    _add_property_class_facts(rows, all_df, plain, statement_concepts)
    _add_commitment_facts(rows, all_df, plain, statement_concepts)
    return rows.rows, "ok"


# ── segment facts ────────────────────────────────────────────

SEGMENT_AXIS = "dim_us-gaap_StatementBusinessSegmentsAxis"
# The only other axis a segment fact may carry: segment totals are often tagged
# "Consolidation Items: Operating Segments" (as opposed to eliminations etc.).
_CONSOLIDATION_AXIS = "dim_srt_ConsolidationItemsAxis"
_OPERATING_SEGMENTS = "us-gaap:OperatingSegmentsMember"

# Synthetic concept -> (uniform label, candidate XBRL concepts in order of preference).
# One synthetic concept per line item keeps a series continuous when a company switches
# tags between years; within a filing the first candidate tagged for a segment/period wins.
SEGMENT_FACTS = {
    "edgar_rag_SegmentRevenue": (
        "Segment revenue",
        ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues")),
    "edgar_rag_SegmentOperatingIncome": ("Segment operating income", ("us-gaap:OperatingIncomeLoss",)),
    # Dominion measures segment profit as net income rather than operating income.
    "edgar_rag_SegmentNetIncome": ("Segment net income", ("us-gaap:NetIncomeLoss",)),
    "edgar_rag_SegmentCapitalExpenditures": (
        "Segment capital expenditures",
        ("us-gaap:SegmentExpenditureAdditionToLongLivedAssets", "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
         "us-gaap:PaymentsForProceedsFromProductiveAssets")),
    "edgar_rag_SegmentDepreciation": (
        "Segment depreciation and amortization",
        ("us-gaap:DepreciationDepletionAndAmortization", "us-gaap:DepreciationAndAmortization", "us-gaap:Depreciation")),
    "edgar_rag_SegmentAssets": ("Segment assets", ("us-gaap:Assets",)),
}
_SEGMENT_NAME = re.compile(r"(?:Statement, Business )?Segments: (.+?)(?=, [^,:]+: |$)")


# The same segment under different member labels across filings (keeps one series per segment).
# Renamed or reorganized segments (e.g. Microsoft's FY2027 segments) are real changes and stay distinct.
SEGMENT_ALIASES = {
    "Corporate Segment and Other Operating Segment": "Corporate and Other",
    "Dominion Energy South Carolina Inc": "Dominion Energy South Carolina",
    "DESC": "Dominion Energy South Carolina",
}


def segment_name(full_dimension_label):
    """'Consolidation Items: Operating Segments, Segments: AWS' -> 'AWS'."""
    m = _SEGMENT_NAME.search(full_dimension_label or "")
    if not m:
        return None
    name = m.group(1).strip()
    return SEGMENT_ALIASES.get(name, name)


def _segment_only(df):
    """Facts whose only dimension is a business segment (optionally marked as an operating segment)."""
    if SEGMENT_AXIS not in df.columns:
        return df.iloc[0:0]
    other_axes = [c for c in df.columns if c.startswith("dim_") and c not in (SEGMENT_AXIS, _CONSOLIDATION_AXIS)]
    mask = df[SEGMENT_AXIS].notna() & df[other_axes].isna().all(axis=1)
    if _CONSOLIDATION_AXIS in df.columns:
        mask &= df[_CONSOLIDATION_AXIS].isna() | (df[_CONSOLIDATION_AXIS] == _OPERATING_SEGMENTS)
    return df[mask]


def _add_segment_facts(rows, all_df):
    seg = _segment_only(all_df)
    if seg.empty:
        return
    for concept, (label, candidates) in SEGMENT_FACTS.items():
        for source in candidates:  # preference order: rows.add drops later duplicates
            for _, f in seg[seg["concept"] == source].iterrows():
                name = segment_name(f["full_dimension_label"])
                if name:
                    rows.add(f, "segment", concept, label, standard_concept=_key(source), segment=name)


# ── lease facts ──────────────────────────────────────────────

LEASE_FLOW_CONCEPTS = {
    "us-gaap_RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability":
        "Right-of-use assets obtained for new finance leases",
    "us-gaap_RightOfUseAssetObtainedInExchangeForOperatingLeaseLiability":
        "Right-of-use assets obtained for new operating leases",
}


def _add_lease_flow_facts(rows, by_concept, statement_concepts):
    for concept, label in LEASE_FLOW_CONCEPTS.items():
        if concept in statement_concepts:  # already captured from the cash flow statement
            continue
        for _, f in by_concept.get(concept, pd.DataFrame()).iterrows():
            rows.add(f, "leases", concept, label)


# Future payments on leases signed but not yet commenced. Every company tags this its own
# way, and most have changed tags at least once since 2020, so the total for the filing's
# balance date is taken from the first source that has it:
#   1. dedicated concepts (amzn:/goog:/fb:/meta:/orcl: ...LeaseNotYetCommenced...), summed
#      across concepts (Google splits current/non-current, Meta finance/operating);
#   2. the generic purchase-obligation concept, non-dimensioned, when its label says it is
#      about leases (Google and Meta 10-Qs);
#   3. the same concept broken out by a lease member (Amazon, Google, Meta 2024) or by lease
#      type (Microsoft: finance + operating until mid-2025, then one total tagged twice,
#      hence distinct values are summed).
# Stored under one synthetic concept so the clean view dedupes it like any other fact;
# standard_concept records the tag(s) it was read from.
LEASES_NOT_COMMENCED_CONCEPT = "edgar_rag_LeasesNotYetCommenced"
LEASES_NOT_COMMENCED_LABEL = "Leases not yet commenced (future lease payments)"
_NYC_CONCEPT = re.compile(r"LeaseNotYetCommenced(?:Amount|CurrentAmount|NoncurrentAmount|"
                          r"FutureMinimumPaymentsDue|LeaseCommitments)$")
_PURCHASE_OBLIGATION = "us-gaap:UnrecordedUnconditionalPurchaseObligationBalanceSheetAmount"
_LEASE_TEXT = re.compile(r"not yet commenced|lease commitments|undiscounted lease payments", re.I)
_LEASE_TYPE_DIM = re.compile(r"^Lease (?:Contractual Term|Arrangement, Type): (?:Finance|Operating) Lease$")
_NOISE_DIMS = re.compile(r"Subsequent Event|Statistical Measurement", re.I)
# Companies whose lease member has no lease wording in it.
_LEASE_MEMBER_OVERRIDES = {"CRWV": re.compile(r"Data Center and Office Buildings", re.I)}

# CoreWeave: MW of power still to be delivered at sites whose lease payments it treats as too
# uncertain to include in the total above (one large single site plus a group of sites).
UNDELIVERED_POWER_CONCEPT = "crwv:ElectricalPowerAccessNotYetCommenced"
UNDELIVERED_POWER_LABEL = "Undelivered contracted power at leased data center sites"


def _instants(df):
    return pd.to_datetime(df["period_instant"], errors="coerce")


def balance_date(filing, df):
    """The filing's balance-sheet date: period_of_report, else the most common instant."""
    try:
        return pd.to_datetime(filing.period_of_report).date()
    except (AttributeError, TypeError, ValueError):
        instants = _instants(df).dropna()
        return instants.mode().iloc[0].date() if not instants.empty else None


def leases_not_commenced(df, as_of, ticker=None):
    """(total USD, source concepts) for leases not yet commenced at `as_of`, or None."""
    dims = df["full_dimension_label"].fillna("")
    df = df[(_instants(df) == pd.Timestamp(as_of)) & df["numeric_value"].notna() & ~dims.str.contains(_NOISE_DIMS)]
    dims = df["full_dimension_label"].fillna("")
    plain = dims == ""

    dedicated = df[plain & df["concept"].str.contains(_NYC_CONCEPT)].drop_duplicates("concept")
    if not dedicated.empty:
        return float(dedicated["numeric_value"].sum()), sorted(dedicated["concept"])

    po = df["concept"] == _PURCHASE_OBLIGATION
    labelled = df[po & plain & df["label"].fillna("").str.contains(_LEASE_TEXT)]
    if not labelled.empty:
        return float(labelled["numeric_value"].max()), [_PURCHASE_OBLIGATION]

    member = _LEASE_MEMBER_OVERRIDES.get(ticker)
    if member is not None:
        by_member = dims.str.contains(member)
    else:
        by_member = dims.str.contains(_LEASE_TEXT) | dims.str.match(_LEASE_TYPE_DIM)
    broken_out = df[po & ~plain & by_member]
    if not broken_out.empty:
        return float(broken_out["numeric_value"].drop_duplicates().sum()), [_PURCHASE_OBLIGATION]
    return None


def _add_lease_commitments(rows, all_df, ticker, as_of):
    if as_of is None:
        return
    found = leases_not_commenced(all_df, as_of, ticker)
    at_date = all_df[_instants(all_df) == pd.Timestamp(as_of)]
    if found:
        value, sources = found
        rows.add(at_date.iloc[0], "leases", LEASES_NOT_COMMENCED_CONCEPT, LEASES_NOT_COMMENCED_LABEL,
                 standard_concept=", ".join(_key(s) for s in sources), value=value / 1_000_000, unit="USD_millions")
    power = at_date[(at_date["concept"] == UNDELIVERED_POWER_CONCEPT)
                    & ~at_date["full_dimension_label"].fillna("").str.contains(_NOISE_DIMS)]
    if not power.empty:
        rows.add(power.iloc[0], "leases", _key(UNDELIVERED_POWER_CONCEPT), UNDELIVERED_POWER_LABEL,
                 value=float(power["numeric_value"].drop_duplicates().sum()), unit="MW")


# ── backlog, property classes, purchase commitments ──────────
# Three more note disclosures that bear on AI-infrastructure economics. All are balance-date
# values; every date a filing reports (current and comparative) is stored.
#   backlog      remaining performance obligations: contracted revenue not yet recognized
#   ppe          gross property and equipment by asset class (servers, buildings, construction
#                in progress, ...): what the capex was spent on, and how much is not yet in service
#   commitments  purchase commitments other than leases (leases not yet commenced are above)

RPO_CONCEPT = "edgar_rag_RemainingPerformanceObligation"
RPO_LABEL = "Remaining performance obligations (contracted revenue not yet recognized)"
_RPO_SOURCE = "us-gaap:RevenueRemainingPerformanceObligation"
_RPO_TIMING_AXIS = "dim_us-gaap_RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionStartDateAxis"

_PPE_CLASS_AXIS = "dim_us-gaap_PropertyPlantAndEquipmentByTypeAxis"
_PPE_GROSS_SOURCES = (
    "us-gaap:PropertyPlantAndEquipmentGross",
    # Alphabet reports owned assets together with finance-lease assets.
    "us-gaap:PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
)
_CIP_SOURCE = "us-gaap:ConstructionInProgressGross"  # Oracle tags it as its own concept, without a class member
# Microsoft tags each class as its own concept instead of a member of the class axis.
_PPE_CLASS_CONCEPT = re.compile(
    r"^(?:us-gaap:(?:Land|LandAndLandImprovements|BuildingsAndImprovementsGross|LeaseholdImprovementsGross|"
    r"FurnitureAndFixturesGross|MachineryAndEquipmentGross|CapitalizedComputerSoftwareGross|"
    r"PropertyPlantAndEquipmentOther)|[a-z]+:Computer(?:Equipment|Hardware)\w*)$")
PPE_LABEL = "Property and equipment, gross: {}"

_PO_CATEGORY_AXIS = "dim_us-gaap_UnrecordedUnconditionalPurchaseObligationByCategoryOfItemPurchasedAxis"
# Company totals, each its own line item because the definitions differ between companies.
COMMITMENT_TOTALS = {
    _PURCHASE_OBLIGATION: ("edgar_rag_PurchaseObligations", "Unconditional purchase obligations (total, excluding leases)"),
    "us-gaap:PurchaseObligation": ("us-gaap_PurchaseObligation", "Purchase obligations (total)"),
    # Meta: non-cancelable commitments other than leases; Amazon: the total of its commitments table.
    "us-gaap:ContractualObligation": ("us-gaap_ContractualObligation", "Contractual commitments (total as the company defines it)"),
    "us-gaap:LongTermPurchaseCommitmentAmount": ("us-gaap_LongTermPurchaseCommitmentAmount", "Long-term purchase commitments"),
}
COMMITMENT_CATEGORY_LABEL = "Purchase obligations: {}"


def _camel(text_):
    return "".join(w[:1].upper() + w[1:] for w in re.findall(r"[A-Za-z0-9]+", text_))


def _with_only(df, *axes):
    """Facts dimensioned on exactly the given axes and no others."""
    if any(a not in df.columns for a in axes):
        return df.iloc[0:0]
    others = [c for c in df.columns if c.startswith("dim_") and c not in axes]
    mask = df[list(axes)].notna().all(axis=1)
    if others:
        mask &= df[others].isna().all(axis=1)
    return df[mask]


def _member(f):
    """'Long-Lived Asset, Class: Construction in progress' -> 'Construction in progress'."""
    return str(f["full_dimension_label"]).rsplit(": ", 1)[-1].strip()


def _instant_facts(df):
    return df[df["period_type"] == "instant"]


def _add_backlog_facts(rows, all_df, plain):
    source = _instant_facts(all_df[all_df["concept"] == _RPO_SOURCE])
    if source.empty:
        return
    std = _key(_RPO_SOURCE)
    for _, f in plain[plain["concept"] == _RPO_SOURCE].iterrows():
        rows.add(f, "backlog", RPO_CONCEPT, RPO_LABEL, standard_concept=std)
    # Amazon tags the total only on the expected-timing axis; tranches, where given, add up to it.
    timed = _with_only(source, _RPO_TIMING_AXIS)
    for _, group in timed.groupby("period_instant"):
        tranches = group.drop_duplicates(_RPO_TIMING_AXIS)
        rows.add(tranches.iloc[0], "backlog", RPO_CONCEPT, RPO_LABEL, standard_concept=std,
                 value=float(tranches["numeric_value"].sum()) / 1_000_000, unit="USD_millions")
    for _, f in _with_only(source, SEGMENT_AXIS).iterrows():
        name = segment_name(f["full_dimension_label"])
        if name:
            rows.add(f, "backlog", RPO_CONCEPT, RPO_LABEL, standard_concept=std, segment=name)


def _add_property_class_facts(rows, all_df, plain, statement_concepts):
    for source in _PPE_GROSS_SOURCES:
        by_class = _instant_facts(_with_only(all_df[all_df["concept"] == source], _PPE_CLASS_AXIS))
        for _, f in by_class.iterrows():
            name = _member(f)
            rows.add(f, "ppe", f"edgar_rag_PPEGross_{_camel(name)}", PPE_LABEL.format(name),
                     standard_concept=str(f[_PPE_CLASS_AXIS]).replace(":", "_", 1))
        if _key(source) not in statement_concepts:  # the total, unless the balance sheet already has it
            for _, f in _instant_facts(plain[plain["concept"] == source]).iterrows():
                rows.add(f, "ppe", "edgar_rag_PPEGross_Total", PPE_LABEL.format("Total"), standard_concept=_key(source))
    separate = plain[plain["concept"].str.match(_PPE_CLASS_CONCEPT)]
    for _, f in _instant_facts(separate).iterrows():
        if _key(f["concept"]) not in statement_concepts:
            name = str(f["label"]).strip()
            rows.add(f, "ppe", f"edgar_rag_PPEGross_{_camel(name)}", PPE_LABEL.format(name),
                     standard_concept=_key(f["concept"]))
    if _key(_CIP_SOURCE) not in statement_concepts:
        for _, f in _instant_facts(plain[plain["concept"] == _CIP_SOURCE]).iterrows():
            rows.add(f, "ppe", "edgar_rag_PPEGross_ConstructionInProgress",
                     PPE_LABEL.format("Construction in progress"), standard_concept=_key(_CIP_SOURCE))


def _add_commitment_facts(rows, all_df, plain, statement_concepts):
    for source, (concept, label) in COMMITMENT_TOTALS.items():
        if _key(source) in statement_concepts:
            continue
        facts_ = plain[plain["concept"] == source]
        if source == "us-gaap:LongTermPurchaseCommitmentAmount":
            # Alphabet tags its balance-date commitment in a year-to-date context.
            facts_ = facts_.assign(period_instant=facts_["period_instant"].fillna(facts_["period_end"]),
                                   period_type="instant")
        facts_ = _instant_facts(facts_)
        # Meta and Alphabet tag leases not yet commenced with the purchase-obligation concept.
        facts_ = facts_[~facts_["label"].fillna("").str.contains(_LEASE_TEXT)]
        for _, f in facts_.iterrows():
            rows.add(f, "commitments", concept, label, standard_concept=_key(source))
    by_category = _instant_facts(_with_only(all_df[all_df["concept"] == _PURCHASE_OBLIGATION], _PO_CATEGORY_AXIS))
    for _, f in by_category.iterrows():
        name = _member(f)
        if _LEASE_TEXT.search(name) or re.search(r"\blease", name, re.I):
            continue
        rows.add(f, "commitments", f"edgar_rag_PurchaseObligations_{_camel(name)}",
                 COMMITMENT_CATEGORY_LABEL.format(name), standard_concept=_key(_PURCHASE_OBLIGATION))


FACT_COLUMNS = ["accession", "ticker", "filing_type", "filing_date", "statement_type", "concept",
                "label", "standard_concept", "segment", "value", "unit", "period_type", "period_start",
                "period_end", "duration_days", "fiscal_period", "fiscal_year", "fiscal_quarter",
                "is_derived"]


# ── derived quarters ─────────────────────────────────────────

_YTD_STEPS = {"6M": "Q", "9M": "6M", "FY": "9M"}  # period -> the YTD period one quarter shorter


def _runs(versions):
    """Consecutive filings reporting the same value: [(first filing key, value, last row)], oldest first."""
    runs = []
    for v in versions:
        value = float(v["value"])
        if runs and runs[-1][1] == value:
            runs[-1] = (runs[-1][0], value, v)
        else:
            runs.append(((v["filing_date"], v["accession"]), value, v))
    return runs


def _same_basis_pair(longer, shorter):
    """(longer row, shorter value) reported on the same basis, preferring the most recent basis.

    Companies recast prior periods (segment reorganizations, discontinued operations,
    reclassifications), and a recast full-year figure is often never accompanied by a recast
    year-to-date one, so "latest minus latest" can subtract across two bases (Microsoft's
    FY2023 Q4 Intelligent Cloud revenue came out at 9.0B between quarters of 22.1B and 20.0B).

    A value of the shorter period holds from the first filing that reports it until a filing
    reports a different one. The matching longer-period value is the first one filed within
    that span: the 10-K that follows the Q3 10-Q, or the next 10-Q's year-to-date column. The
    newest shorter value with such a partner wins; if the newest has none (recast, but the
    longer period not yet re-reported), the previous basis is used.
    """
    runs = _runs(shorter)
    for i in range(len(runs) - 1, -1, -1):
        since, value, _ = runs[i]
        until = runs[i + 1][0] if i + 1 < len(runs) else None
        for row in longer:  # oldest first
            filed = (row["filing_date"], row["accession"])
            if filed >= since and (until is None or filed < until):
                return row, value
    return None


_STATEMENT_ORDER = {"income_statement": 0, "balance_sheet": 1}


def derive_quarters(rows, fye_month):
    """Compute missing discrete quarters as differences of year-to-date flows.

    rows: every reported duration fact of one company in every filing that reports it
    (monetary flows only; `concept` is the series concept). A 6M/9M/FY value minus the
    value for the same fiscal year one quarter shorter (same period_start) gives that
    quarter; the two are taken from the same reporting basis (see _same_basis_pair), so
    the derived row carries the filing of the longer-period value actually used, which
    is not always the latest one. Only emitted where no filing reports that quarter.
    """
    versions = {}
    reported_quarters = set()
    ordered = sorted(rows, key=lambda r: (r["filing_date"], r["accession"],
                                          _STATEMENT_ORDER.get(r["statement_type"], 2)))
    for r in ordered:
        series = (r["concept"], r["unit"], r.get("segment"))
        if r["fiscal_period"] == "Q":
            reported_quarters.add((*series, r["period_end"]))
        bucket = versions.setdefault((*series, r["period_start"], r["fiscal_period"]), [])
        if not bucket or bucket[-1]["accession"] != r["accession"]:  # one value per filing
            bucket.append(r)

    derived = []
    for (*series, period_start, fiscal_period), longer in versions.items():
        shorter = versions.get((*series, period_start, _YTD_STEPS.get(fiscal_period)))
        period_end = longer[-1]["period_end"]
        if not shorter or (*series, period_end) in reported_quarters:
            continue
        pair = _same_basis_pair(longer, shorter)
        if pair is None:
            continue
        r, prev_value = pair
        start = shorter[-1]["period_end"] + timedelta(days=1)
        days = (period_end - start).days + 1
        if classify_period(start, period_end, fye_month) != "Q":
            continue
        fiscal_year, fiscal_quarter = fiscal_labels(period_end, fye_month)
        derived.append({
            **{k: r[k] for k in ("accession", "ticker", "filing_type", "filing_date", "statement_type",
                                 "concept", "label", "standard_concept", "unit")},
            "segment": r.get("segment"),
            "value": float(r["value"]) - prev_value,
            "period_type": "duration", "period_start": start, "period_end": period_end,
            "duration_days": days, "fiscal_period": "Q",
            "fiscal_year": fiscal_year, "fiscal_quarter": fiscal_quarter, "is_derived": True,
        })
    return derived


# ── series: one concept per line item ────────────────────────

# Companies re-tag line items between filings (fb_ -> meta_, a custom tag -> the
# us-gaap one, a taxonomy update), and a newer filing's comparative columns use
# the new tag for periods already stored under the old one, so the clean view
# showed the same value twice. Separately, distinct items can share a label
# (Oracle "Services" revenue and expense, basic and diluted EPS), which reads as
# a duplicate once the concept name is dropped. build_series maps each reported
# (statement, concept, label) to a series concept and display label fixing both.

SERIES_STATEMENTS = ("balance_sheet", "income_statement", "cashflow")


def norm_label(label):
    """Label for matching: no parentheticals ("(loss)", "(Note 12)"), punctuation or case."""
    label = re.sub(r"\([^)]*\)", " ", label.replace("’", "'").lower())
    return " ".join(re.findall(r"[a-z0-9&']+", label))


_FILLER = {"and", "of", "the", "in", "for", "from", "to", "net", "other", "a", "on", "by", "total"}


def _similar(a, b):
    """Normalised labels that plausibly name the same item (wording drift between filings)."""
    ta, tb = set(a.split()) - _FILLER, set(b.split()) - _FILLER
    if not ta or not tb:
        return True
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.5


def _concept_words(concept):
    local = re.sub(r"\d+$", "", concept.split("_", 1)[-1])
    # Some company tags lower-case the joining words (goog_PropertyPlantandEquipment...).
    local = re.sub(r"(?<=[a-z])(and|of|for|in|to|after)(?=[A-Z])", lambda m: m.group(1).title(), local)
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", local)


def _qualifiers(concepts):
    """Short suffixes telling apart concepts that share a label: the words of each
    concept name that are not common to all of them ("basic" / "diluted")."""
    words = {c: _concept_words(c) for c in concepts}
    common = set.intersection(*(set(w) for w in words.values()))
    quals = {c: " ".join(w for w in words[c] if w not in common).lower() for c in concepts}
    if len(set(quals.values())) < len(quals) or not all(quals.values()):
        quals = {c: " ".join(words[c]).lower() for c in concepts}
    if len(set(quals.values())) < len(quals):
        quals = {c: c for c in concepts}
    return quals


def build_series(rows):
    """Map reported statement facts of one company to continuous series.

    rows: dicts with accession, filing_date, statement_type, concept, label, unit,
    period_start, period_end, value (reported, consolidated statement facts).
    Returns {(statement_type, concept, label): (series_concept, display_label)}.

    Two concepts are one re-tagged line item when they were used with the same
    label, never report the same period in the same filing (side-by-side items
    such as basic and diluted EPS always do), and report the same non-zero value
    for that label for at least one period (the comparative column of the first
    filing with the new tag). The merged item takes the most recently used
    concept. A concept whose labels are unrelated (a generic tag reused for
    different line items, e.g. Dominion's ProceedsFromSaleOfProductiveAssets)
    only moves the rows with the matching label; otherwise label wording drift
    stays within one series. Distinct items still sharing a label and unit in
    one statement get a qualifier from their concept names.
    """
    items, slots = {}, {}
    for r in rows:
        st = r["statement_type"]
        if st not in SERIES_STATEMENTS:
            continue
        k = (st, r["concept"], norm_label(r["label"]))
        i = items.setdefault(k, {"vals": set(), "units": set(), "labels": set(), "latest": (date.min, "")})
        i["labels"].add(r["label"])
        i["units"].add(r["unit"])
        if float(r["value"]) != 0:
            i["vals"].add((r["unit"], r["period_start"], r["period_end"], round(float(r["value"]), 4)))
        i["latest"] = max(i["latest"], (r["filing_date"], r["accession"]))
        slots.setdefault((st, r["concept"]), set()).add(
            (r["accession"], r["unit"], r["period_start"], r["period_end"]))

    # Start with one group per concept, or per label for a reused generic concept.
    by_concept = {}
    for k in items:
        by_concept.setdefault(k[:2], []).append(k)
    group = {}
    for ks in by_concept.values():
        labels = [k[2] for k in ks]
        reused = any(not _similar(a, b) for n, a in enumerate(labels) for b in labels[n + 1:])
        for members in ([frozenset([k]) for k in ks] if reused else [frozenset(ks)]):
            for k in members:
                group[k] = members

    # Merge, strongest evidence first, never joining two groups whose concepts
    # reported the same period in one filing (or two label groups of one concept).
    by_label = {}
    for k in items:
        by_label.setdefault((k[0], k[2]), []).append(k)
    scored = []
    for ks in by_label.values():
        ks.sort()
        for n, a in enumerate(ks):
            for b in ks[n + 1:]:
                shared = len(items[a]["vals"] & items[b]["vals"])
                if shared:
                    scored.append((shared, a, b))
    for _, a, b in sorted(scored, reverse=True):
        ga, gb = group[a], group[b]
        if ga == gb:
            continue
        concepts_a, concepts_b = {k[:2] for k in ga}, {k[:2] for k in gb}
        if set().union(*(slots[c] for c in concepts_a)) & set().union(*(slots[c] for c in concepts_b)):
            continue
        merged = ga | gb
        for k in merged:
            group[k] = merged

    series = {}
    for members in set(group.values()):
        head = max(members, key=lambda k: items[k]["latest"])[1]
        for k in members:
            series[k] = head

    # Distinct series still sharing a label and unit within one statement.
    clash = {}
    for (st, concept, label), head in series.items():
        for unit in items[(st, concept, label)]["units"]:
            clash.setdefault((st, label, unit), set()).add(head)
    qualifier = {}
    for (st, label, _), heads in clash.items():
        if len(heads) > 1:
            for c, q in _qualifiers(sorted(heads)).items():
                qualifier[(st, label, c)] = q

    out = {}
    for (st, concept, label), head in series.items():
        q = qualifier.get((st, label, head))
        for raw in items[(st, concept, label)]["labels"]:
            out[(st, concept, raw)] = (head, f"{raw} ({q})" if q else raw)
    return out
