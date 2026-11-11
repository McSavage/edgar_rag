"""Canonical metrics: one name per line item across companies.

Companies tag the same line item with different XBRL concepts and label it differently
still (capex is "Purchases of property and equipment" at Amazon, "Additions to property
and equipment" at Microsoft, "Plant construction and other property additions" at
Dominion). Matching on label wording is guesswork, so each metric lists the concepts
that can carry it, in order of preference; `build_metric_map` resolves them against the
concepts a company actually reports, and the `financial_metrics` view exposes the result:

    SELECT ticker, period_end, value FROM financial_metrics
    WHERE metric = 'capex' AND fiscal_period = 'Q'

Patterns are regular expressions matched against the series concept (see fact_series),
so a line item a company re-tagged stays one metric. Company totals only; segment
figures are reached through the `segment` column of financial_facts_clean.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    patterns: tuple   # regexes on the concept, most preferred first
    kind: str         # flow (a period) | stock (a balance date)


METRICS = [
    # income statement
    Metric("revenue", "total revenue / net sales",
           (r"^us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax$", r"^us-gaap_Revenues$"), "flow"),
    Metric("cost_of_revenue", "cost of revenue / cost of sales",
           (r"^us-gaap_CostOfRevenue$", r"^us-gaap_CostOfGoodsAndServicesSold$"), "flow"),
    Metric("operating_income", "operating income (loss)", (r"^us-gaap_OperatingIncomeLoss$",), "flow"),
    Metric("net_income", "net income (loss)", (r"^us-gaap_NetIncomeLoss$", r"^us-gaap_ProfitLoss$"), "flow"),
    Metric("research_and_development", "research and development expense (Amazon: technology and infrastructure)",
           (r"^us-gaap_ResearchAndDevelopmentExpense", r"^amzn_TechnologyAnd(Infrastructure|Content)Expense$"), "flow"),
    Metric("interest_expense", "interest expense",
           (r"^us-gaap_InterestExpense(Nonoperating)?$", r"^us-gaap_InterestAndDebtExpense$"), "flow"),
    # cash flow
    Metric("capex", "cash capital expenditures: purchases of / additions to property and equipment",
           (r"PaymentsToAcquire(PropertyPlantAndEquipment|ProductiveAssets)",), "flow"),
    Metric("operating_cash_flow", "net cash provided by operating activities",
           (r"^us-gaap_NetCashProvidedByUsedInOperatingActivities$",), "flow"),
    Metric("depreciation_amortization",
           "depreciation and amortization (Alphabet and Oracle: depreciation of property and equipment only)",
           (r"^[a-z-]+_DepreciationDepletionAndAmortization", r"^[a-z-]+_DepreciationAmortizationAndOther$",
            r"^us-gaap_DepreciationAndAmortization$", r"^us-gaap_Depreciation$",
            r"^[a-z-]+_DepreciationAndImpairment"), "flow"),
    Metric("stock_based_compensation", "stock-based compensation expense",
           (r"^us-gaap_ShareBasedCompensation$", r"^us-gaap_AllocatedShareBasedCompensationExpense$"), "flow"),
    Metric("finance_lease_additions", "right-of-use assets obtained for new finance leases",
           (r"^us-gaap_RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability$",), "flow"),
    Metric("operating_lease_additions", "right-of-use assets obtained for new operating leases",
           (r"^us-gaap_RightOfUseAssetObtainedInExchangeForOperatingLeaseLiability$",), "flow"),
    # balance sheet
    Metric("total_assets", "total assets", (r"^us-gaap_Assets$",), "stock"),
    Metric("total_liabilities", "total liabilities", (r"^us-gaap_Liabilities$",), "stock"),
    Metric("stockholders_equity", "total stockholders' equity",
           (r"^us-gaap_StockholdersEquity$",
            r"^us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest$"), "stock"),
    Metric("cash", "cash and cash equivalents", (r"^us-gaap_CashAndCashEquivalentsAtCarryingValue$",), "stock"),
    Metric("long_term_debt", "long-term debt, non-current portion",
           (r"^us-gaap_LongTermDebtNoncurrent$", r"^us-gaap_LongTermDebtAndCapitalLeaseObligations$",
            r"^us-gaap_LongTermNotesAndLoans$", r"^d_LongTermDebt"), "stock"),
    Metric("property_and_equipment_net", "property and equipment, net of accumulated depreciation",
           (r"^us-gaap_PropertyPlantAndEquipmentNet$",
            r"^us-gaap_PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization$"),
           "stock"),
    # note disclosures (uniform edgar_rag_* concepts, see facts.py)
    Metric("property_and_equipment_gross", "property and equipment at cost, before depreciation",
           (r"^us-gaap_PropertyPlantAndEquipmentGross$", r"^edgar_rag_PPEGross_Total$",
            r"^us-gaap_PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization$"),
           "stock"),
    Metric("construction_in_progress", "property not yet in service: construction in progress",
           (r"^edgar_rag_PPEGross_(ConstructionInProgress|AssetsNotYetInService)$",), "stock"),
    Metric("servers_and_network_equipment",
           "gross servers and network equipment (Alphabet: technical infrastructure; CoreWeave: technology equipment; "
           "Microsoft: servers, network equipment and software; Oracle: computer, network, machinery and equipment)",
           (r"^edgar_rag_PPEGross_(Servers\w*|TechnicalInfrastructure|TechnologyEquipment|Computer\w*)$",), "stock"),
    Metric("leases_not_yet_commenced",
           "future payments on leases signed but not yet started (mostly data centers): the forward capacity pipeline",
           (r"^edgar_rag_LeasesNotYetCommenced$",), "stock"),
    Metric("remaining_performance_obligations", "backlog: contracted revenue not yet recognized",
           (r"^edgar_rag_RemainingPerformanceObligation$",), "stock"),
]
METRICS_BY_NAME = {m.name: m for m in METRICS}
METRIC_NAMES = [m.name for m in METRICS]


def match(concept):
    """[(metric name, rank)] for every metric the concept can carry (lower rank = preferred)."""
    out = []
    for m in METRICS:
        for rank, pattern in enumerate(m.patterns):
            if re.search(pattern, concept):
                out.append((m.name, rank))
                break
    return out


def build_metric_map(concepts):
    """[(metric, concept, rank)] for the concepts one company reports."""
    return [(name, c, rank) for c in sorted(set(concepts)) for name, rank in match(c)]
