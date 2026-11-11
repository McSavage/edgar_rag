"""Data-quality checks over the stored facts and chunks (no network).

    python -m edgar_rag check

Each check returns a DataFrame of offending rows; an empty frame is a pass.
"""

import pandas as pd
from sqlalchemy import text

from . import db

# Four discrete quarters must sum to the fiscal year. Where a company recast a year after the
# fact, the latest full-year figure and the quarters can legitimately be on different bases
# (the quarters are never all re-reported), so a mismatch is only a defect when no filing ever
# changed any figure of that year for that line item.
_QUARTER_SUMS = """
WITH q AS (
    SELECT ticker, concept, segment, fiscal_year, max(label) AS label, sum(value) AS quarters,
           count(*) AS n, count(*) FILTER (WHERE is_derived) AS n_derived
    FROM financial_facts_clean
    WHERE fiscal_period = 'Q' AND unit = 'USD_millions'
      AND statement_type IN ('income_statement', 'cashflow', 'segment')
    GROUP BY ticker, concept, segment, fiscal_year
), fy AS (
    SELECT ticker, concept, segment, fiscal_year, value AS full_year
    FROM financial_facts_clean
    WHERE fiscal_period = 'FY' AND unit = 'USD_millions'
      AND statement_type IN ('income_statement', 'cashflow', 'segment')
), restated AS (
    -- line item x fiscal year where some period was reported with more than one value
    SELECT DISTINCT ticker, concept, segment, fiscal_year FROM (
        SELECT f.ticker, COALESCE(s.series_concept, f.concept) AS concept, f.segment, f.fiscal_year
        FROM financial_facts f
        LEFT JOIN fact_series s
          ON s.ticker = f.ticker AND s.statement_type = f.statement_type AND s.concept = f.concept
         AND s.label = f.label
        WHERE NOT f.is_derived AND f.period_type = 'duration'
        GROUP BY 1, 2, 3, 4, f.period_start, f.period_end
        HAVING count(DISTINCT f.value) > 1
    ) x
)
SELECT q.ticker, q.fiscal_year, q.concept, q.segment, q.label, q.quarters, fy.full_year,
       q.quarters - fy.full_year AS difference, q.n_derived,
       (r.ticker IS NOT NULL) AS restated
FROM q
JOIN fy ON fy.ticker = q.ticker AND fy.concept = q.concept AND fy.fiscal_year = q.fiscal_year
       AND fy.segment IS NOT DISTINCT FROM q.segment
LEFT JOIN restated r ON r.ticker = q.ticker AND r.concept = q.concept AND r.fiscal_year = q.fiscal_year
       AND r.segment IS NOT DISTINCT FROM q.segment
WHERE q.n = 4 AND abs(q.quarters - fy.full_year) > greatest(1, abs(fy.full_year) * 0.005)
ORDER BY q.ticker, q.fiscal_year, q.concept
"""


def quarter_sums():
    """Line items whose four quarters do not sum to the fiscal year (0.5% / $1M tolerance)."""
    with db.get_engine().connect() as conn:
        return pd.read_sql(text(_QUARTER_SUMS), conn)


# A filing whose narrative is a fraction of what the same company's other filings of that form
# yield was almost certainly mis-parsed (the status columns only catch zero chunks).
THIN_SHARE = 0.4

_THIN_FILINGS = """
WITH typical AS (
    SELECT ticker, filing_type, percentile_cont(0.5) WITHIN GROUP (ORDER BY n_chunks) AS median, count(*) AS n
    FROM filings WHERE n_chunks IS NOT NULL GROUP BY ticker, filing_type
)
SELECT f.ticker, f.filing_type, f.filing_date, f.accession, f.n_chunks, t.median AS typical_chunks
FROM filings f JOIN typical t USING (ticker, filing_type)
WHERE t.n >= 4 AND f.n_chunks < :share * t.median
ORDER BY f.ticker, f.filing_date
"""


def thin_filings():
    """Filings with far fewer narrative chunks than is typical for that company and form."""
    with db.get_engine().connect() as conn:
        return pd.read_sql(text(_THIN_FILINGS), conn, params={"share": THIN_SHARE})


def capped_sections():
    """Sections cut off at the per-section chunk cap (text after the cap was dropped)."""
    from .narrative import MAX_CHUNKS_PER_SECTION, TOC_MAX_CHUNKS_PER_SECTION
    with db.get_engine().connect() as conn:
        return pd.read_sql(text("""
            SELECT ticker, filing_type, filing_date, accession, section, count(*) AS chunks
            FROM document_chunks GROUP BY 1, 2, 3, 4, 5
            HAVING count(*) IN (:cap, :toc_cap) ORDER BY 1, 3
        """), conn, params={"cap": MAX_CHUNKS_PER_SECTION, "toc_cap": TOC_MAX_CHUNKS_PER_SECTION})


def unembedded_chunks():
    with db.get_engine().connect() as conn:
        return conn.execute(text("SELECT count(*) FROM document_chunks WHERE embedding IS NULL")).scalar()


def run():
    """Print every check's result. Returns the number of unexplained problems."""
    sums = quarter_sums()
    unexplained = sums[~sums["restated"]]
    print(f"Quarter sums: {len(sums)} line-item years where four quarters differ from the fiscal year; "
          f"{len(sums) - len(unexplained)} follow a restatement, {len(unexplained)} unexplained")
    if not sums.empty:
        print(sums.groupby(["ticker", "restated"]).size().unstack(fill_value=0).to_string())
    if not unexplained.empty:
        print("\nUnexplained:")
        print(unexplained.drop(columns="restated").to_string(index=False))
    problems = len(unexplained)
    for title, frame in (("Thin filings (narrative far below the company's norm)", thin_filings()),
                         ("Sections cut off at the chunk cap", capped_sections())):
        print(f"\n{title}: {len(frame)}")
        if not frame.empty:
            print(frame.to_string(index=False))
        problems += len(frame)
    missing = unembedded_chunks()
    print(f"\nChunks without an embedding: {missing:,}")
    return problems + (1 if missing else 0)
