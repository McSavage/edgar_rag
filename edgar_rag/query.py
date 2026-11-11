"""Natural-language query interface: route with Claude, retrieve from SQL + pgvector, synthesize.

Moved out of edgar_rag_interface.ipynb so the notebook is a thin client:

    from edgar_rag.query import ask
    ask("How much did Amazon spend on capex each quarter of 2025?")
"""

import json
import math
import os
from datetime import date
from functools import lru_cache

import anthropic
import pandas as pd
from sqlalchemy import text

from . import config, db
from .embeddings import embed_query, to_pgvector


# ── Claude ───────────────────────────────────────────────────

class ClaudeRefusal(RuntimeError):
    pass


@lru_cache(maxsize=1)
def claude_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_claude(prompt, max_tokens, effort=None, output_format=None):
    """One Messages API call, with server-side refusal fallback enabled."""
    output_config = {}
    if effort:
        output_config["effort"] = effort
    if output_format:
        output_config["format"] = output_format
    response = claude_client().beta.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},  # anthropic 0.78 predates the typed `fallbacks` param
        **({"output_config": output_config} if output_config else {}),
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise ClaudeRefusal(f"Claude declined the request ({getattr(details, 'category', 'unknown')})")
    return next((b.text for b in response.content if b.type == "text"), "")


# ── company metadata ─────────────────────────────────────────

@lru_cache(maxsize=1)
def fiscal_year_ends():
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("SELECT ticker, fiscal_year_end_month FROM companies")).fetchall()
    return {t: m for t, m in rows if m}


@lru_cache(maxsize=1)
def segment_names():
    """{ticker: [segment, ...]} reported since 2024, latest first (older names were reorganized away)."""
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, segment FROM financial_facts_clean
            WHERE segment IS NOT NULL AND period_end >= '2024-01-01'
            GROUP BY ticker, segment ORDER BY ticker, max(period_end) DESC, segment
        """)).fetchall()
    out = {}
    for ticker, segment in rows:
        out.setdefault(ticker, []).append(segment)
    return out


def _company_lines():
    months = fiscal_year_ends()
    segments = segment_names()
    lines = []
    for c in config.COMPANIES:
        names = ", ".join((c.name, *c.aliases))
        fye = months.get(c.ticker)
        fy_note = "" if fye in (None, 12) else f"; fiscal year ends {date(2000, fye, 1):%B}"
        seg_note = f"; segments: {', '.join(segments[c.ticker])}" if segments.get(c.ticker) else ""
        lines.append(f"- {c.ticker}: {names} ({c.sector}{fy_note}{seg_note})")
    return "\n".join(lines)


# ── routing ──────────────────────────────────────────────────

ROUTER_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "enum": ["quantitative", "qualitative", "both"]},
            "tickers": {"type": "array", "items": {"type": "string", "enum": config.TICKERS}},
            "sql_keywords": {"type": "array", "items": {"type": "string"}},
            "period_granularity": {"type": "string", "enum": ["quarterly", "annual", "any"]},
            "include_segments": {"type": "boolean"},
            "segment_filter": {"type": "array", "items": {"type": "string"}},
            "date_start": {"type": "string"},
            "date_end": {"type": "string"},
            "narrative_search_query": {"type": "string"},
        },
        "required": ["query_type", "tickers", "sql_keywords", "period_granularity", "include_segments",
                     "segment_filter", "date_start", "date_end", "narrative_search_query"],
        "additionalProperties": False,
    },
}

ROUTER_PROMPT = """Analyze this question about SEC filings and extract search parameters.

Question: {question}

Companies in the database:
{companies}

Financial facts come from the balance sheet, income statement and cash flow statement.
Each fact is tagged with its exact period: a discrete fiscal quarter, a full fiscal year,
or a balance-sheet date. Example line-item labels (use them to write precise sql_keywords):
- "Revenue", "Total revenues", "Net sales", "Cost of revenue", "Operating income", "Net income"
- "Technology and infrastructure", "Research and development", "General and administrative"
- Capex: "Purchase of property and equipment" (matches Amazon, Alphabet, Meta, CoreWeave, SpaceX),
  "Additions to property and equipment" (Microsoft), "Capital expenditures" (Oracle)
- "Plant construction and other property additions" (Dominion's capex line)
- "Property and equipment, net", "Total assets", "Cash and cash equivalents"
- "Long-term debt", "Total liabilities", "Total stockholders equity"
- "Depreciation and amortization", "Stock-based compensation"

Beyond the three statements, two note disclosures are stored with uniform labels:
- Business segments (only searched when include_segments is true): "Segment revenue", "Segment operating income",
  "Segment net income", "Segment capital expenditures", "Segment depreciation and amortization", "Segment assets".
  Segment capex exists for Amazon (AWS, North America, International; quarterly) and Dominion (Dominion Energy
  Virginia, Dominion Energy South Carolina, Contracted Energy; annual). Microsoft (e.g. Intelligent Cloud),
  Alphabet (Google Cloud, Google Services), Meta, Oracle and SpaceX have segment revenue / operating income only.
- Leases: "Leases not yet commenced" (future payments on signed, mostly data-center, leases that have not started:
  the forward build-out pipeline, a balance-date value), "Right-of-use assets obtained for new finance leases" and
  "... new operating leases" (leased capacity added in the period, which cash capex does not include), and for
  CoreWeave "Undelivered contracted power" (MW). For questions about total infrastructure spending, commitments
  or the data-center pipeline, add lease keywords next to the capex label.

Fields:
- query_type: "quantitative" for numbers/metrics, "qualitative" for risks/strategy/narrative, "both" for combined.
- tickers: every company the question mentions, mapped to its ticker. Empty list if none (searches all).
- sql_keywords: 2-5 phrases that would match line-item labels via SQL ILIKE. No apostrophes.
- period_granularity: "quarterly" if the question is about quarters or quarterly trends, "annual" if about
  years or fiscal years, otherwise "any".
- include_segments: true if the question is about a business segment (AWS, Google Cloud, Intelligent Cloud,
  Dominion Energy Virginia, ...) or asks for a segment breakdown; otherwise false (company totals only).
- segment_filter: when the question singles out particular segments, their exact names from the company list
  above (e.g. ["AWS"] for Amazon Web Services); [] for every segment. Company totals are always included.
- date_start / date_end: YYYY-MM-DD bounds on the period end date if the question implies a time window
  (e.g. "last 2 years"), else "". Today is {today}. For a company with a non-calendar fiscal year, a quarter
  or year like "Q2 2025" is ambiguous: make the window cover both the fiscal and the calendar reading
  (Microsoft fiscal Q2 2025 is Oct-Dec 2024; calendar Q2 2025 is Apr-Jun 2025).
- narrative_search_query: a concise, semantically rich query for vector search over filing text;
  "" if purely quantitative."""

_FALLBACK_ANALYSIS = {"query_type": "both", "tickers": [], "sql_keywords": ["revenue", "income"],
                      "period_granularity": "any", "include_segments": False, "segment_filter": [],
                      "date_start": "", "date_end": ""}


def analyze_query(question):
    prompt = ROUTER_PROMPT.format(question=question, companies=_company_lines(), today=date.today().isoformat())
    try:
        analysis = json.loads(call_claude(prompt, max_tokens=4000, effort="low", output_format=ROUTER_SCHEMA))
    except (json.JSONDecodeError, ClaudeRefusal) as e:
        print(f"⚠ Query analysis failed ({e}); using defaults")
        analysis = {**_FALLBACK_ANALYSIS, "narrative_search_query": question}
    analysis["sql_keywords"] = [k.replace("'", "").replace("’", "") for k in analysis["sql_keywords"]]
    return analysis


# ── retrieval: financial facts ───────────────────────────────

_GRANULARITY_FILTER = {
    "quarterly": "fiscal_period IN ('Q', 'instant')",
    "annual": "(fiscal_period = 'FY' OR (fiscal_period = 'instant' AND fiscal_quarter = 4))",
    "any": "fiscal_period IN ('Q', 'FY', 'instant')",  # YTD/TTM totals excluded: they read as quarters
}


def search_financial_facts(keywords, tickers=None, date_start="", date_end="", granularity="any", limit=20,
                           segments=False, segment_filter=()):
    """Keyword search over financial_facts_clean, balanced per ticker.

    segments=False searches company totals only; True also searches business-segment facts,
    restricted to `segment_filter` names (substring match) when given.
    """
    if not keywords:
        return pd.DataFrame()
    params, conditions = {}, []
    for i, kw in enumerate(keywords[:8]):
        # DB labels use U+2019 apostrophes; strip both sides so "stockholders" matches "stockholders’".
        conditions.append(f"(REPLACE(REPLACE(label, E'\\u2019', ''), '''', '') ILIKE :kw{i} "
                          f"OR concept ILIKE :kw{i} OR reported_concept ILIKE :kw{i})")
        # Words in order with anything between, so "Purchases of property and equipment" also
        # matches "Purchases of property, plant, and equipment".
        params[f"kw{i}"] = "%" + "%".join(kw.split()) + "%"
    where = [f"({' OR '.join(conditions)})", _GRANULARITY_FILTER.get(granularity, _GRANULARITY_FILTER["any"])]
    if not segments:
        where.append("segment IS NULL")
    elif segment_filter:
        names = [f"segment ILIKE :seg{i}" for i in range(len(segment_filter))]
        where.append(f"(segment IS NULL OR {' OR '.join(names)})")
        params.update({f"seg{i}": f"%{name}%" for i, name in enumerate(segment_filter)})
    if date_start:
        where.append("period_end >= :date_start")
        params["date_start"] = date_start
    if date_end:
        where.append("period_end <= :date_end")
        params["date_end"] = date_end
    # Breadth first: the latest value of every matching line item, then the next-latest, ...
    # (sorting by label alone let alphabetically early labels crowd out e.g. capex). A line
    # item is its series concept, so one whose label wording changed is still one item.
    sql = f"""
        SELECT ticker, fiscal_year, fiscal_quarter, fiscal_period, period_start, period_end,
               statement_type, label, segment, value, unit, is_derived, filing_type, filing_date
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY concept, segment, unit, fiscal_period
                                         ORDER BY period_end DESC) AS rn
            FROM financial_facts_clean
            WHERE {' AND '.join(where)} AND ticker = :ticker
        ) ranked
        ORDER BY rn, label, segment
        LIMIT :limit
    """
    per_ticker = limit * 3 if (date_start or date_end) else limit
    if segments:  # each segment repeats every line item
        per_ticker *= 2 if segment_filter else 3
    frames = []
    with db.get_engine().connect() as conn:
        for ticker in (tickers or config.TICKERS):
            df = pd.read_sql(text(sql), conn, params={**params, "ticker": ticker, "limit": per_ticker})
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not tickers:  # "all companies": keep the overall result size bounded
        out = out.groupby("ticker", group_keys=False).head(max(5, limit // 2))
    return out.sort_values(["ticker", "label", "segment", "period_end"], ascending=[True, True, True, False],
                           na_position="first").reset_index(drop=True)


# ── retrieval: narrative ─────────────────────────────────────

_NARRATIVE_SQL = """
    SELECT * FROM (
        SELECT ticker, filing_type, filing_date, section, chunk_text,
               1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM document_chunks
        WHERE embedding IS NOT NULL {ticker_filter}
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :inner_limit
    ) sub
    WHERE similarity >= :threshold
    ORDER BY similarity DESC
    LIMIT :limit
"""


def search_narrative(query_text, tickers=None, limit=5, similarity_threshold=0.35):
    """Vector search over filing text. With several tickers, each gets its own share of results."""
    if not query_text:
        return pd.DataFrame()
    q = to_pgvector(embed_query(query_text))
    groups = [[t] for t in tickers] if tickers and len(tickers) > 1 else [tickers or []]
    per_group = max(2, math.ceil(limit / len(groups)))
    frames = []
    with db.get_engine().connect() as conn:
        # HNSW + iterative scan keeps returning candidates until the ticker filter is satisfied.
        conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        for group in groups:
            params = {"q": q, "inner_limit": per_group * 3, "threshold": similarity_threshold, "limit": per_group}
            ticker_filter = ""
            if group:
                ticker_filter = "AND ticker IN (" + ", ".join(f":t{i}" for i in range(len(group))) + ")"
                params.update({f"t{i}": t for i, t in enumerate(group)})
            frames.append(pd.read_sql(text(_NARRATIVE_SQL.format(ticker_filter=ticker_filter)), conn, params=params))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.sort_values("similarity", ascending=False).reset_index(drop=True) if not out.empty else out


# ── synthesis ────────────────────────────────────────────────

def format_value(value, unit):
    v = float(value)
    if unit == "USD_millions":
        return f"${v:,.0f}M"
    if unit == "USD_per_share":
        return f"${v:,.2f}/share"
    if unit == "shares":
        return f"{v:,.0f} shares"
    return f"{v:,.4g} {unit}"


def format_period(row):
    fy, fq, fp = row["fiscal_year"], row["fiscal_quarter"], row["fiscal_period"]
    if fp == "instant":
        return f"as of {row['period_end']} (end of FY{fy} Q{int(fq)})"
    name = f"FY{fy}" if fp == "FY" else f"FY{fy} Q{int(fq)}"
    return f"{name}, {row['period_start']} to {row['period_end']}"


def build_context(data):
    parts = []
    fin = data.get("financial_data")
    if fin is not None and not fin.empty:
        parts.append("# Financial Data\n")
        for ticker, rows in fin.groupby("ticker", sort=True):
            parts.append(f"\n## {ticker}\n")
            for _, r in rows.iterrows():
                derived = " [derived: full-period total minus year-to-date]" if r["is_derived"] else ""
                segment = f" [segment: {r['segment']}]" if isinstance(r.get("segment"), str) else ""
                parts.append(f"- {r['label']}{segment}: {format_value(r['value'], r['unit'])} — {format_period(r)} "
                             f"[latest reported in {r['filing_type']} filed {r['filing_date']}]{derived}\n")
    nar = data.get("narrative_data")
    if nar is not None and not nar.empty:
        parts.append("\n# Narrative Context\n")
        for _, r in nar.iterrows():
            parts.append(f"\n## {r['ticker']} — {r['section'][:80]} ({r['filing_type']} filed {r['filing_date']}, "
                         f"similarity {r['similarity']:.3f})\n\n{r['chunk_text']}\n")
    return "".join(parts)


SYNTHESIS_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the data below, retrieved from SEC filings.

Question: {question}

Fiscal years: {fiscal_note}. Quarter labels (e.g. "FY2025 Q2") are fiscal quarters; the calendar dates are given with each value.
Values marked "derived" were computed from reported figures (e.g. Q4 = full year minus nine months) because the filing does not report that quarter on its own.
Each value is the most recently reported figure for its period, so restatements win; the cited filing is often a later one that shows the period as a prior-year comparative. That is expected, not an inconsistency.
Values marked with a segment belong to that business segment only; unmarked values are company totals. Segment capex can include assets acquired under finance leases, so it need not sum to the cash capex line.
"Leases not yet commenced" is the total undiscounted payments over the full term (often 15-20 years) of leases signed but not started as of that date: a measure of committed future capacity, not spending in any one period. "Right-of-use assets obtained for new leases" is leased capacity added in the period, which the cash capex line does not include.

Retrieved data:
{context}

Instructions:
- Give a clear, concise answer that cites specific numbers, periods and filings.
- When comparing companies, use a compact table and note differing fiscal calendars.
- If the data does not fully answer the question, say what is missing.
- Do not use information that is not in the data."""


def synthesize_answer(question, data):
    context = build_context(data)
    if not context.strip():
        return "No relevant data found to answer this question."
    months = fiscal_year_ends()
    odd = [f"{t} ends in {date(2000, m, 1):%B}" for t, m in sorted(months.items()) if m != 12]
    fiscal_note = "calendar year unless noted" + (f" ({'; '.join(odd)})" if odd else "")
    try:
        return call_claude(SYNTHESIS_PROMPT.format(question=question, fiscal_note=fiscal_note, context=context),
                           max_tokens=16000)
    except ClaudeRefusal as e:
        return f"⚠ {e}"


# ── entry points ─────────────────────────────────────────────

def query_edgar(question, max_financial_results=20, max_narrative_results=5, synthesize=True, verbose=True):
    """Route a question, retrieve facts and/or narrative, and optionally synthesize an answer."""
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n{'=' * 60}\nQUERY: {question}\n{'=' * 60}")
    a = analyze_query(question)
    log(f"\nQuery type: {a['query_type']}\nTickers: {a['tickers'] or 'all'}\nSQL keywords: {a['sql_keywords']}"
        f"\nGranularity: {a['period_granularity']}" + (f"\nSegments: {', '.join(a.get('segment_filter') or []) or 'all'}" if a.get("include_segments") else ""))
    if a["date_start"] or a["date_end"]:
        log(f"Date range: {a['date_start'] or '…'} to {a['date_end'] or '…'}")
    if a["narrative_search_query"]:
        log(f"Narrative query: {a['narrative_search_query']}")

    results = {"analysis": a}
    if a["query_type"] in ("quantitative", "both"):
        log("\n🔢 Searching financial facts...")
        results["financial_data"] = search_financial_facts(
            a["sql_keywords"], a["tickers"], a["date_start"], a["date_end"], a["period_granularity"],
            max_financial_results, segments=a.get("include_segments", False),
            segment_filter=a.get("segment_filter") or ())
        log(f"   Found {len(results['financial_data'])} financial facts")
    if a["query_type"] in ("qualitative", "both"):
        log("\n📄 Searching narrative chunks...")
        results["narrative_data"] = search_narrative(a["narrative_search_query"] or question, a["tickers"],
                                                     max_narrative_results)
        log(f"   Found {len(results['narrative_data'])} relevant chunks")
    if synthesize:
        log("\n🤖 Synthesizing answer with Claude...")
        results["answer"] = synthesize_answer(question, results)
    return results


def ask(question, **kwargs):
    """Simple interface: ask a question, print the answer, return the full result."""
    result = query_edgar(question, synthesize=True, **kwargs)
    print(f"\n{'=' * 60}\nANSWER\n{'=' * 60}\n{result['answer']}\n")
    return result
