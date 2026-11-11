"""Natural-language query interface: route with Claude, retrieve from SQL + pgvector, synthesize.

Moved out of edgar_rag_interface.ipynb so the notebook is a thin client:

    from edgar_rag.query import ask
    ask("How much did Amazon spend on capex each quarter of 2025?")
"""

import json
import math
import os
import re
from datetime import date, timedelta
from functools import lru_cache

import anthropic
import pandas as pd
from sqlalchemy import text

from . import config, db
from .metrics import METRIC_NAMES, METRICS
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
# Everything the router is told about the data is read from the database, so it cannot drift
# from what is actually stored. Cached per data version: a new ingest refreshes it.

def _data_version():
    with db.get_engine().connect() as conn:
        return conn.execute(text("SELECT max(processed_at) FROM filings")).scalar()


@lru_cache(maxsize=2)
def _metadata(version):
    with db.get_engine().connect() as conn:
        fye = {t: m for t, m in conn.execute(text("SELECT ticker, fiscal_year_end_month FROM companies")) if m}
        # Segment line items reported since 2024 (older segments were reorganized away), latest first.
        segment_rows = conn.execute(text("""
            SELECT ticker, segment, label, bool_or(fiscal_period = 'Q' OR period_type = 'instant') AS quarterly,
                   max(period_end) AS latest
            FROM financial_facts_clean
            WHERE segment IS NOT NULL AND period_end >= '2024-01-01'
            GROUP BY ticker, segment, label
        """)).fetchall()
        note_rows = conn.execute(text("""
            SELECT statement_type, ticker, label, max(period_end) AS latest
            FROM financial_facts_clean
            WHERE segment IS NULL AND statement_type IN ('leases', 'backlog', 'ppe', 'commitments')
              AND period_end >= '2024-01-01'
            GROUP BY statement_type, ticker, label
        """)).fetchall()
        mapped = conn.execute(text("SELECT DISTINCT metric, ticker FROM metric_concepts")).fetchall()
    segments, segment_items = {}, {}
    for ticker, segment, label, quarterly, latest in sorted(segment_rows, key=lambda r: (r[0], r[4], r[1]), reverse=True):
        if segment not in segments.setdefault(ticker, []):
            segments[ticker].append(segment)
        item = segment_items.setdefault(ticker, {})
        item[label] = item.get(label, False) or quarterly
    notes = {}
    for statement_type, ticker, label, _ in note_rows:
        notes.setdefault(statement_type, {}).setdefault(label, set()).add(ticker)
    metric_tickers = {}
    for metric, ticker in mapped:
        metric_tickers.setdefault(metric, set()).add(ticker)
    return {"fye": fye, "segments": segments, "segment_items": segment_items, "notes": notes,
            "metric_tickers": metric_tickers}


def metadata():
    return _metadata(_data_version())


def fiscal_year_ends():
    return metadata()["fye"]


def segment_names():
    """{ticker: [segment, ...]} reported since 2024, latest first."""
    return metadata()["segments"]


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


def _tickers(tickers):
    """'all companies' or 'AMZN, D' / 'all except SPCX', whichever is shorter."""
    tickers = set(tickers)
    missing = [t for t in config.TICKERS if t not in tickers]
    if not missing:
        return "all companies"
    present = [t for t in config.TICKERS if t in tickers]
    return ", ".join(present) if len(present) <= len(missing) else f"all except {', '.join(missing)}"


def _metric_lines():
    covered = metadata()["metric_tickers"]
    return "\n".join(f"- {m.name}: {m.description} [{_tickers(covered.get(m.name, ()))}]"
                     for m in METRICS if covered.get(m.name))


def _segment_lines():
    lines = []
    for ticker in config.TICKERS:
        items = metadata()["segment_items"].get(ticker)
        if items:
            described = ", ".join(f'"{label}"' + ("" if quarterly else " (annual only)")
                                  for label, quarterly in sorted(items.items()))
            lines.append(f"- {ticker}: {described}")
    return "\n".join(lines)


_NOTE_TITLES = {
    "leases": "Leases",
    "backlog": "Backlog",
    "commitments": "Purchase commitments other than leases (each company defines its own total)",
}


def _note_lines():
    notes = metadata()["notes"]
    lines = []
    for statement_type, title in _NOTE_TITLES.items():
        for label, tickers in sorted(notes.get(statement_type, {}).items()):
            lines.append(f'- {title}: "{label}" [{_tickers(tickers)}]')
    classes = {}
    for label, tickers in notes.get("ppe", {}).items():
        for t in tickers:
            classes.setdefault(t, []).append(label.split(": ", 1)[-1])
    if classes:
        lines.append('- Gross property and equipment by asset class, labelled "Property and equipment, gross: <class>":')
        lines.extend(f"    {t}: {', '.join(sorted(classes[t]))}" for t in config.TICKERS if t in classes)
    return "\n".join(lines)


# ── routing ──────────────────────────────────────────────────

ROUTER_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "enum": ["quantitative", "qualitative", "both"]},
            "tickers": {"type": "array", "items": {"type": "string", "enum": config.TICKERS}},
            "metrics": {"type": "array", "items": {"type": "string", "enum": METRIC_NAMES}},
            "sql_keywords": {"type": "array", "items": {"type": "string"}},
            "period_granularity": {"type": "string", "enum": ["quarterly", "annual", "any"]},
            "include_segments": {"type": "boolean"},
            "segment_filter": {"type": "array", "items": {"type": "string"}},
            "date_start": {"type": "string"},
            "date_end": {"type": "string"},
            "narrative_search_query": {"type": "string"},
        },
        "required": ["query_type", "tickers", "metrics", "sql_keywords", "period_granularity", "include_segments",
                     "segment_filter", "date_start", "date_end", "narrative_search_query"],
        "additionalProperties": False,
    },
}

ROUTER_PROMPT = """Analyze this question about SEC filings and extract search parameters.

Question: {question}

Companies in the database:
{companies}

Financial facts come from the balance sheet, income statement, cash flow statement and a few
note disclosures. Each fact is tagged with its exact period: a discrete fiscal quarter, a full
fiscal year, or a balance-sheet date. There are two ways to select them, and both can be used:

1. Canonical metrics. One name for the same line item at every company, whatever each company
   calls it. Always use these when the question is about one of them [availability in brackets]:
{metrics}

2. Label keywords (sql_keywords), for line items that are not a canonical metric. They are matched
   against each company's own line-item labels, for example "Technology and infrastructure",
   "General and administrative", "Accrued capital expenditures", "Repurchases of common stock",
   "Deferred revenue", "Marketable securities", "Dividends paid".

Note disclosures stored as line items (reach them through the metric of the same meaning if there
is one, otherwise put the quoted label, or a distinctive part of it, in sql_keywords):
{notes}
"Leases not yet commenced" is the forward build-out pipeline: total future payments on signed,
mostly data-center, leases that have not started. Right-of-use assets obtained are the leased
capacity added in a period, which cash capex does not include. For questions about total
infrastructure spending, commitments or the data-center pipeline, add the lease and backlog
metrics next to capex.

Business-segment line items (only searched when include_segments is true; put the quoted labels
in sql_keywords). What each company reports:
{segments}

Fields:
- query_type: "quantitative" for numbers/metrics, "qualitative" for risks/strategy/narrative, "both" for combined.
- tickers: every company the question mentions, mapped to its ticker. Empty list if none (searches all).
- metrics: the canonical metrics the question needs; [] if none apply.
- sql_keywords: 0-5 phrases that would match line-item labels via SQL ILIKE, for anything the metrics
  do not cover. No apostrophes. [] if the metrics cover the question.
- period_granularity: "quarterly" if the question is about quarters or quarterly trends, "annual" if about
  years or fiscal years, otherwise "any".
- include_segments: true if the question is about a business segment or asks for a segment breakdown;
  otherwise false (company totals only).
- segment_filter: when the question singles out particular segments, their exact names from the company list
  above (e.g. ["AWS"] for Amazon Web Services); [] for every segment. Company totals are always included.
- date_start / date_end: YYYY-MM-DD bounds on the period end date if the question implies a time window
  (e.g. "last 2 years"), else "". Today is {today}. For a company with a non-calendar fiscal year, a quarter
  or year like "Q2 2025" is ambiguous: make the window cover both the fiscal and the calendar reading
  (Microsoft fiscal Q2 2025 is Oct-Dec 2024; calendar Q2 2025 is Apr-Jun 2025). The window also limits
  which filings' text is searched.
- narrative_search_query: a concise, semantically rich query for vector search over filing text;
  "" if purely quantitative."""

_FALLBACK_ANALYSIS = {"query_type": "both", "tickers": [], "metrics": ["revenue", "net_income"], "sql_keywords": [],
                      "period_granularity": "any", "include_segments": False, "segment_filter": [],
                      "date_start": "", "date_end": ""}


def analyze_query(question):
    prompt = ROUTER_PROMPT.format(question=question, companies=_company_lines(), metrics=_metric_lines(),
                                  notes=_note_lines(), segments=_segment_lines(), today=date.today().isoformat())
    try:
        analysis = json.loads(call_claude(prompt, max_tokens=4000, effort="low", output_format=ROUTER_SCHEMA))
    except (json.JSONDecodeError, ClaudeRefusal, anthropic.APIError) as e:
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


_METRIC_SQL = """
    SELECT metric, ticker, fiscal_year, fiscal_quarter, fiscal_period, period_start, period_end,
           statement_type, label, segment, value, unit, is_derived, filing_type, filing_date
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker, metric, fiscal_period ORDER BY period_end DESC) AS rn
        FROM financial_metrics
        WHERE metric = ANY(:metrics) AND ticker = ANY(:tickers) AND {where}
    ) ranked
    WHERE rn <= :periods
    ORDER BY ticker, metric, period_end DESC
"""


def search_metrics(metrics, tickers=None, date_start="", date_end="", granularity="any", periods=None):
    """Canonical metrics (see metrics.py) per company and period, newest first.

    Unlike the keyword search this cannot miss a company because of label wording, and it
    returns a time series: the latest `periods` values per company, metric and period type
    (default 12 quarters / 6 years, or everything inside a date window).
    """
    if not metrics:
        return pd.DataFrame()
    where, params = [_GRANULARITY_FILTER.get(granularity, _GRANULARITY_FILTER["any"])], {}
    if date_start:
        where.append("period_end >= :date_start")
        params["date_start"] = date_start
    if date_end:
        where.append("period_end <= :date_end")
        params["date_end"] = date_end
    if periods is None:
        periods = 40 if (date_start or date_end) else {"quarterly": 12, "annual": 6}.get(granularity, 8)
        if not tickers:  # all companies: keep the context bounded
            periods = min(periods, 8)
    params.update(metrics=list(metrics), tickers=list(tickers or config.TICKERS), periods=periods)
    with db.get_engine().connect() as conn:
        return pd.read_sql(text(_METRIC_SQL.format(where=" AND ".join(where))), conn, params=params)


def search_facts(metrics, keywords, tickers=None, date_start="", date_end="", granularity="any", limit=20,
                 segments=False, segment_filter=()):
    """Metric rows first, then keyword matches that are not already among them."""
    by_metric = search_metrics(metrics, tickers, date_start, date_end, granularity)
    by_keyword = search_financial_facts(keywords, tickers, date_start, date_end, granularity, limit,
                                        segments=segments, segment_filter=segment_filter)
    frames = [f for f in (by_metric, by_keyword) if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "metric" not in out.columns:
        out["metric"] = None
    out = out.drop_duplicates(["ticker", "label", "segment", "fiscal_period", "period_start", "period_end"])
    return out.sort_values(["ticker", "metric", "label", "segment", "period_end"],
                           ascending=[True, True, True, True, False], na_position="last").reset_index(drop=True)


# ── retrieval: narrative ─────────────────────────────────────

_NARRATIVE_SQL = """
    SELECT ticker, filing_type, filing_date, section, chunk_text,
           1 - (embedding <=> CAST(:q AS vector)) AS similarity
    FROM document_chunks
    WHERE embedding IS NOT NULL {filters}
    ORDER BY embedding <=> CAST(:q AS vector)
    LIMIT :inner_limit
"""

# A period that ends inside the requested window is reported in a filing made up to about
# this long afterwards (10-Ks are due 60-90 days after year end).
_FILING_LAG = timedelta(days=100)


def _passage_key(chunk):
    """Same passage, ignoring whitespace, punctuation and numbers that change between filings."""
    letters = re.sub(r"[^a-z]+", "", chunk.lower())
    return letters.removeprefix("tableofcontents")[:160]


def collapse_repeats(df, keep="latest"):
    """One row per distinct passage.

    Risk factors and accounting policies are repeated nearly verbatim in every filing, so the
    nearest neighbours of a query are often one paragraph from eight filings. The kept row is
    the latest (or earliest) filing that carries it, with the best similarity of the group;
    `repeats` counts the other retrieved filings with the same passage and `first_filed` /
    `last_filed` give their span.
    """
    if df.empty:
        return df.assign(repeats=pd.Series(dtype=int), first_filed=None, last_filed=None)
    df = df.assign(_key=df["chunk_text"].map(_passage_key))
    grouped = df.groupby("_key")
    stats = grouped.agg(similarity=("similarity", "max"), repeats=("filing_date", "nunique"),
                        first_filed=("filing_date", "min"), last_filed=("filing_date", "max"))
    stats["repeats"] -= 1
    order = df.sort_values(["filing_date", "similarity"], ascending=[keep != "latest", False])
    kept = order.drop_duplicates("_key").drop(columns="similarity").join(stats, on="_key")
    return kept.drop(columns="_key")


def search_narrative(query_text, tickers=None, limit=5, similarity_threshold=0.35, date_start="", date_end="",
                     keep="latest", recency_weight=0.01):
    """Vector search over filing text, one row per distinct passage.

    With several tickers, each gets its own share of results. date_start / date_end bound the
    reporting period (YYYY-MM-DD), like the fact search. Repeated passages are collapsed (see
    collapse_repeats; keep="earliest" shows when something was first said). Among passages of
    similar relevance newer filings rank first: `recency_weight` is the similarity given up
    per year of age.
    """
    if not query_text:
        return pd.DataFrame()
    q = to_pgvector(embed_query(query_text))
    groups = [[t] for t in tickers] if tickers and len(tickers) > 1 else [tickers or []]
    per_group = max(2, math.ceil(limit / len(groups)))
    filters, base = "", {"q": q, "inner_limit": max(40, per_group * 10)}
    if date_start:
        filters += " AND filing_date >= :date_start"
        base["date_start"] = date_start
    if date_end:
        filters += " AND filing_date <= :date_end"
        base["date_end"] = date.fromisoformat(date_end) + _FILING_LAG
    frames = []
    with db.get_engine().connect() as conn:
        # HNSW + iterative scan keeps returning candidates until the filters are satisfied.
        conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        for group in groups:
            params, ticker_filter = dict(base), ""
            if group:
                ticker_filter = " AND ticker IN (" + ", ".join(f":t{i}" for i in range(len(group))) + ")"
                params.update({f"t{i}": t for i, t in enumerate(group)})
            found = pd.read_sql(text(_NARRATIVE_SQL.format(filters=filters + ticker_filter)), conn, params=params)
            found = collapse_repeats(found[found["similarity"] >= similarity_threshold], keep)
            frames.append(_rank(found, recency_weight).head(per_group))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return _rank(pd.concat(frames, ignore_index=True), recency_weight).reset_index(drop=True)


def _rank(df, recency_weight):
    if df.empty:
        return df
    age_years = (pd.Timestamp(date.today()) - pd.to_datetime(df["filing_date"])).dt.days / 365.25
    return df.assign(_score=df["similarity"] - recency_weight * age_years).sort_values(
        "_score", ascending=False).drop(columns="_score")


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
                metric = f" (metric: {r['metric']})" if isinstance(r.get("metric"), str) else ""
                parts.append(f"- {r['label']}{metric}{segment}: {format_value(r['value'], r['unit'])} — {format_period(r)} "
                             f"[latest reported in {r['filing_type']} filed {r['filing_date']}]{derived}\n")
    nar = data.get("narrative_data")
    if nar is not None and not nar.empty:
        parts.append("\n# Narrative Context\n")
        for _, r in nar.iterrows():
            repeated = ""
            if r.get("repeats", 0):
                repeated = (f"; the same passage is in {int(r['repeats'])} other retrieved filing(s), "
                            f"from {r['first_filed']} to {r['last_filed']}")
            parts.append(f"\n## {r['ticker']} — {(r['section'] or '')[:80]} ({r['filing_type']} filed {r['filing_date']}, "
                         f"similarity {r['similarity']:.3f}{repeated})\n\n{r['chunk_text']}\n")
    return "".join(parts)


SYNTHESIS_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the data below, retrieved from SEC filings.

Question: {question}

Fiscal years: {fiscal_note}. Quarter labels (e.g. "FY2025 Q2") are fiscal quarters; the calendar dates are given with each value.
Values marked "derived" were computed from reported figures (e.g. Q4 = full year minus nine months) because the filing does not report that quarter on its own. Both figures are taken from the same reporting basis; if a later filing restated the full year, the derived quarter stays on the earlier basis, so four quarters need not sum to a restated annual total.
A value marked with a metric name is the same line item at every company, whatever its label; definitions can still differ slightly (noted in the label).
"Remaining performance obligations" is contracted revenue not yet recognized (backlog) at that date. "Property and equipment, gross" values are at cost before depreciation; construction in progress / assets not yet in service is capital spent on capacity that is not yet operating. Purchase-commitment totals follow each company's own definition and are not comparable across companies.
Each value is the most recently reported figure for its period, so restatements win; the cited filing is often a later one that shows the period as a prior-year comparative. That is expected, not an inconsistency.
Values marked with a segment belong to that business segment only; unmarked values are company totals. Segment capex can include assets acquired under finance leases, so it need not sum to the cash capex line.
"Leases not yet commenced" is the total undiscounted payments over the full term (often 15-20 years) of leases signed but not started as of that date: a measure of committed future capacity, not spending in any one period. "Right-of-use assets obtained for new leases" is leased capacity added in the period, which the cash capex line does not include.

Retrieved data:
{context}

Instructions:
- Give a clear, concise answer that cites specific numbers, periods and filings.
- When comparing companies, use a compact table and note differing fiscal calendars.
- If the data does not fully answer the question, say what is missing.
- A narrative passage noted as repeated in other filings is standing language; say since when it has appeared
  if that matters to the question, and prefer the dated, specific passages for what changed.
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

def query_edgar(question, max_financial_results=20, max_narrative_results=6, synthesize=True, verbose=True):
    """Route a question, retrieve facts and/or narrative, and optionally synthesize an answer."""
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n{'=' * 60}\nQUERY: {question}\n{'=' * 60}")
    a = analyze_query(question)
    log(f"\nQuery type: {a['query_type']}\nTickers: {a['tickers'] or 'all'}\nMetrics: {a.get('metrics') or []}"
        f"\nSQL keywords: {a['sql_keywords']}"
        f"\nGranularity: {a['period_granularity']}" + (f"\nSegments: {', '.join(a.get('segment_filter') or []) or 'all'}" if a.get("include_segments") else ""))
    if a["date_start"] or a["date_end"]:
        log(f"Date range: {a['date_start'] or '…'} to {a['date_end'] or '…'}")
    if a["narrative_search_query"]:
        log(f"Narrative query: {a['narrative_search_query']}")

    results = {"analysis": a}
    if a["query_type"] in ("quantitative", "both"):
        log("\n🔢 Searching financial facts...")
        results["financial_data"] = search_facts(
            a.get("metrics") or [], a["sql_keywords"], a["tickers"], a["date_start"], a["date_end"],
            a["period_granularity"], max_financial_results, segments=a.get("include_segments", False),
            segment_filter=a.get("segment_filter") or ())
        log(f"   Found {len(results['financial_data'])} financial facts")
    if a["query_type"] in ("qualitative", "both"):
        log("\n📄 Searching narrative chunks...")
        results["narrative_data"] = search_narrative(a["narrative_search_query"] or question, a["tickers"],
                                                     max_narrative_results, date_start=a["date_start"],
                                                     date_end=a["date_end"])
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
