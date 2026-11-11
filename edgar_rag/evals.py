"""Golden questions: score the data, the retrieval and the answers instead of eyeballing them.

    python -m edgar_rag eval                    # data level: stored values only (no API calls)
    python -m edgar_rag eval --level retrieval  # + routing and retrieval (Claude router, Voyage)
    python -m edgar_rag eval --level answer     # + the synthesized answer (one more Claude call each)
    python -m edgar_rag eval --ids amzn-capex-quarters msft-ic-fy23-q4

Cases live in evals/golden.json. Each may carry:
  facts     values that must be in the database, and (retrieval level) among the retrieved facts:
            {ticker, metric | label [, segment], fiscal_period, period_end, value [, tolerance]}
  route     what the router must extract: tickers / metrics (must include these), granularity,
            include_segments
  narrative what the retrieved passages must show: tickers that must appear, phrases that must
            occur (any passage), filed_after (at least one passage from a filing on or after it)
  answer    groups of alternatives; the answer must contain one alternative from every group

Levels build on each other, so a failure is attributed to the first stage that breaks:
wrong data, then wrong routing, then missed retrieval, then a wrong answer.
"""

import json
import re
from datetime import date

import pandas as pd
from sqlalchemy import text

from . import config, db

GOLDEN_PATH = config.PROJECT_ROOT / "evals" / "golden.json"
LEVELS = ("data", "retrieval", "answer")


def load_cases(ids=None):
    cases = json.loads(GOLDEN_PATH.read_text())
    if ids:
        unknown = set(ids) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in ids]
    return cases


def _close(actual, expected, tolerance):
    return actual is not None and abs(float(actual) - expected) <= tolerance


def _stored_value(conn, fact):
    """The database's value for an expected fact, or None."""
    params = {"t": fact["ticker"], "p": fact["fiscal_period"], "e": fact["period_end"]}
    if "metric" in fact:
        sql = ("SELECT value FROM financial_metrics WHERE ticker = :t AND metric = :m "
               "AND fiscal_period = :p AND period_end = :e")
        params["m"] = fact["metric"]
    else:
        sql = ("SELECT value FROM financial_facts_clean WHERE ticker = :t AND label ILIKE :l "
               "AND fiscal_period = :p AND period_end = :e AND segment IS NOT DISTINCT FROM :s")
        params.update(l=fact["label"], s=fact.get("segment"))
    return conn.execute(text(sql + " LIMIT 1"), params).scalar()


def check_data(case):
    """Failures (strings) for the case's expected facts against the stored data."""
    failures = []
    with db.get_engine().connect() as conn:
        for fact in case.get("facts", []):
            actual = _stored_value(conn, fact)
            if not _close(actual, fact["value"], fact.get("tolerance", 0.5)):
                failures.append(f"{_describe(fact)}: expected {fact['value']:,}, database has "
                                f"{'nothing' if actual is None else format(float(actual), ',')}")
    return failures


def _describe(fact):
    what = fact.get("metric") or fact["label"]
    segment = f" [{fact['segment']}]" if fact.get("segment") else ""
    return f"{fact['ticker']} {what}{segment} {fact['fiscal_period']} to {fact['period_end']}"


def check_route(case, analysis):
    want, failures = case.get("route", {}), []
    for field in ("tickers", "metrics"):
        missing = set(want.get(field, [])) - set(analysis.get(field) or [])
        if missing:
            failures.append(f"router {field}: missing {sorted(missing)} (got {analysis.get(field)})")
    if "granularity" in want and analysis["period_granularity"] != want["granularity"]:
        failures.append(f"router granularity: {analysis['period_granularity']}, expected {want['granularity']}")
    if "include_segments" in want and bool(analysis.get("include_segments")) != want["include_segments"]:
        failures.append(f"router include_segments: expected {want['include_segments']}")
    return failures


def check_retrieved_facts(case, frame):
    failures = []
    for fact in case.get("facts", []):
        found = False
        if frame is not None and not frame.empty:
            rows = frame[(frame["ticker"] == fact["ticker"]) & (frame["fiscal_period"] == fact["fiscal_period"])
                         & (frame["period_end"].astype(str) == fact["period_end"])]
            found = any(_close(v, fact["value"], fact.get("tolerance", 0.5)) for v in rows["value"])
        if not found:
            failures.append(f"not retrieved: {_describe(fact)} = {fact['value']:,}")
    return failures


def check_narrative(case, frame):
    want = case.get("narrative")
    if not want:
        return []
    if frame is None or frame.empty:
        return ["no narrative passages retrieved"]
    failures = []
    missing = set(want.get("tickers", [])) - set(frame["ticker"])
    if missing:
        failures.append(f"no passage for {sorted(missing)}")
    text_ = " ".join(frame["chunk_text"]).lower()
    for phrase in want.get("phrases", []):
        if phrase.lower() not in text_:
            failures.append(f"no passage mentions \"{phrase}\"")
    if "filed_after" in want and max(frame["filing_date"]) < date.fromisoformat(want["filed_after"]):
        failures.append(f"newest passage is from {max(frame['filing_date'])}, expected one from "
                        f"{want['filed_after']} or later")
    return failures


def check_answer(case, answer):
    plain = re.sub(r"[,$*]", "", answer or "").lower()
    failures = []
    for group in case.get("answer", []):
        if not any(re.sub(r"[,$*]", "", alt).lower() in plain for alt in group):
            failures.append(f"answer lacks any of {group}")
    return failures


def run(level="data", ids=None):
    """Run the golden cases up to `level`. Prints a line per case; returns the number that failed."""
    if level not in LEVELS:
        raise SystemExit(f"level must be one of {LEVELS}")
    cases, failed, rows = load_cases(ids), 0, []
    for case in cases:
        failures = check_data(case)
        if level != "data":
            from .query import query_edgar
            result = query_edgar(case["question"], synthesize=(level == "answer"), verbose=False)
            failures += check_route(case, result["analysis"])
            failures += check_retrieved_facts(case, result.get("financial_data"))
            failures += check_narrative(case, result.get("narrative_data"))
            if level == "answer":
                failures += check_answer(case, result.get("answer"))
        failed += bool(failures)
        rows.append({"case": case["id"], "result": "pass" if not failures else "FAIL"})
        print(f"{'✓' if not failures else '✗'} {case['id']}: {case['question']}")
        for f in failures:
            print(f"      {f}")
    print(f"\n{len(cases) - failed} of {len(cases)} cases pass at level '{level}'")
    return failed
