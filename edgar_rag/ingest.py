"""Ingestion pipeline: discover filings, extract facts + narrative, embed.

Incremental by accession number: `update` only touches filings that are not yet
in the `filings` table (or whose previous attempt failed), so a refresh after a
few months costs a handful of downloads rather than a full re-run.
"""

from datetime import datetime

from sqlalchemy import text

from . import config, db, embeddings, facts, metrics, narrative, sec


def _select(tickers):
    if not tickers:
        return config.COMPANIES
    unknown = set(tickers) - set(config.COMPANIES_BY_TICKER)
    if unknown:
        raise SystemExit(f"Unknown ticker(s): {', '.join(sorted(unknown))}. Add them to config.COMPANIES.")
    return [config.COMPANIES_BY_TICKER[t] for t in tickers]


def _fye(conn, spec):
    month = sec.fiscal_year_end_month(spec)
    conn.execute(text("UPDATE companies SET fiscal_year_end_month = :m WHERE ticker = :t"),
                 {"m": month, "t": spec.ticker})
    return month


def _existing(conn, ticker):
    rows = conn.execute(text("SELECT accession, facts_status, chunks_status FROM filings WHERE ticker = :t"),
                        {"t": ticker}).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _save_filing(conn, filing, spec, **fields):
    params = {"accession": filing.accession_number, "ticker": spec.ticker, "cik": spec.cik,
              "filing_type": filing.form, "filing_date": filing.filing_date,
              "period_end": getattr(filing, "period_of_report", None) or None,
              "processed_at": datetime.now(), **fields}
    sets = ", ".join(f"{k} = EXCLUDED.{k}" for k in params if k != "accession")
    conn.execute(text(f"""
        INSERT INTO filings ({', '.join(params)}) VALUES ({', '.join(':' + k for k in params)})
        ON CONFLICT (accession) DO UPDATE SET {sets}
    """), params)


def _ingest_facts(conn, filing, spec, fye):
    rows, status = facts.extract_facts(filing, spec.ticker, fye)
    conn.execute(text("DELETE FROM financial_facts WHERE accession = :a"), {"a": filing.accession_number})
    n = db.bulk_insert(conn, "financial_facts", rows, facts.FACT_COLUMNS)
    return status, n


def _ingest_chunks(conn, filing, spec):
    rows, status = narrative.extract_narrative_chunks(filing, spec.ticker)
    conn.execute(text("DELETE FROM document_chunks WHERE accession = :a"), {"a": filing.accession_number})
    n = db.bulk_insert(conn, "document_chunks", rows, narrative.CHUNK_COLUMNS)
    return status, n


def process_filing(filing, spec, fye, do_facts=True, do_chunks=True):
    periodic = filing.form in config.PERIODIC_FORMS
    print(f"  📄 {filing.form} filed {filing.filing_date} ({filing.accession_number})")
    fields = {}
    with db.get_engine().begin() as conn:
        # The filings row must exist before facts reference it.
        _save_filing(conn, filing, spec)
        if do_facts:
            if periodic:
                status, n = _ingest_facts(conn, filing, spec, fye)
            else:
                status, n = "none", 0  # prospectuses: narrative only
            fields.update(facts_status=status, n_facts=n)
            print(f"      facts: {n} ({status})")
        if do_chunks:
            status, n = _ingest_chunks(conn, filing, spec)
            if status == "ok" and n == 0:
                status = "empty"  # parsed, but no narrative sections found: worth a look
            fields.update(chunks_status=status, n_chunks=n)
            print(f"      chunks: {n} ({status})")
        conn.execute(text(f"UPDATE filings SET {', '.join(f'{k} = :{k}' for k in fields)}, error = NULL, processed_at = now() "
                          "WHERE accession = :accession"), {**fields, "accession": filing.accession_number})


def build_series(conn, spec):
    rows = conn.execute(text("""
        SELECT accession, filing_date, statement_type, concept, label, unit, period_start, period_end, value
        FROM financial_facts
        WHERE ticker = :t AND NOT is_derived AND segment IS NULL AND statement_type = ANY(:st)
    """), {"t": spec.ticker, "st": list(facts.SERIES_STATEMENTS)}).mappings().all()
    series = facts.build_series(rows)
    conn.execute(text("DELETE FROM fact_series WHERE ticker = :t"), {"t": spec.ticker})
    db.bulk_insert(conn, "fact_series",
                   [{"ticker": spec.ticker, "statement_type": st, "concept": c, "label": lab,
                     "series_concept": sc, "series_label": sl} for (st, c, lab), (sc, sl) in series.items()],
                   ["ticker", "statement_type", "concept", "label", "series_concept", "series_label"])
    merged = sum(1 for (_, c, _), (sc, _) in series.items() if c != sc)
    qualified = sum(1 for (_, _, lab), (_, sl) in series.items() if lab != sl)
    print(f"  ✓ {spec.ticker}: {len(series)} statement line items, {merged} mapped to a re-tagged "
          f"concept, {qualified} labels qualified")


def finalize(conn, spec, fye):
    """Company-level steps after its filings change: series mapping, derived quarters, metric map."""
    build_series(conn, spec)
    derive_quarters(conn, spec, fye)
    build_metric_map(conn, spec)


def build_metric_map(conn, spec):
    concepts = conn.execute(text("SELECT DISTINCT concept FROM financial_facts_clean WHERE ticker = :t AND segment IS NULL"),
                            {"t": spec.ticker}).scalars().all()
    rows = [{"ticker": spec.ticker, "metric": m, "concept": c, "rank": r}
            for m, c, r in metrics.build_metric_map(concepts)]
    conn.execute(text("DELETE FROM metric_concepts WHERE ticker = :t"), {"t": spec.ticker})
    db.bulk_insert(conn, "metric_concepts", rows, ["ticker", "metric", "concept", "rank"])
    missing = sorted(set(metrics.METRIC_NAMES) - {r["metric"] for r in rows})
    print(f"  ✓ {spec.ticker}: {len(metrics.METRIC_NAMES) - len(missing)} of {len(metrics.METRIC_NAMES)} metrics mapped"
          + (f" (none for: {', '.join(missing)})" if missing else ""))


def derive_quarters(conn, spec, fye):
    conn.execute(text("DELETE FROM financial_facts WHERE ticker = :t AND is_derived"), {"t": spec.ticker})
    # Every filing's version of each reported flow (not just the latest), under its series
    # concept and label, so a quarter is derived from two figures on the same basis.
    rows = conn.execute(text("""
        SELECT f.accession, f.ticker, f.filing_type, f.filing_date, f.statement_type,
               COALESCE(s.series_concept, f.concept) AS concept, COALESCE(s.series_label, f.label) AS label,
               f.standard_concept, f.segment, f.value, f.unit, f.period_start, f.period_end, f.fiscal_period
        FROM financial_facts f
        LEFT JOIN fact_series s
          ON s.ticker = f.ticker AND s.statement_type = f.statement_type AND s.concept = f.concept
         AND s.label = f.label
        WHERE f.ticker = :t AND NOT f.is_derived AND f.unit = 'USD_millions' AND f.period_type = 'duration'
          AND f.statement_type = ANY(:st)
    """), {"t": spec.ticker, "st": list(facts.FLOW_FACT_STATEMENTS)}).mappings().all()
    derived = facts.derive_quarters([dict(r) for r in rows], fye)
    n = db.bulk_insert(conn, "financial_facts", derived, facts.FACT_COLUMNS)
    print(f"  ✓ {spec.ticker}: derived {n} quarterly values from year-to-date differences")


def update(tickers=None, dry_run=False, embed=True):
    """Ingest every in-scope filing not yet processed successfully, then embed new chunks."""
    db.migrate()
    for spec in _select(tickers):
        print(f"\n{'=' * 60}\n{spec.ticker} — {spec.name} (CIK {spec.cik})\n{'=' * 60}")
        with db.get_engine().begin() as conn:
            fye = _fye(conn, spec)
            existing = _existing(conn, spec.ticker)
        filings = sec.list_filings(spec)
        todo = []
        for f in filings:
            prev = existing.get(f.accession_number)
            if prev is None:
                todo.append((f, True, True))
            else:
                facts_status, chunks_status = prev
                redo_facts = facts_status in (None, "failed")
                redo_chunks = chunks_status in (None, "failed")
                if redo_facts or redo_chunks:
                    todo.append((f, redo_facts, redo_chunks))
        print(f"  {len(filings)} filings in scope, {len(todo)} to process")
        if dry_run:
            for f, _, _ in todo:
                print(f"    would ingest {f.form} {f.filing_date} {f.accession_number}")
            continue
        for f, do_facts, do_chunks in todo:
            try:
                process_filing(f, spec, fye, do_facts, do_chunks)
            except Exception as e:  # keep going; the failure is recorded and retried next run
                print(f"      ✗ {e}")
                prev_facts, prev_chunks = existing.get(f.accession_number, (None, None))
                with db.get_engine().begin() as conn:
                    _save_filing(conn, f, spec, error=str(e)[:2000],
                                 facts_status="failed" if do_facts else prev_facts,
                                 chunks_status="failed" if do_chunks else prev_chunks)
        if todo:
            with db.get_engine().begin() as conn:
                finalize(conn, spec, fye)
    if embed and not dry_run:
        print()
        embeddings.embed_pending()


def rebuild_facts(tickers=None):
    """Re-extract XBRL facts for every ingested periodic filing (chunks are untouched)."""
    db.migrate()
    for spec in _select(tickers):
        print(f"\n{'=' * 60}\n{spec.ticker} — rebuilding facts\n{'=' * 60}")
        with db.get_engine().begin() as conn:
            fye = _fye(conn, spec)
            existing = _existing(conn, spec.ticker)
        for f in sec.list_filings(spec):
            if f.accession_number in existing and f.form in config.PERIODIC_FORMS:
                try:
                    process_filing(f, spec, fye, do_facts=True, do_chunks=False)
                except Exception as e:
                    print(f"      ✗ {e}")
                    with db.get_engine().begin() as conn:
                        conn.execute(text("UPDATE filings SET facts_status = 'failed', error = :e WHERE accession = :a"),
                                     {"e": str(e)[:2000], "a": f.accession_number})
        with db.get_engine().begin() as conn:
            finalize(conn, spec, fye)


def rebuild_chunks(tickers=None, accessions=None, embed=True):
    """Re-extract narrative chunks for ingested filings (facts are untouched), then embed them.

    accessions: restrict to these filings (e.g. the ones `check` reports as thin or cut off).
    The filings' existing chunks and embeddings are replaced.
    """
    db.migrate()
    wanted = set(accessions) if accessions is not None else None
    for spec in _select(tickers):
        with db.get_engine().begin() as conn:
            fye = _fye(conn, spec)
            existing = _existing(conn, spec.ticker)
        todo = [f for f in sec.list_filings(spec)
                if f.accession_number in existing and (wanted is None or f.accession_number in wanted)]
        if not todo:
            continue
        print(f"\n{'=' * 60}\n{spec.ticker} — rebuilding narrative chunks for {len(todo)} filings\n{'=' * 60}")
        for f in todo:
            try:
                process_filing(f, spec, fye, do_facts=False, do_chunks=True)
            except Exception as e:
                print(f"      ✗ {e}")
                with db.get_engine().begin() as conn:
                    conn.execute(text("UPDATE filings SET chunks_status = 'failed', error = :e WHERE accession = :a"),
                                 {"e": str(e)[:2000], "a": f.accession_number})
    if embed:
        print()
        embeddings.embed_pending()


def rebuild_series(tickers=None):
    """Recompute series mappings and derived quarters from stored facts (no downloads)."""
    db.migrate()
    for spec in _select(tickers):
        with db.get_engine().begin() as conn:
            fye = conn.execute(text("SELECT fiscal_year_end_month FROM companies WHERE ticker = :t"),
                               {"t": spec.ticker}).scalar() or _fye(conn, spec)
            finalize(conn, spec, fye)


def status():
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT f.ticker, count(*) AS filings, min(f.filing_date), max(f.filing_date),
                   count(*) FILTER (WHERE f.facts_status = 'failed' OR f.chunks_status IN ('failed', 'empty')) AS failed,
                   (SELECT count(*) FROM financial_facts x WHERE x.ticker = f.ticker AND NOT x.is_derived) AS facts,
                   (SELECT count(*) FROM financial_facts x WHERE x.ticker = f.ticker AND x.is_derived) AS derived,
                   (SELECT count(*) FROM document_chunks c WHERE c.ticker = f.ticker) AS chunks,
                   (SELECT count(embedding) FROM document_chunks c WHERE c.ticker = f.ticker) AS embedded
            FROM filings f GROUP BY f.ticker ORDER BY f.ticker
        """)).fetchall()
    print(f"{'ticker':6} {'filings':>7} {'first':>10} {'last':>10} {'failed':>6} {'facts':>7} {'derived':>7} {'chunks':>7} {'embedded':>8}")
    tot = [0] * 6
    for t, n, lo, hi, failed, nf, nd, nc, ne in rows:
        print(f"{t:6} {n:7} {lo!s:>10} {hi!s:>10} {failed:6} {nf:7,} {nd:7,} {nc:7,} {ne:8,}")
        for i, v in enumerate((n, failed, nf, nd, nc, ne)):
            tot[i] += v
    print(f"{'total':6} {tot[0]:7} {'':>10} {'':>10} {tot[1]:6} {tot[2]:7,} {tot[3]:7,} {tot[4]:7,} {tot[5]:8,}")
    from . import checks
    thin = checks.thin_filings()
    if not thin.empty:
        print(f"⚠ {len(thin)} filing(s) have far less narrative than usual for that company; "
              "see `python -m edgar_rag check`")
