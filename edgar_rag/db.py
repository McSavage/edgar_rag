"""Database engine, schema and migrations.

`migrate()` is idempotent: it creates the schema on a fresh database and
upgrades the pre-refactor layout in place (see `_migrate_legacy`).
"""

from functools import lru_cache

from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

from . import config


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(config.database_url())


def _column_exists(conn, table, column):
    return conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :t AND column_name = :c
    """), {"t": table, "c": column}).first() is not None


def _table_exists(conn, table):
    return conn.execute(text("SELECT to_regclass(:t)"), {"t": f"public.{table}"}).scalar() is not None


SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id                    SERIAL PRIMARY KEY,
    ticker                TEXT UNIQUE NOT NULL,
    cik                   BIGINT,
    name                  TEXT,
    sector                TEXT,
    fiscal_year_end_month INTEGER,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS filings (
    id             SERIAL PRIMARY KEY,
    accession      TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    cik            BIGINT,
    filing_type    TEXT NOT NULL,
    filing_date    DATE NOT NULL,
    period_end     DATE,
    facts_status   TEXT,          -- ok | none (no XBRL) | failed | legacy
    chunks_status  TEXT,          -- ok | failed
    n_facts        INTEGER,
    n_chunks       INTEGER,
    error          TEXT,
    processed_at   TIMESTAMP,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_filings_accession ON filings(accession);

-- One row per (filing, concept, period, unit). Values carry their real XBRL period:
-- instants (balance sheet) have period_start NULL; durations carry both ends, so a
-- 3-month quarter and a 6-/9-month year-to-date figure ending on the same date are
-- distinct rows instead of colliding.
CREATE TABLE IF NOT EXISTS financial_facts (
    id               BIGSERIAL PRIMARY KEY,
    accession        TEXT NOT NULL REFERENCES filings(accession) ON DELETE CASCADE,
    ticker           TEXT NOT NULL,
    filing_type      TEXT NOT NULL,
    filing_date      DATE NOT NULL,
    statement_type   TEXT NOT NULL,   -- balance_sheet | income_statement | cashflow | segment | leases
    concept          TEXT NOT NULL,   -- e.g. us-gaap_PaymentsToAcquirePropertyPlantAndEquipment
    label            TEXT NOT NULL,   -- company's own line-item label (uniform labels for segment/leases)
    standard_concept TEXT,
    segment          TEXT,            -- business segment (e.g. AWS); NULL = consolidated company
    value            NUMERIC NOT NULL,
    unit             TEXT NOT NULL,   -- USD_millions | USD_per_share | shares | pure | <other measure>
    period_type      TEXT NOT NULL,   -- instant | duration
    period_start     DATE,
    period_end       DATE NOT NULL,
    duration_days    INTEGER,
    fiscal_period    TEXT NOT NULL,   -- instant | Q | 6M | 9M | FY | other
    fiscal_year      INTEGER,
    fiscal_quarter   INTEGER,         -- quarter (Q) or quarter-through (6M/9M/instant)
    is_derived       BOOLEAN NOT NULL DEFAULT FALSE,  -- quarter computed as YTD difference
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_dedupe ON financial_facts
    (accession, statement_type, concept, unit, segment, period_start, period_end, is_derived) NULLS NOT DISTINCT;
CREATE INDEX IF NOT EXISTS idx_facts_lookup ON financial_facts (ticker, concept, period_end);

-- Computed per company after ingest (facts.build_series): maps each reported
-- statement line item (concept + label) to its series concept, merging tags a
-- company changed between filings, and to a display label that tells apart
-- distinct items sharing a label (basic / diluted EPS).
CREATE TABLE IF NOT EXISTS fact_series (
    ticker          TEXT NOT NULL,
    statement_type  TEXT NOT NULL,
    concept         TEXT NOT NULL,
    label           TEXT NOT NULL,
    series_concept  TEXT NOT NULL,
    series_label    TEXT NOT NULL,
    PRIMARY KEY (ticker, statement_type, concept, label)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id           SERIAL PRIMARY KEY,
    accession    TEXT,
    ticker       TEXT NOT NULL,
    filing_date  DATE NOT NULL,
    filing_type  TEXT NOT NULL,
    section      TEXT,
    chunk_index  INTEGER,
    chunk_text   TEXT NOT NULL,
    embedding    vector,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedupe ON document_chunks (accession, section, chunk_index);
"""

# Latest value per (company, series, unit, segment, exact period): reported beats
# derived, then the most recent filing wins (so restatements/recasts supersede originals).
# `concept` is the series concept from fact_series (one per line item even where the
# company re-tagged it) and `label` adds a qualifier where distinct items share a label;
# the as-filed values are reported_concept / reported_label.
# `period_date` is kept as an alias of period_end for older queries.
CLEAN_VIEW = """
CREATE OR REPLACE VIEW financial_facts_clean AS
WITH f AS (
    SELECT f.*, f.concept AS reported_concept, f.label AS reported_label,
           COALESCE(s.series_concept, f.concept) AS series_concept,
           COALESCE(s.series_label, f.label) AS series_label
    FROM financial_facts f
    LEFT JOIN fact_series s
      ON s.ticker = f.ticker AND s.statement_type = f.statement_type AND s.concept = f.concept
     AND s.label = f.label
)
SELECT DISTINCT ON (ticker, series_concept, unit, segment, period_start, period_end)
    ticker, statement_type, series_concept AS concept, series_label AS label, standard_concept, value, unit,
    period_type, period_start, period_end, period_end AS period_date, duration_days,
    fiscal_period, fiscal_year, fiscal_quarter, is_derived,
    filing_type, filing_date, accession, segment, reported_concept, reported_label
FROM f
ORDER BY ticker, series_concept, unit, segment, period_start, period_end,
         is_derived ASC,
         filing_date DESC,
         CASE statement_type WHEN 'income_statement' THEN 0 WHEN 'balance_sheet' THEN 1 ELSE 2 END,
         accession DESC;
"""


def _migrate_legacy(conn):
    """Upgrade the pre-refactor schema in place. Safe to run repeatedly."""
    # filings: accession becomes the key; the old (ticker, type, date) key is dropped.
    if _table_exists(conn, "filings"):
        for col, typ in [("cik", "BIGINT"), ("facts_status", "TEXT"), ("chunks_status", "TEXT"),
                         ("n_facts", "INTEGER"), ("n_chunks", "INTEGER"), ("error", "TEXT"),
                         ("processed_at", "TIMESTAMP")]:
            conn.execute(text(f"ALTER TABLE filings ADD COLUMN IF NOT EXISTS {col} {typ}"))
        conn.execute(text("ALTER TABLE filings DROP CONSTRAINT IF EXISTS filings_ticker_filing_type_filing_date_key"))

    if _table_exists(conn, "companies"):
        for col, typ in [("cik", "BIGINT"), ("fiscal_year_end_month", "INTEGER")]:
            conn.execute(text(f"ALTER TABLE companies ADD COLUMN IF NOT EXISTS {col} {typ}"))

    # financial_facts without period_end is the old layout (single period_date, no duration).
    # Keep it as financial_facts_legacy for comparison; it can be dropped once verified.
    if _table_exists(conn, "financial_facts") and not _column_exists(conn, "financial_facts", "period_end"):
        print("  Migrating: financial_facts -> financial_facts_legacy (old layout had no period durations)")
        conn.execute(text("DROP VIEW IF EXISTS financial_facts_clean"))
        conn.execute(text("ALTER TABLE financial_facts RENAME TO financial_facts_legacy"))
        for old in ("idx_financial_facts_lookup", "idx_financial_facts_dedupe", "financial_facts_pkey"):
            conn.execute(text(f"ALTER INDEX IF EXISTS {old} RENAME TO {old.replace('financial_facts', 'financial_facts_legacy')}"))
        conn.execute(text("ALTER SEQUENCE IF EXISTS financial_facts_id_seq RENAME TO financial_facts_legacy_id_seq"))
        conn.execute(text("UPDATE filings SET facts_status = 'legacy' WHERE facts_status IS NULL"))

    # financial_facts.segment (2026-11): segment facts share concepts with consolidated ones,
    # so segment joins the dedupe key and the clean view's DISTINCT ON.
    if _table_exists(conn, "financial_facts") and not _column_exists(conn, "financial_facts", "segment"):
        print("  Migrating: financial_facts.segment")
        conn.execute(text("DROP VIEW IF EXISTS financial_facts_clean"))
        conn.execute(text("ALTER TABLE financial_facts ADD COLUMN segment TEXT"))
        conn.execute(text("DROP INDEX IF EXISTS idx_facts_dedupe"))  # recreated with segment by SCHEMA

    # document_chunks: link to filings by accession.
    if _table_exists(conn, "document_chunks"):
        if not _column_exists(conn, "document_chunks", "accession"):
            print("  Migrating: document_chunks.accession backfill")
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN accession TEXT"))
            conn.execute(text("""
                UPDATE document_chunks c SET accession = f.accession
                FROM filings f
                WHERE c.ticker = f.ticker AND c.filing_type = f.filing_type AND c.filing_date = f.filing_date
            """))
            conn.execute(text("""
                UPDATE filings f SET chunks_status = 'ok',
                       n_chunks = (SELECT count(*) FROM document_chunks c WHERE c.accession = f.accession)
                WHERE chunks_status IS NULL
            """))
        conn.execute(text("DROP INDEX IF EXISTS idx_document_chunks_dedupe"))
        # The old ivfflat index was created while the column was empty, so its clusters
        # were meaningless (the cause of missing results for D, papered over with probes=10).
        # HNSW needs no training and supports filtered iterative scans (pgvector >= 0.8).
        conn.execute(text("DROP INDEX IF EXISTS idx_chunks_embedding"))


def migrate():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _migrate_legacy(conn)
        conn.execute(text(SCHEMA))
        conn.execute(text(CLEAN_VIEW))
        for c in config.COMPANIES:
            conn.execute(text("""
                INSERT INTO companies (ticker, cik, name, sector)
                VALUES (:ticker, :cik, :name, :sector)
                ON CONFLICT (ticker) DO UPDATE
                SET cik = EXCLUDED.cik, name = EXCLUDED.name, sector = EXCLUDED.sector
            """), {"ticker": c.ticker, "cik": c.cik, "name": c.name, "sector": c.sector})


def ensure_vector_index(conn, dimension):
    """Pin the embedding column to `dimension` and make sure the HNSW index exists."""
    current = conn.execute(text("""
        SELECT format_type(atttypid, atttypmod) FROM pg_attribute
        WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'
    """)).scalar()
    wanted = f"vector({dimension})"
    if current != wanted:
        has_embeddings = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM document_chunks WHERE embedding IS NOT NULL)")).scalar()
        if has_embeddings:
            raise RuntimeError(
                f"document_chunks.embedding is {current} but the model produces {wanted}. "
                "Changing it would discard existing embeddings; run `python -m edgar_rag embed --reset` "
                "if that is intended.")
        conn.execute(text(f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE {wanted}"))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON document_chunks USING hnsw (embedding vector_cosine_ops)
    """))


def bulk_insert(conn, table, rows, columns, on_conflict="ON CONFLICT DO NOTHING"):
    """Fast multi-row insert through the underlying psycopg2 cursor. Returns rows inserted."""
    if not rows:
        return 0
    cursor = conn.connection.cursor()
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s {on_conflict} RETURNING 1"
    inserted = execute_values(cursor, sql, [tuple(r[c] for c in columns) for r in rows],
                              page_size=1000, fetch=True)
    return len(inserted)
