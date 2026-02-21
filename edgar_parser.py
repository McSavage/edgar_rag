"""
EDGAR Filing Parser - XBRL Edition (Hardened)
Uses edgartools XBRL API for financial fact extraction
and markdown parsing for narrative RAG chunks.

Hardening updates:
1) Unit-aware value normalization (avoid blind scaling to millions)
2) Explicit dedupe keys + ON CONFLICT DO NOTHING inserts
"""

import os
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from edgar import Company, set_identity

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TICKERS = ["AMZN", "GOOGL", "META", "MSFT", "ORCL"]
START_DATE = "2023-01-01"
FILING_TYPES = ["10-K", "10-Q"]

# Set your SEC identity (required)
USER_NAME = "Daniel Savage"
USER_EMAIL = "dan.mcsavage@gmail.com"
set_identity(f"{USER_NAME} {USER_EMAIL}")


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DB")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")


def create_tables(engine):
    """Create database schema for financial facts and narrative chunks."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS companies (
                id          SERIAL PRIMARY KEY,
                ticker      TEXT UNIQUE NOT NULL,
                name        TEXT,
                sector      TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS filings (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                filing_type  TEXT NOT NULL,
                filing_date  DATE NOT NULL,
                period_end   DATE,
                accession    TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (ticker, filing_type, filing_date)
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_facts (
                id               SERIAL PRIMARY KEY,
                ticker           TEXT NOT NULL,
                filing_date      DATE NOT NULL,
                filing_type      TEXT NOT NULL,
                period_date      DATE NOT NULL,
                statement_type   TEXT NOT NULL,
                concept          TEXT NOT NULL,
                label            TEXT NOT NULL,
                standard_concept TEXT,
                value            NUMERIC,
                unit             TEXT DEFAULT 'USD',
                unit_confidence  TEXT DEFAULT 'unknown',
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            ALTER TABLE financial_facts
            ADD COLUMN IF NOT EXISTS unit_confidence TEXT DEFAULT 'unknown';
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_financial_facts_lookup
            ON financial_facts(ticker, period_date, concept);
        """))

        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_financial_facts_dedupe
            ON financial_facts(
                ticker, filing_type, filing_date, period_date, statement_type, concept, unit
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                filing_date  DATE NOT NULL,
                filing_type  TEXT NOT NULL,
                section      TEXT,
                chunk_index  INTEGER,
                chunk_text   TEXT NOT NULL,
                embedding    vector(1536),
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunks_dedupe
            ON document_chunks(
                ticker, filing_type, filing_date, section, chunk_index
            );
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW financial_facts_clean AS
            SELECT DISTINCT ON (ticker, period_date, concept)
                ticker,
                period_date,
                statement_type,
                concept,
                label,
                standard_concept,
                value,
                unit,
                filing_type,
                filing_date
            FROM financial_facts
            ORDER BY
                ticker,
                period_date,
                concept,
                CASE WHEN filing_type = '10-K' THEN 0 ELSE 1 END,
                filing_date DESC;
        """))

        conn.commit()
        print("✓ Database tables created successfully")


# ─────────────────────────────────────────────
# XBRL FINANCIAL STATEMENT EXTRACTION
# ─────────────────────────────────────────────

NON_MONETARY_LABEL_HINTS = (
    'per share', 'share', 'shares', 'weighted average', 'ratio', 'percent', '%',
    'employees', 'headcount', 'units'
)

NON_MONETARY_CONCEPT_HINTS = (
    'shares', 'share', 'pershare', 'per_share', 'ratio', 'percentage',
    'percent', 'weightedaverage', 'weighted_average', 'headcount'
)

DATE_RE = re.compile(r'(20\d{2}-\d{2}-\d{2})')
MONETARY_UNIT_TOKENS = ('usd', 'dollar', '$', 'million', 'billion', 'thousand')
MAX_CHUNKS_PER_SECTION = 120

SECTION_CANONICAL_MAP = {
    'risks related to our product offerings': 'Risk Factors - Product Offerings',
    'sensitivity analysis': 'Sensitivity Analysis',
    'industry trends and opportunities': 'Industry Trends and Opportunities',
    'commitments and contingencies': 'Commitments and Contingencies',
    'legal proceedings': 'Legal Proceedings',
    'note about forward-looking statements': 'Forward-Looking Statements',
}

SECTION_PATTERN_RULES = [
    ('incorporated herein by reference', 'Incorporation by Reference'),
    ('the information required by this item', 'Incorporation by Reference'),
    ('in addition, our business may be adversely affected if', 'Risk Factors - Additional Risks'),
    ('our international operations expose us to a number of risks', 'Risk Factors - International Operations'),
    ('risks relating to the evolution of our business', 'Risk Factors - Business Evolution'),
    ('operational risks', 'Risk Factors - Operational Risks'),
    ('legal and regulatory risks', 'Risk Factors - Legal and Regulatory'),
    ('strategic and competitive risks', 'Risk Factors - Strategic and Competitive'),
    ('general risks', 'Risk Factors - General'),
    ('critical accounting estimates', 'Critical Accounting Estimates'),
    ('liquidity and capital resources', 'Liquidity and Capital Resources'),
    ('business overview', 'Overview'),
    ('competition among platform-based ecosystems', 'Competition'),
]


def is_nan(value):
    return isinstance(value, float) and value != value


def infer_period_and_unit(column_key):
    """Infer period date and unit hint from dataframe column metadata."""
    if isinstance(column_key, (tuple, list)):
        date_value = None
        unit_parts = []

        for part in column_key:
            part_str = str(part)
            match = DATE_RE.search(part_str)
            if match and not date_value:
                date_value = match.group(1)
                continue
            if part_str.strip():
                unit_parts.append(part_str.strip())

        unit_hint = ' '.join(unit_parts).strip() or None
        return date_value, unit_hint

    col_text = str(column_key)
    match = DATE_RE.search(col_text)
    if not match:
        return None, None

    date_value = match.group(1)
    unit_hint = re.sub(DATE_RE, '', col_text).strip(' _|:-()[]{}') or None
    return date_value, unit_hint


def extract_period_specs(statement_df):
    """Return list of (column_key, period_date, unit_hint_from_column)."""
    period_specs = []
    for column_key in statement_df.columns:
        period_date, unit_hint = infer_period_and_unit(column_key)
        if period_date:
            period_specs.append((column_key, period_date, unit_hint))
    return period_specs


def detect_unit_hint(row, unit_hint_from_column=None):
    for col in ('unit', 'units', 'uom', 'measure', 'currency'):
        if col in row and row[col] not in (None, ''):
            return str(row[col])

    return unit_hint_from_column


def is_likely_non_monetary(concept, label, unit_hint):
    text_blob = f"{concept or ''} {label or ''} {unit_hint or ''}".lower()
    concept_blob = (concept or '').replace('-', '').replace('_', '').lower()
    return (
        any(hint in text_blob for hint in NON_MONETARY_LABEL_HINTS)
        or any(hint in concept_blob for hint in NON_MONETARY_CONCEPT_HINTS)
    )


def is_explicit_monetary(unit_hint):
    if not unit_hint:
        return False
    unit_text = unit_hint.lower()
    return any(token in unit_text for token in MONETARY_UNIT_TOKENS)


def normalize_value_and_unit(value, concept, label, unit_hint, statement_type):
    """
    Normalize values with unit awareness.

    Strategy:
    - Keep non-monetary metrics unscaled.
        - For clear monetary units:
      - already millions -> keep as-is
      - billions -> convert to millions
      - thousands -> convert to millions
      - dollars/usd -> convert to millions
        - If unit is unknown but this is a likely monetary statement fact,
            infer dollars and convert to millions.
    """
    numeric = float(value)
    unit_text = (unit_hint or '').lower()

    if not unit_hint:
        if statement_type in {'balance_sheet', 'income_statement', 'cashflow'} and not is_likely_non_monetary(concept, label, unit_hint):
            return numeric / 1_000_000, 'millions', 'inferred'
        return numeric, 'unknown', 'unknown'

    if is_likely_non_monetary(concept, label, unit_hint):
        return numeric, unit_hint, 'explicit'

    if not is_explicit_monetary(unit_hint):
        return numeric, unit_hint, 'explicit'

    if 'billion' in unit_text:
        return numeric * 1_000, 'millions', 'explicit'

    if 'million' in unit_text:
        return numeric, 'millions', 'explicit'

    if 'thousand' in unit_text:
        return numeric / 1_000, 'millions', 'explicit'

    if any(token in unit_text for token in ('usd', 'dollar', '$')):
        return numeric / 1_000_000, 'millions', 'explicit'

    return numeric, unit_hint, 'explicit'


def extract_statement_facts(statement_df, statement_type, ticker, filing_date, filing_type):
    """
    Extract financial facts from a statement dataframe.
    Returns list of dicts ready for database insertion.
    """
    clean = statement_df[~statement_df['abstract'] & ~statement_df['dimension']].copy()

    period_specs = extract_period_specs(statement_df)

    facts = []

    for _, row in clean.iterrows():
        concept = row['concept']
        label = row['label']
        standard_concept = row.get('standard_concept')
        
        for period_col, period_date, unit_hint_from_column in period_specs:
            value = row[period_col]
            unit_hint = detect_unit_hint(row, unit_hint_from_column=unit_hint_from_column)

            if value is None or value == '' or is_nan(value):
                continue

            try:
                normalized_value, normalized_unit, unit_confidence = normalize_value_and_unit(
                    value=value,
                    concept=concept,
                    label=label,
                    unit_hint=unit_hint,
                    statement_type=statement_type,
                )
            except (TypeError, ValueError):
                continue

            facts.append({
                'ticker': ticker,
                'filing_date': filing_date,
                'filing_type': filing_type,
                'period_date': period_date,
                'statement_type': statement_type,
                'concept': concept,
                'label': label,
                'standard_concept': standard_concept,
                'value': normalized_value,
                'unit': normalized_unit,
                'unit_confidence': unit_confidence,
            })

    return facts


def extract_xbrl_financials(filing, ticker, filing_date, filing_type):
    """Extract all financial statements from XBRL."""
    try:
        xbrl = filing.xbrl()
        statements = xbrl.statements
    except Exception as e:
        print(f"    ✗ Could not get XBRL statements: {e}")
        return []

    all_facts = []

    try:
        bs = statements.balance_sheet()
        bs_df = bs.to_dataframe()
        facts = extract_statement_facts(bs_df, 'balance_sheet', ticker, filing_date, filing_type)
        all_facts.extend(facts)
        print(f"      ✓ Balance Sheet: {len(facts)} facts")
    except Exception as e:
        print(f"      ✗ Balance Sheet failed: {e}")

    try:
        inc = statements.income_statement()
        inc_df = inc.to_dataframe()
        facts = extract_statement_facts(inc_df, 'income_statement', ticker, filing_date, filing_type)
        all_facts.extend(facts)
        print(f"      ✓ Income Statement: {len(facts)} facts")
    except Exception as e:
        print(f"      ✗ Income Statement failed: {e}")

    try:
        cf = statements.cashflow_statement()
        cf_df = cf.to_dataframe()
        facts = extract_statement_facts(cf_df, 'cashflow', ticker, filing_date, filing_type)
        all_facts.extend(facts)
        print(f"      ✓ Cash Flow: {len(facts)} facts")
    except Exception as e:
        print(f"      ✗ Cash Flow failed: {e}")

    return all_facts


# ─────────────────────────────────────────────
# NARRATIVE TEXT EXTRACTION FOR RAG
# ─────────────────────────────────────────────

def is_narrative_section(section_name):
    """Check if section contains narrative content vs. financial tables."""
    name_lower = section_name.lower()

    exclude_keywords = [
        'balance sheet', 'statement of financial position',
        'income statement', 'statement of operations', 'statement of earnings',
        'cash flow', 'statement of cash flows',
        'stockholders equity', 'shareholders equity', 'statement of equity',
        'statement of comprehensive income',
        'financial statements', 'consolidated statements',
        'table of contents', 'index to', 'index of'
    ]

    if any(kw in name_lower for kw in exclude_keywords):
        return False

    if any(kw in name_lower for kw in ['cover page', 'signatures', 'exhibits', 'header']):
        return False

    return True


def normalize_section_name(section_name):
    """Normalize verbose/sentence-like headings into stable section buckets."""
    if not section_name:
        return 'UNKNOWN'

    raw = re.sub(r'\s+', ' ', section_name.strip())
    lower = raw.lower()

    if lower in SECTION_CANONICAL_MAP:
        return SECTION_CANONICAL_MAP[lower]

    for key, canonical in SECTION_CANONICAL_MAP.items():
        if key in lower:
            return canonical

    for pattern, canonical in SECTION_PATTERN_RULES:
        if pattern in lower:
            return canonical

    if lower.startswith('highlights from the '):
        return 'Highlights'

    if lower.startswith('risks related to '):
        tail = raw[len('Risks Related To '):].strip(' :.-')
        return f"Risk Factors - {tail}" if tail else 'Risk Factors'

    if lower.startswith('note ') and ':' in raw:
        note_title = raw.split(':', 1)[0].strip()
        return note_title

    if len(raw) > 120 or raw.startswith(('•', '-', '*')):
        if 'risk' in lower:
            return 'Risk Factors'
        if 'highlight' in lower:
            return 'Highlights'
        if 'competition' in lower:
            return 'Competition'
        if 'sensitivity' in lower:
            return 'Sensitivity Analysis'
        if 'commitment' in lower or 'contingenc' in lower:
            return 'Commitments and Contingencies'
        return raw[:80].rstrip(' .:;-')

    if raw.isupper() and any(c.isalpha() for c in raw):
        return raw.title()

    return raw


def split_into_sections(markdown_text):
    """Split markdown document into sections based on headers."""
    sections = []
    current_section = {'name': 'HEADER', 'content': []}

    def is_noisy_header(header_text):
        candidate = header_text.strip()
        if not candidate:
            return True
        if candidate.startswith(('•', '-', '*')):
            return True
        if len(candidate) > 180:
            return True
        if sum(ch.isalpha() for ch in candidate) < 3:
            return True
        return False

    for line in markdown_text.split('\n'):
        header_match = re.match(r'^#{1,4}\s+(.+)$', line)
        if header_match:
            header_text = header_match.group(1).strip()

            if is_noisy_header(header_text):
                current_section['content'].append(line)
                continue

            if current_section['content']:
                sections.append({
                    'name': current_section['name'],
                    'content': '\n'.join(current_section['content'])
                })

            current_section = {'name': header_text, 'content': [line]}
        else:
            current_section['content'].append(line)

    if current_section['content']:
        sections.append({
            'name': current_section['name'],
            'content': '\n'.join(current_section['content'])
        })

    return sections


def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for embedding."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = current_chunk[-1:] if overlap > 0 else []
            current_length = len(current_chunk[0]) if current_chunk else 0

        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_narrative_chunks(filing, ticker, filing_date, filing_type):
    """Extract narrative text chunks from markdown for RAG."""
    try:
        markdown = filing.markdown()
    except Exception as e:
        print(f"    ✗ Could not get markdown: {e}")
        return []

    sections = split_into_sections(markdown)
    all_chunks = []

    for section in sections:
        section_name = normalize_section_name(section['name'])
        section_content = section['content']

        if not is_narrative_section(section_name):
            continue

        if len(section_content) < 200:
            continue

        lines = section_content.split('\n')
        narrative_lines = []
        in_table = False

        for line in lines:
            if '|' in line:
                in_table = True
            elif in_table:
                in_table = False
                narrative_lines.append('')

            if not in_table:
                narrative_lines.append(line)

        narrative = '\n'.join(narrative_lines).strip()

        if len(narrative) < 200:
            continue

        chunks = chunk_text(narrative)
        if len(chunks) > MAX_CHUNKS_PER_SECTION:
            chunks = chunks[:MAX_CHUNKS_PER_SECTION]

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'ticker': ticker,
                'filing_date': filing_date,
                'filing_type': filing_type,
                'section': section_name,
                'chunk_index': i,
                'chunk_text': chunk,
            })

    return all_chunks


# ─────────────────────────────────────────────
# MAIN PARSING PIPELINE
# ─────────────────────────────────────────────

def parse_company_filings(ticker, engine):
    """Parse all filings for a company."""
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print('='*60)

    try:
        company = Company(ticker)
    except Exception as e:
        print(f"✗ Could not fetch company: {e}")
        return {'ticker': ticker, 'error': str(e)}

    total_facts = 0
    total_chunks = 0
    filings_processed = 0

    for filing_type in FILING_TYPES:
        try:
            filings = company.get_filings(form=filing_type).filter(date=f"{START_DATE}:")
            print(f"\n  Found {len(filings)} {filing_type} filings")

            for filing in filings:
                filing_date = filing.filing_date
                print(f"\n  📄 {filing_type} filed {filing_date}")

                facts = extract_xbrl_financials(filing, ticker, filing_date, filing_type)
                chunks = extract_narrative_chunks(filing, ticker, filing_date, filing_type)
                print(f"      ✓ Narrative: {len(chunks)} chunks")

                saved_facts = 0
                saved_chunks = 0

                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO filings (ticker, filing_type, filing_date, accession)
                        VALUES (:ticker, :filing_type, :filing_date, :accession)
                        ON CONFLICT (ticker, filing_type, filing_date) DO NOTHING
                    """), {
                        'ticker': ticker,
                        'filing_type': filing_type,
                        'filing_date': filing_date,
                        'accession': filing.accession_number,
                    })

                    for fact in facts:
                        result = conn.execute(text("""
                            INSERT INTO financial_facts
                                (ticker, filing_date, filing_type, period_date, statement_type,
                                 concept, label, standard_concept, value, unit, unit_confidence)
                            VALUES
                                (:ticker, :filing_date, :filing_type, :period_date, :statement_type,
                                 :concept, :label, :standard_concept, :value, :unit, :unit_confidence)
                            ON CONFLICT DO NOTHING
                        """), fact)
                        saved_facts += result.rowcount or 0

                    for chunk in chunks:
                        result = conn.execute(text("""
                            INSERT INTO document_chunks
                                (ticker, filing_date, filing_type, section, chunk_index, chunk_text)
                            VALUES
                                (:ticker, :filing_date, :filing_type, :section, :chunk_index, :chunk_text)
                            ON CONFLICT DO NOTHING
                        """), chunk)
                        saved_chunks += result.rowcount or 0

                    conn.commit()

                total_facts += saved_facts
                total_chunks += saved_chunks
                filings_processed += 1

                print(f"      ✓ Saved: {saved_facts} facts, {saved_chunks} chunks")

        except Exception as e:
            print(f"  ✗ Error processing {filing_type}: {e}")
            continue

    return {
        'ticker': ticker,
        'filings_processed': filings_processed,
        'total_facts': total_facts,
        'total_chunks': total_chunks,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run_parser():
    """Process all companies and their filings."""
    engine = get_engine()

    print("Setting up database tables...")
    create_tables(engine)

    print(f"\n{'='*60}")
    print("EDGAR XBRL PARSER (HARDENED)")
    print('='*60)
    print(f"Companies: {', '.join(TICKERS)}")
    print(f"Filing types: {', '.join(FILING_TYPES)}")
    print(f"Date range: {START_DATE} to present")
    print('='*60)

    results = []
    for ticker in TICKERS:
        result = parse_company_filings(ticker, engine)
        results.append(result)

    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)

    total_filings = sum(r.get('filings_processed', 0) for r in results)
    total_facts = sum(r.get('total_facts', 0) for r in results)
    total_chunks = sum(r.get('total_chunks', 0) for r in results)

    print(f"  Total filings processed: {total_filings}")
    print(f"  Total financial facts:   {total_facts}")
    print(f"  Total narrative chunks:  {total_chunks}")

    print("\n" + "=" * 60)
    print("SUMMARY BY COMPANY")
    print("=" * 60)
    for r in results:
        if 'error' in r:
            print(f"  {r['ticker']:6} ✗ {r['error']}")
        else:
            print(
                f"  {r['ticker']:6} {r['filings_processed']:2} filings  "
                f"{r['total_facts']:5} facts  {r['total_chunks']:5} chunks"
            )


if __name__ == "__main__":
    run_parser()
