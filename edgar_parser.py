"""
EDGAR Filing Parser
Parses markdown filings into:
1. Financial facts (structured table data) -> PostgreSQL
2. Narrative chunks (text sections) -> pgvector embeddings
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

load_dotenv()

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
        # Companies reference table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS companies (
                id          SERIAL PRIMARY KEY,
                ticker      TEXT UNIQUE NOT NULL,
                name        TEXT,
                sector      TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Filings metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS filings (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                filing_type  TEXT NOT NULL,   -- 10-K or 10-Q
                filing_date  DATE NOT NULL,
                file_path    TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (ticker, filing_type, filing_date)
            );
        """))

        # Structured financial facts from tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_facts (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                filing_date  DATE NOT NULL,
                filing_type  TEXT NOT NULL,
                period_date  DATE NOT NULL,   -- The date the value refers to
                metric       TEXT NOT NULL,   -- e.g. "Total stockholders equity"
                value        NUMERIC,
                unit         TEXT DEFAULT 'millions',
                audited      BOOLEAN DEFAULT TRUE,
                section      TEXT,            -- e.g. "CONSOLIDATED BALANCE SHEETS"
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Narrative text chunks for RAG
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                filing_date  DATE NOT NULL,
                filing_type  TEXT NOT NULL,
                section      TEXT,
                chunk_index  INTEGER,
                chunk_text   TEXT NOT NULL,
                embedding    vector(1536),    -- OpenAI/Voyage embedding size
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.commit()
        print("✓ Database tables created successfully")


# ─────────────────────────────────────────────
# PARSING UTILITIES
# ─────────────────────────────────────────────

def parse_filename(filepath: Path) -> tuple[str, str, str]:
    """
    Extract ticker, filing_type, filing_date from filepath.
    e.g. data/filings/GOOGL/10K_2026-02-05.md -> (GOOGL, 10-K, 2026-02-05)
    """
    ticker = filepath.parent.name.upper()
    stem = filepath.stem  # e.g. 10K_2026-02-05

    if stem.startswith("10K"):
        filing_type = "10-K"
    elif stem.startswith("10Q"):
        filing_type = "10-Q"
    else:
        filing_type = "UNKNOWN"

    # Extract date portion
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', stem)
    filing_date = date_match.group(1) if date_match else None

    return ticker, filing_type, filing_date


def clean_metric_name(raw: str) -> str:
    """Clean up metric names from table cells."""
    # Remove HTML artifacts
    text = re.sub(r'<[^>]+>', '', raw)
    # Collapse whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove common noise characters
    text = text.strip(':|$')
    text = text.strip()
    return text


def parse_value(raw: str) -> float | None:
    """Parse numeric value from a table cell."""
    if not raw:
        return None
    # Remove HTML, whitespace, dollar signs, commas
    cleaned = re.sub(r'<[^>]+>', '', raw)
    cleaned = re.sub(r'[\s,$]', '', cleaned)
    # Handle negative numbers formatted as (1,234)
    cleaned = re.sub(r'\((\d+)\)', r'-\1', cleaned)
    # Remove dashes that mean zero
    if cleaned in ['-', '—', '–', '']:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_period_header(header_text: str) -> tuple[str | None, bool]:
    """
    Extract a date from a column header and whether it's audited.
    e.g. 'As of December 31, 2025' -> ('2025-12-31', True)
    e.g. 'As of September 30, 2025 -unaudited' -> ('2025-09-30', False)
    """
    audited = 'unaudited' not in header_text.lower()

    # Month name patterns
    month_map = {
        'january': '01', 'february': '02', 'march': '03',
        'april': '04', 'may': '05', 'june': '06',
        'july': '07', 'august': '08', 'september': '09',
        'october': '10', 'november': '11', 'december': '12'
    }

    # Match "Month DD, YYYY"
    pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})'
    match = re.search(pattern, header_text.lower())
    if match:
        month = month_map[match.group(1)]
        day = match.group(2).zfill(2)
        year = match.group(3)
        return f"{year}-{month}-{day}", audited

    # Match just year e.g. "2025" as fallback - treat as Dec 31
    year_match = re.search(r'\b(20\d{2})\b', header_text)
    if year_match:
        return f"{year_match.group(1)}-12-31", audited

    return None, audited


# ─────────────────────────────────────────────
# TABLE PARSER
# ─────────────────────────────────────────────

def parse_markdown_table(table_text: str, section_name: str, ticker: str,
                          filing_date: str, filing_type: str) -> list[dict]:
    """
    Parse a markdown pipe table into financial fact rows.
    Returns list of dicts ready for DB insertion.
    """
    rows = [line for line in table_text.strip().split('\n') if '|' in line]
    if len(rows) < 2:
        return []

    facts = []

    # ── Find header row and extract period dates ──────────────────────────────
    # Header is typically the first row; separator row has dashes
    header_row = rows[0]
    header_cells = [c.strip() for c in header_row.split('|') if c.strip()]

    # Combine header text to find date columns
    # Header may span multiple rows for complex headers - join them
    combined_header = ' '.join([
        ' '.join([c.strip() for c in row.split('|') if c.strip()])
        for row in rows[:3]  # Check first 3 rows for header info
    ])

    # Find period dates from header
    period_dates = []
    # Look for date patterns in header cells
    for cell in header_cells:
        period, audited = parse_period_header(cell)
        if period:
            period_dates.append((period, audited))

    # If we couldn't find dates in individual cells, try combined header
    if not period_dates:
        # Find all dates in combined header
        month_pattern = r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})'
        date_mentions = re.findall(month_pattern, combined_header.lower())
        for dm in date_mentions:
            period, audited = parse_period_header(dm)
            if period:
                period_dates.append((period, audited))

    if not period_dates:
        return []  # Can't parse without period dates

    # ── Skip header and separator rows, parse data rows ──────────────────────
    data_rows = []
    for row in rows:
        # Skip separator rows (contain ---)
        if re.match(r'^\|[-:\s|]+\|$', row):
            continue
        # Skip rows that are clearly headers (no numeric data)
        cells = [c.strip() for c in row.split('|')]
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells if c.strip()]
        if cells:
            data_rows.append(cells)

    # Skip first 1-3 rows (headers)
    # Find where numeric data starts
    data_start = 0
    for i, row_cells in enumerate(data_rows):
        # Check if any cell looks like a number
        has_number = any(parse_value(c) is not None for c in row_cells)
        if has_number:
            data_start = i
            break

    # ── Extract metric/value pairs ────────────────────────────────────────────
    for row_cells in data_rows[data_start:]:
        if not row_cells:
            continue

        # First cell is the metric name
        metric = clean_metric_name(row_cells[0])
        if not metric or len(metric) < 3:
            continue

        # Skip rows that are section headers (no numeric values)
        numeric_cells = [parse_value(c) for c in row_cells[1:] if c and c not in ['$', '']]
        numeric_values = [v for v in numeric_cells if v is not None]

        if not numeric_values:
            continue

        # Match values to period dates
        # Filter out empty/dollar-sign cells to get actual values
        value_cells = [c for c in row_cells[1:] if c and c not in ['$', '', '|']]
        values = [parse_value(c) for c in value_cells]
        values = [v for v in values if v is not None]

        for i, (period_date, audited) in enumerate(period_dates):
            if i < len(values):
                facts.append({
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'filing_type': filing_type,
                    'period_date': period_date,
                    'metric': metric,
                    'value': values[i],
                    'audited': audited,
                    'section': section_name
                })

    return facts


# ─────────────────────────────────────────────
# SECTION / NARRATIVE PARSER
# ─────────────────────────────────────────────

# SEC filing sections we care about for narrative RAG
NARRATIVE_SECTIONS = {
    'item 1':  'Business',
    'item 1a': 'Risk Factors',
    'item 2':  'Properties',
    'item 7':  'MD&A',
    'item 7a': 'Market Risk',
}

TABLE_SECTION_KEYWORDS = [
    'balance sheet', 'income statement', 'statement of operations',
    'cash flow', 'stockholders equity', 'shareholders equity',
    'financial statements', 'financial position'
]


def split_into_sections(markdown_text: str) -> list[dict]:
    """
    Split markdown document into sections based on headers.
    Returns list of {name, content} dicts.
    """
    sections = []
    current_section = {'name': 'HEADER', 'content': []}

    for line in markdown_text.split('\n'):
        # Check for section headers (##, ###, ####)
        header_match = re.match(r'^#{1,4}\s+(.+)$', line)
        if header_match:
            header_text = header_match.group(1).strip()

            # Save current section if it has content
            if current_section['content']:
                sections.append({
                    'name': current_section['name'],
                    'content': '\n'.join(current_section['content'])
                })

            current_section = {'name': header_text, 'content': [line]}
        else:
            current_section['content'].append(line)

    # Don't forget the last section
    if current_section['content']:
        sections.append({
            'name': current_section['name'],
            'content': '\n'.join(current_section['content'])
        })

    return sections


def is_table_section(section_name: str) -> bool:
    """Determine if a section is primarily financial tables."""
    name_lower = section_name.lower()
    return any(kw in name_lower for kw in TABLE_SECTION_KEYWORDS)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    Tries to split on paragraph boundaries.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            # Keep last paragraph for overlap
            current_chunk = current_chunk[-1:] if overlap > 0 else []
            current_length = len(current_chunk[0]) if current_chunk else 0

        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_tables_from_section(content: str) -> list[str]:
    """Extract markdown tables from section content."""
    tables = []
    lines = content.split('\n')
    current_table = []
    in_table = False

    for line in lines:
        if '|' in line:
            in_table = True
            current_table.append(line)
        else:
            if in_table and current_table:
                tables.append('\n'.join(current_table))
                current_table = []
                in_table = False

    if current_table:
        tables.append('\n'.join(current_table))

    return tables


def extract_narrative_from_section(content: str) -> str:
    """Remove tables from section content, keep narrative text."""
    lines = content.split('\n')
    narrative_lines = []
    in_table = False

    for line in lines:
        if '|' in line:
            in_table = True
        elif in_table:
            in_table = False
            # Add a blank line after table
            narrative_lines.append('')
        
        if not in_table:
            narrative_lines.append(line)

    return '\n'.join(narrative_lines).strip()


# ─────────────────────────────────────────────
# MAIN PARSING PIPELINE
# ─────────────────────────────────────────────

def parse_filing(filepath: Path, engine) -> dict:
    """
    Parse a single markdown filing.
    Extracts financial facts and narrative chunks.
    Returns summary of what was extracted.
    """
    ticker, filing_type, filing_date = parse_filename(filepath)

    if not filing_date:
        print(f"  ✗ Could not parse date from {filepath.name}")
        return {}

    print(f"\n  Processing {ticker} {filing_type} {filing_date}...")

    content = filepath.read_text(encoding='utf-8', errors='ignore')
    sections = split_into_sections(content)

    all_facts = []
    all_chunks = []

    for section in sections:
        section_name = section['name']
        section_content = section['content']

        # ── Extract financial facts from tables ────────────────────────────
        tables = extract_tables_from_section(section_content)
        for table in tables:
            facts = parse_markdown_table(
                table, section_name, ticker, filing_date, filing_type
            )
            all_facts.extend(facts)

        # ── Extract narrative chunks for RAG ──────────────────────────────
        # Only chunk sections that are narrative (not pure financial tables)
        if not is_table_section(section_name):
            narrative = extract_narrative_from_section(section_content)
            if len(narrative) > 100:  # Skip very short sections
                chunks = chunk_text(narrative)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'ticker': ticker,
                        'filing_date': filing_date,
                        'filing_type': filing_type,
                        'section': section_name,
                        'chunk_index': i,
                        'chunk_text': chunk,
                    })

    # ── Save to database ─────────────────────────────────────────────────────
    facts_saved = 0
    chunks_saved = 0

    with engine.connect() as conn:
        # Register filing
        conn.execute(text("""
            INSERT INTO filings (ticker, filing_type, filing_date, file_path)
            VALUES (:ticker, :filing_type, :filing_date, :file_path)
            ON CONFLICT (ticker, filing_type, filing_date) DO NOTHING
        """), {
            'ticker': ticker,
            'filing_type': filing_type,
            'filing_date': filing_date,
            'file_path': str(filepath)
        })

        # Save financial facts
        for fact in all_facts:
            try:
                conn.execute(text("""
                    INSERT INTO financial_facts
                        (ticker, filing_date, filing_type, period_date, metric, value, audited, section)
                    VALUES
                        (:ticker, :filing_date, :filing_type, :period_date, :metric, :value, :audited, :section)
                """), fact)
                facts_saved += 1
            except Exception:
                pass  # Skip duplicates or bad data

        # Save narrative chunks (embeddings added later)
        for chunk in all_chunks:
            try:
                conn.execute(text("""
                    INSERT INTO document_chunks
                        (ticker, filing_date, filing_type, section, chunk_index, chunk_text)
                    VALUES
                        (:ticker, :filing_date, :filing_type, :section, :chunk_index, :chunk_text)
                """), chunk)
                chunks_saved += 1
            except Exception:
                pass

        conn.commit()

    return {
        'ticker': ticker,
        'filing_type': filing_type,
        'filing_date': filing_date,
        'facts_saved': facts_saved,
        'chunks_saved': chunks_saved
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run_parser(data_dir: str = "data/filings"):
    """Process all markdown filings in the data directory."""
    engine = get_engine()

    print("Setting up database tables...")
    create_tables(engine)

    filing_path = Path(data_dir)
    md_files = list(filing_path.rglob("*.md"))

    print(f"\nFound {len(md_files)} filing(s) to process")
    print("=" * 60)

    results = []
    for filepath in sorted(md_files):
        result = parse_filing(filepath, engine)
        if result:
            results.append(result)
            print(f"    ✓ Facts: {result['facts_saved']}  Chunks: {result['chunks_saved']}")

    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)

    total_facts = sum(r['facts_saved'] for r in results)
    total_chunks = sum(r['chunks_saved'] for r in results)
    print(f"  Total filings processed: {len(results)}")
    print(f"  Total financial facts:   {total_facts}")
    print(f"  Total narrative chunks:  {total_chunks}")
    print(f"\nNext step: Generate embeddings for {total_chunks} chunks")


if __name__ == "__main__":
    run_parser()