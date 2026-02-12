"""
EDGAR Filing Parser v2 - XBRL Edition
Uses edgartools XBRL API for clean financial statement extraction
and markdown parsing for narrative RAG chunks.
"""

import os
import re
from pathlib import Path
from datetime import datetime
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
                filing_type  TEXT NOT NULL,
                filing_date  DATE NOT NULL,
                period_end   DATE,
                accession    TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (ticker, filing_type, filing_date)
            );
        """))

        # Financial facts from XBRL
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_facts (
                id               SERIAL PRIMARY KEY,
                ticker           TEXT NOT NULL,
                filing_date      DATE NOT NULL,
                filing_type      TEXT NOT NULL,
                period_date      DATE NOT NULL,
                statement_type   TEXT NOT NULL,    -- balance_sheet, income_statement, cashflow
                concept          TEXT NOT NULL,    -- XBRL concept name (e.g. us-gaap_StockholdersEquity)
                label            TEXT NOT NULL,    -- Human readable label
                standard_concept TEXT,             -- Standardized concept name
                value            NUMERIC,
                unit             TEXT DEFAULT 'USD',
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Create index for common queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_financial_facts_lookup 
            ON financial_facts(ticker, period_date, concept);
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
                embedding    vector(1536),
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Create views for clean querying
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

def extract_statement_facts(statement_df, statement_type, ticker, filing_date, filing_type):
    """
    Extract financial facts from a statement dataframe.
    Returns list of dicts ready for database insertion.
    """
    # Filter to clean summary facts (no abstracts, no dimension breakdowns)
    clean = statement_df[
        (statement_df['abstract'] == False) & 
        (statement_df['dimension'] == False)
    ].copy()

    # Find period date columns
    period_cols = [c for c in statement_df.columns if re.match(r'\d{4}-\d{2}-\d{2}', str(c))]
    
    facts = []
    
    for _, row in clean.iterrows():
        concept = row['concept']
        label = row['label']
        standard_concept = row.get('standard_concept')
        
        # Extract value for each period
        for period_date in period_cols:
            value = row[period_date]
            
            # Skip None/NaN/empty values
            if value is None or value == '' or (isinstance(value, float) and value != value):
                continue
                
            # Convert from full dollars to millions
            value_millions = float(value) / 1_000_000 if value != 0 else 0
            
            facts.append({
                'ticker': ticker,
                'filing_date': filing_date,
                'filing_type': filing_type,
                'period_date': period_date,
                'statement_type': statement_type,
                'concept': concept,
                'label': label,
                'standard_concept': standard_concept,
                'value': value_millions,
                'unit': 'millions'
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
    
    # Balance Sheet
    try:
        bs = statements.balance_sheet()
        bs_df = bs.to_dataframe()
        facts = extract_statement_facts(bs_df, 'balance_sheet', ticker, filing_date, filing_type)
        all_facts.extend(facts)
        print(f"      ✓ Balance Sheet: {len(facts)} facts")
    except Exception as e:
        print(f"      ✗ Balance Sheet failed: {e}")
    
    # Income Statement
    try:
        inc = statements.income_statement()
        inc_df = inc.to_dataframe()
        facts = extract_statement_facts(inc_df, 'income_statement', ticker, filing_date, filing_type)
        all_facts.extend(facts)
        print(f"      ✓ Income Statement: {len(facts)} facts")
    except Exception as e:
        print(f"      ✗ Income Statement failed: {e}")
    
    # Cash Flow Statement
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

# Narrative sections we care about (not financial tables)
NARRATIVE_SECTIONS = [
    'business',
    'risk factors',
    'properties',
    'legal proceedings',
    "management's discussion",
    'md&a',
    'market risk',
]


def is_narrative_section(section_name):
    """Check if section contains narrative content vs. financial tables."""
    name_lower = section_name.lower()
    
    # Explicitly EXCLUDE only pure financial statement sections
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
    
    # Exclude cover page and signature sections
    if any(kw in name_lower for kw in ['cover page', 'signatures', 'exhibits']):
        return False
    
    # Include everything else - default to capturing content
    return True


def split_into_sections(markdown_text):
    """Split markdown document into sections based on headers."""
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
            # Keep last paragraph for overlap
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
        section_name = section['name']
        section_content = section['content']
        
        # Only process narrative sections
        if not is_narrative_section(section_name):
            continue
        
        # Skip very short sections
        if len(section_content) < 200:
            continue
        
        # Remove tables (pipe characters indicate markdown tables)
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
        
        # Chunk the narrative
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
                
                # Extract XBRL financial facts
                facts = extract_xbrl_financials(filing, ticker, filing_date, filing_type)
                
                # Extract narrative chunks
                chunks = extract_narrative_chunks(filing, ticker, filing_date, filing_type)
                print(f"      ✓ Narrative: {len(chunks)} chunks")
                
                # Save to database
                with engine.connect() as conn:
                    # Register filing
                    try:
                        conn.execute(text("""
                            INSERT INTO filings (ticker, filing_type, filing_date, accession)
                            VALUES (:ticker, :filing_type, :filing_date, :accession)
                            ON CONFLICT (ticker, filing_type, filing_date) DO NOTHING
                        """), {
                            'ticker': ticker,
                            'filing_type': filing_type,
                            'filing_date': filing_date,
                            'accession': filing.accession_number
                        })
                    except Exception as e:
                        print(f"      ⚠ Filing registration: {e}")
                    
                    # Save financial facts
                    for fact in facts:
                        try:
                            conn.execute(text("""
                                INSERT INTO financial_facts
                                    (ticker, filing_date, filing_type, period_date, statement_type,
                                     concept, label, standard_concept, value, unit)
                                VALUES
                                    (:ticker, :filing_date, :filing_type, :period_date, :statement_type,
                                     :concept, :label, :standard_concept, :value, :unit)
                            """), fact)
                        except Exception:
                            pass  # Skip duplicates
                    
                    # Save narrative chunks
                    for chunk in chunks:
                        try:
                            conn.execute(text("""
                                INSERT INTO document_chunks
                                    (ticker, filing_date, filing_type, section, chunk_index, chunk_text)
                                VALUES
                                    (:ticker, :filing_date, :filing_type, :section, :chunk_index, :chunk_text)
                            """), chunk)
                        except Exception:
                            pass  # Skip duplicates
                    
                    conn.commit()
                
                total_facts += len(facts)
                total_chunks += len(chunks)
                filings_processed += 1
                
                print(f"      ✓ Saved to database")
                
        except Exception as e:
            print(f"  ✗ Error processing {filing_type}: {e}")
            continue
    
    return {
        'ticker': ticker,
        'filings_processed': filings_processed,
        'total_facts': total_facts,
        'total_chunks': total_chunks
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
    print(f"EDGAR XBRL PARSER V2")
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
            print(f"  {r['ticker']:6} {r['filings_processed']:2} filings  "
                  f"{r['total_facts']:5} facts  {r['total_chunks']:5} chunks")


if __name__ == "__main__":
    run_parser()