# EDGAR Financial RAG System

A hybrid Retrieval-Augmented Generation (RAG) system for analyzing SEC filings using natural language queries. The system combines structured financial data (XBRL) with narrative content embeddings to provide comprehensive answers about company financials, risks, strategies, and business context.

## 🎯 Overview

This is a work in progress. The original intent of this project was to support my interest in calculating LCOC - Levelised Cost Of Compute - $/PFLOP-hours for a datacenter. I wasn't optimistic that SEC filings would provide a lot of useful information for that, and this data seems to confirm it. You can see the quality of the results in the `edgar_rag_interface` notebook. I expect to iterate on this a few more times. I thought the exercise of building the data pipeline and infrastructure would be useful in itself.

This system provides a natural language interface to analyze SEC filings (10-K and 10-Q reports) for major tech companies, a regulated utility, an AI cloud provider, and a launch/satellite company. It intelligently routes queries to either structured financial databases or vector-based semantic search depending on the question type, then synthesizes coherent answers using Claude.

**Current Data Coverage:**

- **Companies:** Amazon (AMZN), Alphabet (GOOGL), Meta (META), Microsoft (MSFT), Oracle (ORCL), Dominion Energy (D), CoreWeave (CRWV), SpaceX (SPCX)
- **Time Period:** January 2020 - Present (CRWV from its March 2025 IPO, SPCX from its June 2026 IPO)
- **Filings:** 176 filings (10-K annual reports, 10-Q quarterly reports, and SpaceX's 424B4 IPO prospectus) through October 2026

## ✨ Key Features

### 1. **Structured Financial Data (XBRL)**

- 45,741 reported financial facts (plus 4,765 derived quarterly values) from balance sheets, income statements, cash flow statements, segment disclosures and lease notes
- Every value carries its exact XBRL period (start and end date), classified as a fiscal quarter, fiscal year, year-to-date or balance-sheet date, with fiscal-year/quarter labels that respect non-calendar fiscal years (MSFT: June, ORCL: May)
- Quarters a filing doesn't report on its own (Q4, and cash-flow quarters, since 10-Qs only report cash flows year-to-date) are derived from year-to-date differences and flagged `is_derived`
- One series per line item even where a company changed the XBRL tag between filings (Meta's `fb_` → `meta_`, Dominion's capex tag), and distinct items that share a label are told apart ("Services (expense)" vs "Services (sales revenue net)", "(basic)" vs "(diluted)")
- **Business segments:** revenue, operating income, capex, D&A and assets per segment, e.g. AWS capex by quarter, or capex for Dominion Energy Virginia (the Data Center Alley utility)
- **Leases:** future payments on leases signed but not yet commenced (the forward data-center pipeline: Microsoft $364.6B, Oracle $288B, Meta $355.7B as of their latest 10-Qs), right-of-use assets obtained for new leases (leased capacity that the cash capex line misses), and CoreWeave's undelivered contracted power in MW
- SQL-queryable database for precise numerical queries with date ranges and filtering
- Best for: Revenue, expenses, equity, assets, and specific financial metrics

### 2. **Narrative Content Vector Search**

- 40,150 text chunks from risk factors, MD&A sections, and business descriptions
- Vector embeddings via Voyage AI for semantic similarity search
- PostgreSQL with pgvector (HNSW index with filtered iterative scans) for efficient retrieval
- Best for: Risks, strategies, qualitative discussions, and business context

### 3. **Intelligent Query Router**

- Claude classifies each query as quantitative, qualitative, or hybrid, and extracts companies, line-item keywords, quarterly vs. annual granularity and a date window (as structured output)
- Automatically retrieves from appropriate data sources, balanced per company
- Synthesizes answers combining numbers with narrative context

## 🏗️ Architecture

```text
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Router   │  ◄── Claude classifies query type
│  (Claude AI)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────────┐
│  SQL   │  │  Vector    │
│ Query  │  │  Search    │
│ (XBRL) │  │ (pgvector) │
└───┬────┘  └─────┬──────┘
    │             │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │   Response   │  ◄── Claude synthesizes final answer
    │  Generation  │
    └──────────────┘
```

## 📦 Project Structure

```text
edgar_rag/
├── edgar_rag/                 # the package
│   ├── config.py              # companies (by CIK), dates, models, identity — the one place to edit
│   ├── db.py                  # schema + idempotent migrations
│   ├── sec.py                 # EDGAR access via edgartools
│   ├── facts.py               # XBRL fact extraction, period classification, derived quarters
│   ├── narrative.py           # section splitting + chunking for RAG
│   ├── embeddings.py          # Voyage AI embeddings
│   ├── ingest.py              # incremental pipeline (update / rebuild-facts / status)
│   ├── query.py               # router, retrieval, synthesis, ask()
│   └── __main__.py            # CLI: python -m edgar_rag ...
├── edgar_rag_interface.ipynb  # interactive interface (thin client over edgar_rag.query)
└── requirements.txt
```

## 🚀 Setup

### Prerequisites

- Python 3.13+
- PostgreSQL 15+ with the pgvector extension (0.8+)
- API Keys:
  - Anthropic API key (for Claude)
  - Voyage AI API key (for embeddings)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd edgar_rag
   ```

2. **Create and activate virtual environment:**

   ```bash
   python -m venv power
   source power/bin/activate  # On Windows: power\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL:**

   ```sql
   CREATE EXTENSION vector;
   ```

5. **Configure environment variables:**

   Create a `.env` file in the project root:

   ```env
   # Database Configuration
   POSTGRES_USER=your_username
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=edgar_rag

   # API Keys
   ANTHROPIC_API_KEY=your_anthropic_key
   VOYAGE_API_KEY=your_voyage_key

   # SEC requires a "Name email" identity for EDGAR access
   SEC_IDENTITY="Your Name you@example.com"

   # Optional overrides
   # CLAUDE_MODEL=claude-sonnet-5-5
   # EMBEDDING_MODEL=voyage-3-lite
   ```

### Data Pipeline

One command creates the schema, finds filings that aren't in the database yet, extracts XBRL facts and narrative chunks, and embeds the new chunks:

```bash
python -m edgar_rag update                 # everything new, all companies
python -m edgar_rag update --tickers SPCX  # one company
python -m edgar_rag update --dry-run       # list what would be ingested
python -m edgar_rag status                 # per-company coverage
```

The pipeline is incremental by accession number, so a refresh after a few months only downloads the new filings. Filings that fail are recorded and retried on the next run. Other commands:

```bash
python -m edgar_rag rebuild-facts   # re-extract XBRL facts for ingested filings (after changing facts.py)
python -m edgar_rag rebuild-series  # recompute series mappings and derived quarters from stored facts (no downloads)
python -m edgar_rag embed           # embed any chunks still missing embeddings
python -m edgar_rag ask "What was Microsoft's revenue in fiscal Q2 2025?"
```

## 💡 Usage

### Interactive Interface

Open the main interface notebook:

```bash
jupyter notebook edgar_rag_interface.ipynb
```

### Simple Query Interface

```python
# Ask any question about the companies
ask("How much did Amazon spend on infrastructure in 2025?")
ask("What risks does Microsoft mention about AI and power?")
ask("Compare data center spending across the hyperscalers")
```

Or from the shell: `python -m edgar_rag ask "..."`.

### Example Use Cases

**Quantitative Analysis:**

```python
ask("What was Microsoft's total revenue in Q2 2025?")
ask("Compare operating expenses between Meta and Alphabet in 2024")
ask("Show me Amazon's cash flow trends over the past 8 quarters")
ask("How much did Amazon spend on purchases of property and equipment each quarter of 2025?")
ask("What was AWS capex each quarter since 2024?")
ask("How have leases not yet commenced grown at Microsoft, Meta and Oracle over the last two years?")
ask("What were Dominion Energy Virginia's capital expenditures by year?")
```

**Qualitative Analysis:**

```python
ask("What are the main risks Oracle faces in cloud infrastructure?")
ask("How does Meta describe its AI strategy?")
ask("What regulatory concerns are mentioned by these companies?")
ask("How does Dominion Energy describe data center demand growth in its filings?")
ask("What does SpaceX say about Starlink and its capital expenditure plans?")
```

**Hybrid Analysis:**

```python
ask("Analyze Amazon's infrastructure investments and related risk factors")
ask("Compare AI spending and strategic priorities across companies")
ask("What companies mention energy shortages and how much are they investing?")
ask("How is Dominion Energy's generation mix changing alongside hyperscaler capex growth?")
```

## 🔧 Configuration

### Models

Set in `.env` (defaults in `edgar_rag/config.py`):

```env
CLAUDE_MODEL=claude-sonnet-5-5   # query routing + answer synthesis
EMBEDDING_MODEL=voyage-3-lite    # 512 dimensions; voyage-3 = 1024 dimensions
```

Changing the embedding model changes the vector dimension; run `python -m edgar_rag embed --reset` to re-embed everything.

### Companies and Date Range

Edit `COMPANIES` and `START_DATE` in `edgar_rag/config.py`. The query router's company list is generated from it, so this is the only place to change:

```python
CompanySpec("SPCX", 1181412, "SpaceX", "Space launch and satellite connectivity",
            ("Space Exploration Technologies", "Starlink"), extra_forms=("424B4",)),
```

Companies are looked up by CIK, not ticker. Tickers get reused: `Company("SPCX")` in edgartools resolves to an unrelated ETF trust. Then run `python -m edgar_rag update`.

## 📊 Database Schema

### `financial_facts`

Structured financial data from XBRL, one row per filing × concept × period × unit:

- `statement_type`: `balance_sheet`, `income_statement`, `cashflow`, `segment` or `leases`
- `concept`, `label` (the company's own line-item label), `value`, `unit` (`USD_millions`, `USD_per_share`, `shares`, `pure`, `MW`)
- `segment`: business segment name (e.g. `AWS`); NULL for company totals
- `period_start` / `period_end` / `duration_days`, `fiscal_period` (`Q`, `6M`, `9M`, `FY`, `TTM`, `instant`), `fiscal_year`, `fiscal_quarter`
- `is_derived` for quarters computed from year-to-date differences
- `financial_facts_clean` (view): latest value per company × series × segment × period, so restatements win. `concept` and `label` are the series concept and display label from `fact_series`; the as-filed values are in `reported_concept` / `reported_label`

### `fact_series`

Recomputed per company after each ingest (`facts.build_series`). It maps each reported statement line item (concept + label) to:

- `series_concept`: companies re-tag line items between filings, and the next filing's comparative column then repeats an old period under the new tag. Two concepts are merged when they were used with the same label, never report the same period in one filing, and report the same value for at least one period. The merged series takes the most recent concept.
- `series_label`: the reported label, plus a qualifier taken from the concept name where distinct items in one statement share a label and unit (Oracle's "Services" revenue and expense, basic and diluted EPS).

Segment and lease facts use uniform labels across companies ("Segment capital expenditures", "Leases not yet commenced (future lease payments)") under synthetic `edgar_rag_*` concepts, because companies change the underlying XBRL tags between years. `standard_concept` records the tag each value was read from.

### `document_chunks`

Narrative content for vector search:

- Text chunks from SEC filings, linked to `filings` by accession
- Normalized section names (risk factors, MD&A, business, notes)
- Vector embeddings (512 or 1024 dimensions), HNSW index

### `filings`

One row per filing (keyed by accession), with per-filing ingestion status and counts

### `companies`

Reference table for company information (CIK, sector, fiscal year end), populated from `config.py`

## 🎓 Technical Details

- **Embeddings:** Voyage AI (voyage-3-lite with 512 dimensions)
- **Vector Search:** PostgreSQL + pgvector (HNSW)
- **LLM:** Claude Sonnet 5.5 (Anthropic) for query routing (structured outputs) and response generation
- **Financial Data:** XBRL parsing via edgartools library
- **Database:** PostgreSQL 15+

## 📝 Notes

- SEC requires identity for API access (`SEC_IDENTITY` in `.env`)
- Voyage AI has batch size limit of 128 texts per API call
- The system automatically routes queries based on intent classification
- Financial facts keep the company's own line-item labels; the same concept can be labelled differently across companies (e.g. capex is "Purchases of property and equipment" at Amazon, "Plant construction and other property additions" at Dominion)
- Segment capex exists only where a company tags it: Amazon (quarterly) and Dominion (annual). Microsoft, Alphabet, Meta, Oracle and SpaceX report segment revenue and operating income only; CoreWeave reports a single segment
- "Leases not yet commenced" is every company's own tagging, found by a fixed precedence of rules in `facts.py` (dedicated concepts, then lease-labelled purchase obligations). Values are total undiscounted payments over terms of up to 20 years, not annual spending, and some arrangements are conditional. Google's 10-Qs before 2025 and some Oracle 10-Qs don't tag the figure

## 🤝 Contributing

I'm not seeking contributions at this time

## 📄 License

[Your License Here]

## 👤 Author

Daniel Savage (<dan.mcsavage@gmail.com>)

---

**Last Updated:** November 2026
