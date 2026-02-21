# EDGAR Financial RAG System

A hybrid Retrieval-Augmented Generation (RAG) system for analyzing SEC filings using natural language queries. The system combines structured financial data (XBRL) with narrative content embeddings to provide comprehensive answers about company financials, risks, strategies, and business context.

## 🎯 Overview

This system provides a natural language interface to analyze SEC filings (10-K and 10-Q reports) for major tech companies. It intelligently routes queries to either structured financial databases or vector-based semantic search depending on the question type, then synthesizes coherent answers using Claude.

**Current Data Coverage:**

- **Companies:** Amazon (AMZN), Alphabet (GOOGL), Meta (META), Microsoft (MSFT), Oracle (ORCL)
- **Time Period:** January 2023 - Present
- **Filings:** 64+ total filings (10-K annual reports and 10-Q quarterly reports)

## ✨ Key Features

### 1. **Structured Financial Data (XBRL)**

- 10,847+ financial facts extracted from balance sheets, income statements, and cash flow statements
- Standardized XBRL concepts for consistent metrics across all companies
- SQL-queryable database for precise numerical queries with date ranges and filtering
- Best for: Revenue, expenses, equity, assets, and specific financial metrics

### 2. **Narrative Content Vector Search**

- 17,730+ text chunks from risk factors, MD&A sections, and business descriptions
- Vector embeddings via Voyage AI for semantic similarity search
- PostgreSQL with pgvector extension for efficient retrieval
- Best for: Risks, strategies, qualitative discussions, and business context

### 3. **Intelligent Query Router**

- Claude classifies each query as quantitative, qualitative, or hybrid
- Automatically retrieves from appropriate data sources
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
├── fetch_filings.py          # Download SEC filings from EDGAR
├── edgar_parser_v2.py         # Parse XBRL and extract financial facts
├── generate_embeddings.py     # Generate vector embeddings for narrative chunks
├── edgar_rag_interface.ipynb  # Main interactive interface
├── edgar_data_check.ipynb     # Data validation and statistics
├── test_embeddings.ipynb      # Test embedding generation
└── data/
    └── filings/               # Downloaded SEC filings in markdown format
        ├── AMZN/
        ├── GOOGL/
        ├── META/
        ├── MSFT/
        └── ORCL/
```

## 🚀 Setup

### Prerequisites

- Python 3.13+
- PostgreSQL with pgvector extension
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
   pip install edgartools sqlalchemy psycopg2-binary python-dotenv voyageai anthropic tqdm jupyter ipykernel
   ```

4. **Set up PostgreSQL:**

   ```bash
   # Install pgvector extension
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
   ```

### Data Pipeline

Run these scripts in order to set up your data:

1. **Fetch SEC filings:**

   ```bash
   python fetch_filings.py
   ```

   Downloads 10-K and 10-Q filings from EDGAR and saves them as markdown files in `data/filings/`.

2. **Parse filings and extract financial data:**

   ```bash
   python edgar_parser_v2.py
   ```

   Extracts XBRL financial facts and narrative chunks, then loads them into PostgreSQL.

3. **Generate embeddings:**

   ```bash
   python generate_embeddings.py
   ```

   Creates vector embeddings for narrative chunks using Voyage AI and stores them in the database.

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

### Example Use Cases

**Quantitative Analysis:**

```python
ask("What was Microsoft's total revenue in Q2 2025?")
ask("Compare operating expenses between Meta and Alphabet in 2024")
ask("Show me Amazon's cash flow trends over the past 8 quarters")
```

**Qualitative Analysis:**

```python
ask("What are the main risks Oracle faces in cloud infrastructure?")
ask("How does Meta describe its AI strategy?")
ask("What regulatory concerns are mentioned by these companies?")
```

**Hybrid Analysis:**

```python
ask("Analyze Amazon's infrastructure investments and related risk factors")
ask("Compare AI spending and strategic priorities across companies")
ask("What companies mention energy shortages and how much are they investing?")
```

## 🔧 Configuration

### Embedding Model

In `generate_embeddings.py`, you can configure the embedding model:

```python
EMBEDDING_MODEL = "voyage-3-lite"  # 512 dimensions, fast & cheap
# OR
EMBEDDING_MODEL = "voyage-3"       # 1024 dimensions, higher quality
```

### Companies and Date Range

In `fetch_filings.py`, customize the companies and time period:

```python
TICKERS = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]
START_DATE = "2023-01-01"
```

## 📊 Database Schema

### `financial_facts`

Structured financial data from XBRL:

- Standardized financial metrics
- Balance sheet, income statement, and cash flow data
- Period dates and filing dates

### `document_chunks`

Narrative content for vector search:

- Text chunks from SEC filings
- Section types (risk_factors, business, mda)
- Vector embeddings (512 or 1024 dimensions)

### `filings`

Metadata about each SEC filing

### `companies`

Reference table for company information

## 🎓 Technical Details

- **Embeddings:** Voyage AI (voyage-3-lite with 512 dimensions)
- **Vector Search:** PostgreSQL + pgvector extension
- **LLM:** Claude (Anthropic) for query routing and response generation
- **Financial Data:** XBRL parsing via edgartools library
- **Database:** PostgreSQL 14+

## 📝 Notes

- SEC requires identity for API access (configured in scripts with name/email)
- Voyage AI has batch size limit of 128 texts per API call
- The system automatically routes queries based on intent classification
- Financial facts use standardized XBRL concepts for cross-company comparisons

## 🤝 Contributing

Contributions are welcome! Some ideas for improvements:

- Add more companies and sectors
- Implement caching for common queries
- Add real-time filing updates
- Create visualization dashboards
- Add support for other filing types (8-K, proxy statements)

## 📄 License

[Your License Here]

## 👤 Author

Daniel Savage (<dan.mcsavage@gmail.com>)

---

**Last Updated:** February 2026
