"""Single source of truth for configuration.

Everything that used to be duplicated across fetch_filings.py, edgar_parser.py,
generate_embeddings.py and the notebook (ticker list, dates, SEC identity, model
names, DB connection) lives here. The query router's company list and its
ticker validation are generated from COMPANIES, so adding a company is a
one-line change.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Explicit path: find_dotenv() walks stack frames and fails under `python -` / some notebooks.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class CompanySpec:
    ticker: str          # label used throughout the DB and the router
    cik: int             # EDGAR lookups go by CIK, never by ticker (see SPCX)
    name: str
    sector: str
    aliases: tuple = ()  # extra names the router should map to this ticker
    extra_forms: tuple = field(default_factory=tuple)  # forms beyond 10-K/10-Q (narrative only)


COMPANIES = [
    CompanySpec("AMZN", 1018724, "Amazon", "Hyperscaler", ("AWS", "Amazon Web Services")),
    CompanySpec("GOOGL", 1652044, "Alphabet", "Hyperscaler", ("Google", "Google Cloud")),
    CompanySpec("META", 1326801, "Meta Platforms", "Hyperscaler", ("Facebook",)),
    CompanySpec("MSFT", 789019, "Microsoft", "Hyperscaler", ("Azure",)),
    CompanySpec("ORCL", 1341439, "Oracle", "Hyperscaler", ("OCI", "Oracle Cloud")),
    CompanySpec("D", 715957, "Dominion Energy", "Regulated utility", ("Dominion", "Virginia Electric and Power")),
    CompanySpec("CRWV", 1769628, "CoreWeave", "AI cloud provider"),
    # Company("SPCX") in edgartools resolves to an unrelated ETF trust (CIK 1719812);
    # SpaceX must be looked up by CIK. No 10-K until FY2026 is reported, so the IPO
    # prospectus (424B4) is ingested for its risk factors / business / MD&A narrative.
    CompanySpec("SPCX", 1181412, "SpaceX", "Space launch and satellite connectivity",
                ("Space Exploration Technologies", "Starlink"), extra_forms=("424B4",)),
]
COMPANIES_BY_TICKER = {c.ticker: c for c in COMPANIES}
TICKERS = [c.ticker for c in COMPANIES]

START_DATE = "2020-01-01"
PERIODIC_FORMS = ("10-K", "10-Q")

# SEC requires a "Name email" User-Agent for EDGAR access.
SEC_IDENTITY = os.getenv("SEC_IDENTITY", "")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3-lite")
EMBEDDING_BATCH_SIZE = 128  # Voyage max texts per request
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


def database_url():
    return (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    )
