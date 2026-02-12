import os
from edgar import set_identity, Company
from datetime import datetime
from pathlib import Path

# Set identity (required by SEC)
set_identity("Daniel Savage dan.mcsavage@gmail.com")

# Your target companies
TICKERS = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]
START_DATE = "2023-01-01"

# Create data directory
DATA_DIR = Path("data/filings")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching EDGAR filings...")
print(f"Companies: {', '.join(TICKERS)}")
print(f"Date range: {START_DATE} to present")
print("=" * 60)

for ticker in TICKERS:
    print(f"\n📁 Fetching {ticker}...")
    
    try:
        # Get company
        company = Company(ticker)
        
        # Get filings
        filings_10k = company.get_filings(form="10-K").filter(date=f"{START_DATE}:")
        filings_10q = company.get_filings(form="10-Q").filter(date=f"{START_DATE}:")
        
        # Create company directory
        company_dir = DATA_DIR / ticker
        company_dir.mkdir(exist_ok=True)
        
        # Download 10-Ks
        for filing in filings_10k:
            print(f"  ✓ 10-K filed {filing.filing_date}")
            # Save filing details
            filing_path = company_dir / f"10K_{filing.filing_date}.md"
            filing_path.write_text(filing.markdown())
        
        # Download 10-Qs
        for filing in filings_10q:
            print(f"  ✓ 10-Q filed {filing.filing_date}")
            filing_path = company_dir / f"10Q_{filing.filing_date}.md"
            filing_path.write_text(filing.markdown())
            
        print(f"  Done! Files saved to {company_dir}")
        
    except Exception as e:
        print(f"  ✗ Error fetching {ticker}: {e}")

print("\n" + "=" * 60)
print("✓ Fetching complete!")
print(f"Files saved to: {DATA_DIR.absolute()}")