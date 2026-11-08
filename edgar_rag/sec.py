"""EDGAR access via edgartools. Companies are always resolved by CIK."""

from functools import lru_cache

from edgar import Company, set_identity

from . import config

_identity_set = False


def _ensure_identity():
    global _identity_set
    if not _identity_set:
        if not config.SEC_IDENTITY:
            raise RuntimeError('SEC_IDENTITY is not set. Add e.g. SEC_IDENTITY="Your Name you@example.com" to .env')
        set_identity(config.SEC_IDENTITY)
        _identity_set = True


@lru_cache(maxsize=None)
def get_company(cik):
    _ensure_identity()
    return Company(cik)


def list_filings(spec):
    """All filings in scope for a company since START_DATE, oldest first."""
    company = get_company(spec.cik)
    filings = []
    for form in (*config.PERIODIC_FORMS, *spec.extra_forms):
        for f in company.get_filings(form=form).filter(date=f"{config.START_DATE}:"):
            if f.form != form:  # skip amendments (10-K/A etc.) that some filters let through
                continue
            filings.append(f)
    return sorted(filings, key=lambda f: (str(f.filing_date), f.accession_number))


def fiscal_year_end_month(spec):
    """Month (1-12) the company's fiscal year ends, from EDGAR company metadata (e.g. '0630')."""
    fye = getattr(get_company(spec.cik), "fiscal_year_end", None)
    try:
        return int(str(fye)[:2])
    except (TypeError, ValueError):
        return 12
