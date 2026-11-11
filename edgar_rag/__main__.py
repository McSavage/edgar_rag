"""Command line entry point.

    python -m edgar_rag update [--tickers SPCX MSFT] [--dry-run] [--no-embed]
    python -m edgar_rag rebuild-facts [--tickers ...]
    python -m edgar_rag rebuild-series [--tickers ...]
    python -m edgar_rag embed [--reset]
    python -m edgar_rag status
    python -m edgar_rag migrate
    python -m edgar_rag ask "How much did Amazon spend on capex each quarter of 2025?"
"""

import argparse

from . import db, embeddings, ingest


def main():
    parser = argparse.ArgumentParser(prog="python -m edgar_rag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("update", help="ingest new filings (incremental by accession), then embed")
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--dry-run", action="store_true", help="list what would be ingested")
    p.add_argument("--no-embed", action="store_true")

    p = sub.add_parser("rebuild-facts", help="re-extract XBRL facts for already-ingested filings")
    p.add_argument("--tickers", nargs="+")

    p = sub.add_parser("rebuild-series", help="recompute series mappings and derived quarters from stored facts")
    p.add_argument("--tickers", nargs="+")

    p = sub.add_parser("embed", help="embed chunks that have no embedding")
    p.add_argument("--reset", action="store_true", help="discard ALL embeddings and re-embed (model change)")

    sub.add_parser("status", help="per-company coverage")
    sub.add_parser("migrate", help="create/upgrade the database schema")

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")

    args = parser.parse_args()
    if args.command == "update":
        ingest.update(args.tickers, dry_run=args.dry_run, embed=not args.no_embed)
        if not args.dry_run:
            print()
            ingest.status()
    elif args.command == "rebuild-facts":
        ingest.rebuild_facts(args.tickers)
        print()
        ingest.status()
    elif args.command == "rebuild-series":
        ingest.rebuild_series(args.tickers)
    elif args.command == "embed":
        embeddings.embed_pending(reset=args.reset)
    elif args.command == "status":
        ingest.status()
    elif args.command == "migrate":
        db.migrate()
        print("✓ Schema up to date")
    elif args.command == "ask":
        from .query import ask
        ask(args.question)


if __name__ == "__main__":
    main()
