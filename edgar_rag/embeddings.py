"""Voyage AI embeddings for narrative chunks."""

import os
from functools import lru_cache

import voyageai
from sqlalchemy import text
from tqdm import tqdm

from . import config, db


@lru_cache(maxsize=1)
def voyage_client():
    return voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


def to_pgvector(values):
    return "[" + ",".join(map(str, values)) + "]"


def embed_query(query_text):
    return voyage_client().embed(texts=[query_text], model=config.EMBEDDING_MODEL,
                                 input_type="query").embeddings[0]


def model_dimension():
    return len(voyage_client().embed(texts=["dimension probe"], model=config.EMBEDDING_MODEL,
                                     input_type="document").embeddings[0])


def embed_pending(reset=False):
    """Embed every chunk that has no embedding yet. Returns (embedded, failed)."""
    engine = db.get_engine()
    dimension = model_dimension()
    with engine.begin() as conn:
        if reset:
            conn.execute(text("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw"))
            conn.execute(text("UPDATE document_chunks SET embedding = NULL"))
        db.ensure_vector_index(conn, dimension)
        total = conn.execute(text("SELECT count(*) FROM document_chunks WHERE embedding IS NULL")).scalar()

    if total == 0:
        print("✓ All chunks already embedded")
        return 0, 0

    print(f"Embedding {total:,} chunks with {config.EMBEDDING_MODEL} ({dimension} dims)")
    embedded = failed = 0
    last_id = 0  # keyset pagination: a failing batch is skipped, never re-fetched forever
    with tqdm(total=total) as bar:
        while True:
            with engine.connect() as conn:
                batch = conn.execute(text("""
                    SELECT id, chunk_text FROM document_chunks
                    WHERE embedding IS NULL AND id > :last_id
                    ORDER BY id LIMIT :n
                """), {"last_id": last_id, "n": config.EMBEDDING_BATCH_SIZE}).fetchall()
            if not batch:
                break
            last_id = batch[-1][0]
            try:
                vectors = voyage_client().embed(texts=[r[1] for r in batch], model=config.EMBEDDING_MODEL,
                                                input_type="document").embeddings
            except Exception as e:
                print(f"\n✗ Batch ending at id {last_id} failed: {e}")
                failed += len(batch)
                bar.update(len(batch))
                continue
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE document_chunks SET embedding = CAST(:e AS vector) WHERE id = :id"),
                    [{"e": to_pgvector(v), "id": r[0]} for r, v in zip(batch, vectors)],
                )
            embedded += len(batch)
            bar.update(len(batch))
    print(f"✓ Embedded {embedded:,} chunks" + (f", {failed:,} failed (re-run to retry)" if failed else ""))
    return embedded, failed
