"""
Generate embeddings for narrative chunks using Voyage AI
Voyage AI is Anthropic's recommended embedding provider
"""

import os
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import voyageai
from tqdm import tqdm

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Voyage AI requires its own API key
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Voyage embedding model
# voyage-3 is the latest, best for retrieval tasks
# Output dimension depends on model and API behavior.
# We probe this at runtime to prevent schema mismatches.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3-lite")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "0"))  # 0 = auto-detect

# Batch size for API calls (Voyage allows up to 128 texts per batch)
BATCH_SIZE = 128

# ─────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────

def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DB")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")


def get_current_embedding_dimension(engine):
    """Return current vector dimension for document_chunks.embedding, or None if unavailable."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relname = 'document_chunks'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1
        """))
        row = result.fetchone()

    if not row or not row[0]:
        return None

    match = re.search(r'vector\((\d+)\)', row[0])
    return int(match.group(1)) if match else None


def update_embedding_dimension(engine, dimension):
    """Update the embedding vector dimension in the database if needed."""
    current_dimension = get_current_embedding_dimension(engine)

    if current_dimension == dimension:
        print(f"✓ Embedding dimension already set to {dimension}; reusing existing column")
        return

    print(f"Updating embedding dimension from {current_dimension} to {dimension}...")
    with engine.connect() as conn:
        # Drop and recreate the embedding column with correct dimension
        conn.execute(text(f"""
            ALTER TABLE document_chunks 
            DROP COLUMN IF EXISTS embedding CASCADE;
        """))
        
        conn.execute(text(f"""
            ALTER TABLE document_chunks 
            ADD COLUMN embedding vector({dimension});
        """))
        
        # Create index for vector similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
        
        conn.commit()
        print("✓ Embedding column updated")


def probe_model_dimension(voyage_client):
    """Probe the model to get the actual embedding vector length."""
    result = voyage_client.embed(
        texts=["dimension probe"],
        model=EMBEDDING_MODEL,
        input_type="document"
    )
    return len(result.embeddings[0])


# ─────────────────────────────────────────────
# EMBEDDING GENERATION
# ─────────────────────────────────────────────

def generate_embeddings(texts, voyage_client, expected_dimension):
    """
    Generate embeddings for a batch of texts using Voyage AI.
    """
    try:
        result = voyage_client.embed(
            texts=texts,
            model=EMBEDDING_MODEL,
            input_type="document"  # Use "document" for texts to be retrieved
        )
        if result.embeddings and len(result.embeddings[0]) != expected_dimension:
            raise ValueError(
                f"Model returned dimension {len(result.embeddings[0])}, expected {expected_dimension}"
            )
        return result.embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def process_chunks(engine, voyage_client, embedding_dimension):
    """
    Fetch all chunks without embeddings and generate them in batches.
    """
    # Get total count
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL
        """))
        total_chunks = result.scalar()
    
    if total_chunks == 0:
        print("✓ All chunks already have embeddings!")
        return
    
    print(f"\nGenerating embeddings for {total_chunks:,} chunks...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dimension: {embedding_dimension}")
    print("=" * 60)
    
    # Process in batches
    offset = 0
    total_processed = 0
    
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        while offset < total_chunks:
            # Fetch batch of chunks
            with engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT id, chunk_text
                    FROM document_chunks
                    WHERE embedding IS NULL
                    ORDER BY id
                    LIMIT {BATCH_SIZE}
                """))
                
                batch = result.fetchall()
            
            if not batch:
                break
            
            # Extract texts and IDs
            chunk_ids = [row[0] for row in batch]
            chunk_texts = [row[1] for row in batch]
            
            # Generate embeddings
            embeddings = generate_embeddings(chunk_texts, voyage_client, embedding_dimension)
            
            if embeddings is None:
                print(f"\n✗ Failed to generate embeddings for batch starting at offset {offset}")
                offset += len(batch)
                pbar.update(len(batch))
                continue
            
            # Store embeddings in database
            with engine.connect() as conn:
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    # Convert embedding to PostgreSQL array format
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    
                    # Use CAST to avoid parameter binding issues with ::
                    conn.execute(text("""
                        UPDATE document_chunks
                        SET embedding = CAST(:embedding AS vector)
                        WHERE id = :id
                    """), {
                        'embedding': embedding_str,
                        'id': chunk_id
                    })
                
                conn.commit()
            
            total_processed += len(batch)
            offset += len(batch)
            pbar.update(len(batch))
    
    print(f"\n✓ Generated embeddings for {total_processed:,} chunks")


# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────

def print_statistics(engine):
    """Print embedding coverage statistics."""
    print("\n" + "=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)
    
    with engine.connect() as conn:
        # Overall stats
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(embedding) as chunks_with_embeddings,
                COUNT(*) - COUNT(embedding) as chunks_without_embeddings
            FROM document_chunks
        """))
        stats = result.fetchone()
        
        print(f"\nTotal chunks:              {stats[0]:,}")
        print(f"Chunks with embeddings:    {stats[1]:,}")
        print(f"Chunks without embeddings: {stats[2]:,}")
        
        if stats[0] > 0:
            coverage = (stats[1] / stats[0]) * 100
            print(f"Coverage:                  {coverage:.1f}%")
        
        # Per-company stats
        result = conn.execute(text("""
            SELECT 
                ticker,
                COUNT(*) as total,
                COUNT(embedding) as with_embedding
            FROM document_chunks
            GROUP BY ticker
            ORDER BY ticker
        """))
        
        print("\nPer-Company Coverage:")
        print("-" * 40)
        for row in result:
            ticker = row[0]
            total = row[1]
            with_emb = row[2]
            pct = (with_emb / total * 100) if total > 0 else 0
            print(f"  {ticker:6} {with_emb:5,}/{total:5,} ({pct:5.1f}%)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EDGAR RAG - EMBEDDING GENERATION")
    print("=" * 60)
    
    # Initialize Voyage AI client
    print("\nInitializing Voyage AI client...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print("✓ Voyage AI client initialized")

    # Determine actual model output dimension
    model_dimension = probe_model_dimension(voyage_client)
    print(f"✓ Model output dimension: {model_dimension}")

    if EMBEDDING_DIMENSION > 0 and EMBEDDING_DIMENSION != model_dimension:
        print(
            f"⚠ EMBEDDING_DIMENSION={EMBEDDING_DIMENSION} does not match model output "
            f"dimension {model_dimension}; using {model_dimension}."
        )

    embedding_dimension = model_dimension
    
    # Get database engine
    engine = get_engine()
    
    # Update embedding dimension if needed
    update_embedding_dimension(engine, embedding_dimension)
    
    # Generate embeddings
    process_chunks(engine, voyage_client, embedding_dimension)
    
    # Print statistics
    print_statistics(engine)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test vector similarity search")
    print("  2. Build RAG query interface")
    print("  3. Query your financial data + narratives!")


if __name__ == "__main__":
    main()