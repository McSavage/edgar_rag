import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

load_dotenv()

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)
conn.autocommit = True
cursor = conn.cursor()

print("Creating scratch tables...")

# Drop tables if they exist (fresh start)
cursor.execute("DROP TABLE IF EXISTS sample_documents CASCADE;")
cursor.execute("DROP TABLE IF EXISTS sample_embeddings CASCADE;")

# Create a simple documents table
cursor.execute("""
    CREATE TABLE sample_documents (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

# Create a table with vector embeddings
cursor.execute("""
    CREATE TABLE sample_embeddings (
        id SERIAL PRIMARY KEY,
        doc_id INTEGER REFERENCES sample_documents(id),
        chunk_text TEXT,
        embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

# Insert some sample data
cursor.execute("""
    INSERT INTO sample_documents (title, content, category) VALUES
    ('First Document', 'This is a test document about AI', 'tech'),
    ('Second Document', 'Another document about databases', 'tech'),
    ('Third Document', 'Something about finance', 'finance');
""")

print("✓ Scratch database created!")
print("\nTables created:")
print("  - sample_documents")
print("  - sample_embeddings")

# Show what we created
cursor.execute("SELECT * FROM sample_documents;")
docs = cursor.fetchall()
print(f"\nInserted {len(docs)} sample documents")

cursor.close()
conn.close()

print("\n" + "="*50)
print("NAVIGATION OPTIONS:")
print("="*50)
print("\n1. COMMAND LINE (psql):")
print("   sudo -u postgres psql -d edgar_rag")
print("   Then try: \\dt  (list tables)")
print("            \\d sample_documents  (describe table)")
print("            SELECT * FROM sample_documents;")
print("            \\q  (quit)")

print("\n2. VS CODE PostgreSQL Extension:")
print("   - Click the PostgreSQL icon in the left sidebar")
print("   - Expand your connection")
print("   - Browse tables visually")
print("   - Right-click tables for options")

print("\n3. PYTHON (this file):")
print("   - Query with psycopg2")
print("   - Use pandas: pd.read_sql()")
print("   - Explore in Jupyter notebooks")

print("\n4. pgAdmin (if you want a full GUI):")
print("   sudo apt install pgadmin4")