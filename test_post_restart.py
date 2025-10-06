import sys
import os
from pathlib import Path

# Suppress sentence-transformers output
os.environ['SENTENCE_TRANSFORMERS_CACHE'] = '/tmp/.cache'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, str(Path.cwd() / "src"))

from src.session_memory_store import SessionMemoryStore
import sqlite3
import time

print("=" * 80)
print("POST-RESTART EMBEDDING GENERATION TEST")
print("=" * 80)

db_path = Path.cwd() / "memory" / "agent_session_memory.db"
store = SessionMemoryStore(db_path=str(db_path))

# Step 1: Create fresh test document
print("\n[Step 1] Creating fresh test document...")

start_time = time.time()

result = store.store_memory(
    memory_type="knowledge_base",
    agent_id="post-restart-test",
    session_id="post-restart-verification-20251006",
    content="""Post-Restart Test: This document verifies embedding generation works after MCP server restart.  It should create chunks with embeddings.""",
    title="Post-Restart Embedding Generation Test",
    auto_chunk=True,
    session_iter=1
)

elapsed = time.time() - start_time

print(f"Document stored in {elapsed:.2f}s")
print(f"Memory ID: {result.get('memory_id')}")
print(f"Chunks created: {result.get('chunks_created', 0)}")

memory_id = result.get('memory_id')

# Step 2: Query database
print("\n" + "=" * 80)
print("[Step 2] DATABASE CHECK")
print("=" * 80)

conn = sqlite3.connect(str(db_path))

chunk_count = conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?", (memory_id,)).fetchone()[0]
chunks_with_embeddings = conn.execute("SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ? AND embedding IS NOT NULL", (memory_id,)).fetchone()[0]
percentage = round(100.0 * chunks_with_embeddings / chunk_count, 1) if chunk_count > 0 else 0.0
vec_count = conn.execute("SELECT COUNT(*) FROM vec_session_search vs JOIN memory_chunks mc ON vs.chunk_id = mc.id WHERE mc.parent_id = ?", (memory_id,)).fetchone()[0]

sample = conn.execute("SELECT mc.chunk_index, length(mc.embedding) as emb_size FROM memory_chunks mc WHERE mc.parent_id = ? LIMIT 3", (memory_id,)).fetchall()

conn.close()

print(f"\nTotal chunks: {chunk_count}")
print(f"Chunks with embeddings: {chunks_with_embeddings}")
print(f"Percentage: {percentage}%")
print(f"Vec search entries: {vec_count}")
print(f"\nSample:")
for chunk_idx, emb_size in sample:
    status = f"✅ {emb_size} bytes" if emb_size else "❌ NULL"
    print(f"  Chunk {chunk_idx}: {status}")

# Step 3: Semantic search
print("\n" + "=" * 80)
print("[Step 3] SEMANTIC SEARCH TEST")
print("=" * 80)

search_result = store.search_with_granularity(
    memory_type="knowledge_base",
    granularity="specific_chunks",
    query="embedding test",
    session_id="post-restart-verification-20251006",
    limit=3
)

print(f"Search success: {search_result.get('success')}")
print(f"Results: {len(search_result.get('results', []))}")

if search_result.get('results'):
    for idx, r in enumerate(search_result['results'][:2], 1):
        print(f"  {idx}. Similarity: {r.get('similarity', 'N/A'):.4f}")

# FINAL VERDICT
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

criteria = {
    "chunks_created": chunk_count > 0,
    "embeddings_generated": percentage == 100.0,
    "vec_table_populated": vec_count > 0,
    "search_returns_results": len(search_result.get('results', [])) > 0,
    "similarity_scores_valid": False
}

if search_result.get('results'):
    first_sim = search_result['results'][0].get('similarity', 2.0)
    criteria["similarity_scores_valid"] = 0.0 <= first_sim <= 1.0

print("\nCRITERIA:")
for c, p in criteria.items():
    print(f"  {'✅' if p else '❌'} {c}")

if all(criteria.values()):
    print("\n🎉 ✅ PASS - EMBEDDINGS WORKING!")
else:
    print("\n⚠️  ❌ FAIL - EMBEDDINGS NOT WORKING")
print("=" * 80)
