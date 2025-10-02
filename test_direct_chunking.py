#!/usr/bin/env python3
"""
Direct test that bypasses MCP and directly calls store_memory
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.session_memory_store import SessionMemoryStore
import sqlite3
import sqlite_vec

# Test content
test_content = """---
title: Direct Chunking Test
---

# Main Section

This is the main section with some content.

## Subsection 1

Content for subsection 1.

### Deep Subsection

More detailed content.

## Subsection 2

Final section content.
"""

print("=" * 70)
print("DIRECT CHUNKING TEST (Bypassing MCP)")
print("=" * 70)

# Initialize store
db_path = Path("./memory/agent_session_memory.db")
print(f"\nDatabase: {db_path}")
print(f"Database exists: {db_path.exists()}")

store = SessionMemoryStore(db_path)
print("Store initialized")

# Call store_memory directly with auto_chunk=True
print("\nCalling store.store_memory() with auto_chunk=True...")
result = store.store_memory(
    memory_type="reports",
    agent_id="direct-test",
    session_id="direct-session",
    content=test_content,
    session_iter=1,
    task_code="direct-test",
    title="Direct Test",
    auto_chunk=True  # EXPLICITLY TRUE
)

print("\nResult:")
for key, value in result.items():
    print(f"  {key}: {value}")

if "chunks_stored" in result:
    print(f"\n✅ SUCCESS: chunks_stored field present: {result['chunks_stored']}")
else:
    print("\n⚠️  WARNING: No chunks_stored field in result")

# Query database directly
memory_id = result.get("memory_id")
if memory_id:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
        (memory_id,)
    ).fetchone()[0]

    print(f"\n📊 Database verification:")
    print(f"  Memory ID: {memory_id}")
    print(f"  Chunks in DB: {chunk_count}")

    if chunk_count > 0:
        print(f"\n✅ CHUNKING WORKS! Found {chunk_count} chunks")

        chunks = conn.execute("""
            SELECT chunk_index, chunk_type, header_path, level,
                   token_count, substr(content, 1, 60) as preview
            FROM memory_chunks
            WHERE parent_id = ?
            ORDER BY chunk_index
        """, (memory_id,)).fetchall()

        print(f"\nChunk details:")
        for chunk in chunks:
            print(f"  [{chunk[0]}] {chunk[1]} | path: '{chunk[2]}' | level: {chunk[3]} | tokens: {chunk[4]}")
            print(f"       preview: {chunk[5]}...")
    else:
        print(f"\n❌ CHUNKING FAILED: No chunks in database")

    conn.close()

print("\n" + "=" * 70)