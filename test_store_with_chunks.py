#!/usr/bin/env python3
"""Direct test of store_memory with chunking"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.session_memory_store import SessionMemoryStore

# Initialize store
db_path = Path("./memory/agent_session_memory.db")
store = SessionMemoryStore(db_path)

test_content = """---
title: Direct Store Test
---

# Main Header

This is a test document with multiple sections.

## Section 1

Content for section 1 with substantial text to ensure chunking happens.

### Subsection 1.1

More detailed content in subsection 1.1.

## Section 2

Content for section 2.

### Subsection 2.1

Detailed content here.

### Subsection 2.2

More content in 2.2.

## Section 3

Final section with concluding remarks.
"""

print("=" * 60)
print("Testing store_memory with auto_chunk=True")
print("=" * 60)

try:
    result = store.store_memory(
        memory_type="reports",
        agent_id="test-agent",
        session_id="direct-test-session",
        content=test_content,
        session_iter=1,
        task_code="chunking-test",
        title="Direct Store Chunk Test",
        auto_chunk=True
    )

    print("\n✅ Storage result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if "chunks_stored" in result:
        print(f"\n🎉 Chunks were stored: {result['chunks_stored']}")
    else:
        print("\n⚠️  No chunks_stored field in result")

    # Query database to check
    memory_id = result.get("memory_id")
    if memory_id:
        import sqlite3
        import sqlite_vec

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
            (memory_id,)
        ).fetchone()[0]

        print(f"\n📊 Database query result:")
        print(f"  Chunks in DB for memory_id {memory_id}: {chunk_count}")

        if chunk_count > 0:
            chunks = conn.execute("""
                SELECT chunk_index, chunk_type, header_path, level,
                       substr(content, 1, 50) as preview
                FROM memory_chunks
                WHERE parent_id = ?
                ORDER BY chunk_index
            """, (memory_id,)).fetchall()

            print(f"\n  Chunk details:")
            for chunk in chunks:
                print(f"    [{chunk[0]}] {chunk[1]} | {chunk[2]} (level {chunk[3]}) | {chunk[4]}...")

        conn.close()

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)