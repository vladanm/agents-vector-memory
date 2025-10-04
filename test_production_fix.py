#!/usr/bin/env python3
"""
Production Metadata Fix Verification Test
==========================================

Tests that metadata is populated correctly when auto_chunk=False (production pattern).
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore
from src.config import Config

db_path = '/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db'
store = SessionMemoryStore(db_path)

print("=" * 60)
print("PRODUCTION METADATA FIX VERIFICATION TEST")
print("=" * 60)

# Test 1: Create memory WITHOUT auto_chunk (production pattern)
test_content = '''
# API Documentation

This is a test document with **markdown** formatting.

## Code Example
```python
def hello():
    print("Hello World")
```

## Data Table
| Name | Value |
|------|-------|
| Test | 123   |
'''

print("\n[TEST 1] Creating memory with auto_chunk=False (production pattern)...")
result = store.store_memory(
    agent_id="test-agent",
    session_id="prod-test-session",
    content=test_content,
    memory_type="working_memory",
    title="Production Test Memory",
    description="Testing metadata with auto_chunk=False",
    auto_chunk=False  # ← CRITICAL: This is how production uses it
)

memory_id = result['memory_id']
print(f"✓ Created memory ID: {memory_id}")

# Verify metadata was populated
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT document_summary, estimated_tokens, keywords,
           contains_code, contains_table, document_structure
    FROM session_memories WHERE id = ?
""", (memory_id,))

row = cursor.fetchone()
if not row:
    print("✗ CRITICAL: Memory not found!")
    conn.close()
    sys.exit(1)

doc_summary, est_tokens, keywords, has_code, has_table, doc_structure = row

print("\n" + "=" * 60)
print("METADATA VERIFICATION RESULTS")
print("=" * 60)

tests_passed = 0
tests_failed = 0

# Test assertions
checks = [
    ("document_summary", doc_summary, doc_summary is not None and len(doc_summary) > 0),
    ("estimated_tokens", est_tokens, est_tokens is not None and est_tokens > 0),
    ("keywords", keywords, keywords is not None and len(keywords) > 2),  # More than just "[]"
    ("contains_code", has_code, has_code == 1),
    ("contains_table", has_table, has_table == 1),
    ("document_structure", doc_structure, doc_structure is not None)
]

for field, value, passed in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    if passed:
        tests_passed += 1
    else:
        tests_failed += 1

    # Truncate long values for display
    display_value = value
    if isinstance(value, str) and len(value) > 100:
        display_value = value[:97] + "..."

    print(f"{status} - {field}: {display_value}")

# Additional statistics query
print("\n" + "=" * 60)
print("DATABASE STATISTICS")
print("=" * 60)

cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN keywords != '[]' AND keywords IS NOT NULL THEN 1 ELSE 0 END) as with_keywords,
        SUM(contains_code) as with_code,
        SUM(contains_table) as with_tables
    FROM session_memories
""")

total, with_keywords, with_code, with_tables = cursor.fetchone()
keyword_pct = (with_keywords / total * 100) if total > 0 else 0

print(f"Total memories: {total}")
print(f"With keywords: {with_keywords} ({keyword_pct:.1f}%)")
print(f"With code: {with_code}")
print(f"With tables: {with_tables}")

conn.close()

print("\n" + "=" * 60)
print(f"Results: {tests_passed}/6 tests passed")
if tests_failed > 0:
    print(f"⚠️  WARNING: {tests_failed} tests failed")
    print("\nFIX STATUS: INCOMPLETE - Metadata still not populating correctly")
    sys.exit(1)
else:
    print("\n🎉 SUCCESS: All metadata fields populated correctly!")
    print("\nFIX STATUS: VERIFIED - Production usage now populates metadata")
    print(f"Improvement: Metadata population rate now at {keyword_pct:.1f}%")
    sys.exit(0)
