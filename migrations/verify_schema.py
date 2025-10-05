#!/usr/bin/env python3
"""
Schema Verification Script
===========================

Verifies that the database has all required columns and indexes.
"""

import sqlite3
import sys
from pathlib import Path


def verify_schema(db_path: str):
    """Verify database schema completeness."""

    print(f"🔍 Verifying schema for: {db_path}\n")

    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Expected columns for session_memories
    expected_session_memories = [
        'id', 'memory_type', 'agent_id', 'session_id', 'session_iter', 'task_code',
        'content', 'title', 'description', 'tags', 'metadata', 'content_hash',
        'created_at', 'updated_at', 'accessed_at', 'access_count',
        'document_structure', 'document_summary', 'estimated_tokens', 'chunk_strategy'
    ]

    # Expected columns for memory_chunks
    expected_memory_chunks = [
        'id', 'parent_id', 'chunk_index', 'content', 'chunk_type',
        'start_char', 'end_char', 'token_count', 'header_path', 'level',
        'prev_chunk_id', 'next_chunk_id', 'content_hash', 'created_at',
        'parent_title', 'section_hierarchy', 'granularity_level',
        'chunk_position_ratio', 'sibling_count', 'depth_level',
        'contains_code', 'contains_table', 'keywords',
        'original_content', 'is_contextually_enriched'
    ]

    # Expected indexes
    expected_indexes = [
        'idx_agent_session',
        'idx_agent_session_iter',
        'idx_agent_session_task',
        'idx_memory_type',
        'idx_created_at',
        'idx_session_iter',
        'idx_memory_type_iter',
        'idx_chunks_granularity',
        'idx_chunks_section',
        'idx_chunks_parent_title',
        'idx_chunks_contains_code'
    ]

    all_passed = True

    # Check session_memories table
    print("📋 Checking session_memories table...")
    cursor = conn.execute("PRAGMA table_info(session_memories)")
    actual_columns = [row['name'] for row in cursor.fetchall()]

    missing = set(expected_session_memories) - set(actual_columns)
    extra = set(actual_columns) - set(expected_session_memories)

    if missing:
        print(f"  ❌ Missing columns: {', '.join(missing)}")
        all_passed = False
    if extra:
        print(f"  ⚠️  Extra columns: {', '.join(extra)}")
    if not missing and not extra:
        print(f"  ✅ All {len(actual_columns)} columns present")

    # Check memory_chunks table
    print("\n📋 Checking memory_chunks table...")
    cursor = conn.execute("PRAGMA table_info(memory_chunks)")
    actual_columns = [row['name'] for row in cursor.fetchall()]

    missing = set(expected_memory_chunks) - set(actual_columns)
    extra = set(actual_columns) - set(expected_memory_chunks)

    if missing:
        print(f"  ❌ Missing columns: {', '.join(missing)}")
        all_passed = False
    if extra:
        print(f"  ⚠️  Extra columns: {', '.join(extra)}")
    if not missing and not extra:
        print(f"  ✅ All {len(actual_columns)} columns present")

    # Check indexes
    print("\n📋 Checking indexes...")
    cursor = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND name LIKE 'idx_%'
    """)
    actual_indexes = [row['name'] for row in cursor.fetchall()]

    missing = set(expected_indexes) - set(actual_indexes)
    extra = set(actual_indexes) - set(expected_indexes)

    if missing:
        print(f"  ❌ Missing indexes: {', '.join(missing)}")
        all_passed = False
    if extra:
        print(f"  ⚠️  Extra indexes: {', '.join(extra)}")
    if not missing and not extra:
        print(f"  ✅ All {len(actual_indexes)} indexes present")

    conn.close()

    if all_passed:
        print("\n✅ Schema verification PASSED")
        return True
    else:
        print("\n❌ Schema verification FAILED")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_schema.py <db_path>")
        print("\nExample:")
        print("  python verify_schema.py /path/to/agent_session_memory.db")
        sys.exit(1)

    db_path = sys.argv[1]
    success = verify_schema(db_path)
    sys.exit(0 if success else 1)
