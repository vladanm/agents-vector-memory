#!/usr/bin/env python3
"""
Database Migration: Add metadata columns to session_memories table
Author: Code Explorer Agent
Date: 2025-10-03
Purpose: Add keywords, contains_code, contains_table columns for document-level metadata
"""

import sqlite3
import sys
from pathlib import Path

def run_migration(db_path: str):
    """Add metadata columns to session_memories table"""
    print(f"Starting migration on database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(session_memories)")
        columns = {row[1] for row in cursor.fetchall()}

        print(f"Current columns in session_memories: {len(columns)}")

        migrations = []
        if 'keywords' not in columns:
            migrations.append(("keywords", "ALTER TABLE session_memories ADD COLUMN keywords TEXT DEFAULT '[]'"))
        if 'contains_code' not in columns:
            migrations.append(("contains_code", "ALTER TABLE session_memories ADD COLUMN contains_code INTEGER DEFAULT 0"))
        if 'contains_table' not in columns:
            migrations.append(("contains_table", "ALTER TABLE session_memories ADD COLUMN contains_table INTEGER DEFAULT 0"))

        if not migrations:
            print("✓ All columns already exist - migration not needed")
            return True

        # Execute migrations
        print(f"\nAdding {len(migrations)} new columns:")
        for col_name, sql in migrations:
            print(f"  - Adding column: {col_name}")
            cursor.execute(sql)

        # Create index for filtering by content type
        print("\nCreating index for content type filtering...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_type
            ON session_memories(contains_code, contains_table)
        """)

        conn.commit()
        print(f"\n✓ Successfully added {len(migrations)} columns")

        # Verify
        cursor.execute("PRAGMA table_info(session_memories)")
        new_columns = {row[1] for row in cursor.fetchall()}

        assert 'keywords' in new_columns, "keywords column not found after migration"
        assert 'contains_code' in new_columns, "contains_code column not found after migration"
        assert 'contains_table' in new_columns, "contains_table column not found after migration"

        print(f"✓ Migration verified - total columns now: {len(new_columns)}")

        # Show sample of new columns
        cursor.execute("""
            SELECT name, type, dflt_value
            FROM pragma_table_info('session_memories')
            WHERE name IN ('keywords', 'contains_code', 'contains_table')
        """)
        print("\nNew columns details:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]} (default: {row[2]})")

        return True

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"

    print("=" * 60)
    print("DATABASE MIGRATION: Add Metadata Columns")
    print("=" * 60)

    success = run_migration(db_path)

    if success:
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("MIGRATION FAILED")
        print("=" * 60)
        sys.exit(1)
