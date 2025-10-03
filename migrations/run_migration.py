#!/usr/bin/env python3
"""
Database Migration Runner
=========================

Applies SQL migrations to the vector memory database.
"""

import sqlite3
import sys
from pathlib import Path


def run_migration(db_path: str, migration_file: str):
    """
    Run a SQL migration file against the database.

    Args:
        db_path: Path to SQLite database
        migration_file: Path to SQL migration file
    """
    print(f"🔧 Running migration: {migration_file}")
    print(f"📁 Database: {db_path}")

    # Read migration SQL
    migration_path = Path(migration_file)
    if not migration_path.exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)

    with open(migration_path, 'r') as f:
        sql = f.read()

    # Connect to database
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Split SQL into individual statements
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]

        total = len(statements)
        print(f"\n📊 Executing {total} SQL statements...")

        for idx, statement in enumerate(statements, 1):
            # Skip comments
            if statement.startswith('--'):
                continue

            try:
                conn.execute(statement)
                print(f"  ✅ [{idx}/{total}] Executed successfully")
            except sqlite3.OperationalError as e:
                # Ignore "duplicate column" errors (migration already applied)
                if "duplicate column" in str(e).lower():
                    print(f"  ⚠️  [{idx}/{total}] Column already exists (skipping)")
                else:
                    raise

        conn.commit()
        print(f"\n✅ Migration completed successfully!")

        # Show updated schema
        print(f"\n📋 Updated memory_chunks schema:")
        cursor = conn.execute("PRAGMA table_info(memory_chunks)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col['name']}: {col['type']}")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_migration.py <db_path> <migration_file>")
        print("\nExample:")
        print("  python run_migration.py /path/to/agent_session_memory.db 001_add_granularity_metadata.sql")
        sys.exit(1)

    db_path = sys.argv[1]
    migration_file = sys.argv[2]

    run_migration(db_path, migration_file)
