#!/usr/bin/env python3
"""
Update Existing Data After Schema Migration
============================================

Populates the new metadata fields for existing chunks.
"""

import sqlite3
import sys
from pathlib import Path


def update_existing_data(db_path: str):
    """
    Update existing chunks with metadata.

    Args:
        db_path: Path to SQLite database
    """
    print(f"🔧 Updating existing data in: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        print("\n📊 Step 1: Setting original_content from content...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET original_content = content
            WHERE original_content IS NULL OR original_content = ''
        """)
        conn.commit()
        print(f"  ✅ Updated {result.rowcount} chunks")

        print("\n📊 Step 2: Calculating granularity based on token_count...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET granularity_level = CASE
                WHEN token_count < 400 THEN 'fine'
                WHEN token_count BETWEEN 400 AND 1200 THEN 'medium'
                ELSE 'coarse'
            END
            WHERE token_count > 0
        """)
        conn.commit()
        print(f"  ✅ Updated {result.rowcount} chunks")

        print("\n📊 Step 3: Setting parent_title from session_memories...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET parent_title = (
                SELECT title FROM session_memories WHERE id = memory_chunks.parent_id
            )
            WHERE parent_title IS NULL
        """)
        conn.commit()
        print(f"  ✅ Updated {result.rowcount} chunks")

        print("\n📊 Step 4: Setting section_hierarchy from header_path...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET section_hierarchy = header_path
            WHERE section_hierarchy IS NULL AND header_path IS NOT NULL AND header_path != ''
        """)
        conn.commit()
        print(f"  ✅ Updated {result.rowcount} chunks")

        print("\n📊 Step 5: Calculating position ratio...")
        # Get all parent_ids
        parents = conn.execute("SELECT DISTINCT parent_id FROM memory_chunks").fetchall()

        updated_count = 0
        for parent_row in parents:
            parent_id = parent_row['parent_id']

            # Get total chunks for this parent
            total_chunks = conn.execute("""
                SELECT COUNT(*) as cnt FROM memory_chunks WHERE parent_id = ?
            """, (parent_id,)).fetchone()['cnt']

            if total_chunks > 0:
                # Update each chunk's position ratio
                conn.execute("""
                    UPDATE memory_chunks
                    SET chunk_position_ratio = CAST(chunk_index AS REAL) / ?
                    WHERE parent_id = ?
                """, (float(total_chunks), parent_id))
                updated_count += 1

        conn.commit()
        print(f"  ✅ Updated position ratios for {updated_count} documents")

        print("\n📊 Step 6: Setting depth_level from level...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET depth_level = COALESCE(level, 0)
        """)
        conn.commit()
        print(f"  ✅ Updated {result.rowcount} chunks")

        print("\n📊 Step 7: Detecting code blocks...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET contains_code = 1
            WHERE content LIKE '%```%'
        """)
        conn.commit()
        print(f"  ✅ Marked {result.rowcount} chunks as containing code")

        print("\n📊 Step 8: Detecting tables...")
        result = conn.execute("""
            UPDATE memory_chunks
            SET contains_table = 1
            WHERE content LIKE '%|%|%|%|%'
        """)
        conn.commit()
        print(f"  ✅ Marked {result.rowcount} chunks as containing tables")

        print("\n✅ Data update completed successfully!")

        # Show statistics
        print("\n📈 Statistics:")
        stats = conn.execute("""
            SELECT
                granularity_level,
                COUNT(*) as count,
                AVG(token_count) as avg_tokens
            FROM memory_chunks
            WHERE granularity_level IS NOT NULL
            GROUP BY granularity_level
        """).fetchall()

        for stat in stats:
            print(f"  {stat['granularity_level']:8s}: {stat['count']:4d} chunks (avg {stat['avg_tokens']:.0f} tokens)")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 002_update_existing_data.py <db_path>")
        print("\nExample:")
        print("  python 002_update_existing_data.py /path/to/agent_session_memory.db")
        sys.exit(1)

    db_path = sys.argv[1]
    update_existing_data(db_path)
