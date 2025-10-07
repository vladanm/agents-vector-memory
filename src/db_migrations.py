"""
Database migrations for session memory store.
"""

import sqlite3
from pathlib import Path


def run_migrations(db_path: str) -> None:
    """
    Initialize or upgrade database schema.

    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)

    try:
        # Enable extension loading (required for sqlite-vec)
        conn.enable_load_extension(True)

        # Load sqlite-vec extension for vector search
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception:
            # If sqlite_vec not available, continue without it
            # Vector search features will not work but basic functionality will
            pass

        # Disable extension loading for security
        conn.enable_load_extension(False)

        # Enable WAL mode FIRST (before creating tables) for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

        # Set synchronous mode to NORMAL for performance (safe with WAL)
        conn.execute("PRAGMA synchronous=NORMAL")

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")

        # Create main memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_iter INTEGER DEFAULT 1,
                task_code TEXT,
                content TEXT NOT NULL,
                original_content TEXT,
                title TEXT,
                description TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                content_hash TEXT UNIQUE NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                auto_chunk INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                auto_chunked INTEGER DEFAULT 0
            )
        """)

        # ============================================================
        # SCHEMA UPGRADE: Add missing columns to existing databases
        # ============================================================
        # Check if we need to add missing columns (for databases created with old schema)
        cursor = conn.execute("PRAGMA table_info(session_memories)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add original_content column if missing (should be after content, but ALTER TABLE adds at end)
        if 'original_content' not in existing_columns:
            conn.execute("ALTER TABLE session_memories ADD COLUMN original_content TEXT")

        # Add embedding column if missing
        if 'embedding' not in existing_columns:
            conn.execute("ALTER TABLE session_memories ADD COLUMN embedding BLOB")

        # Add auto_chunk column if missing
        if 'auto_chunk' not in existing_columns:
            conn.execute("ALTER TABLE session_memories ADD COLUMN auto_chunk INTEGER DEFAULT 0")

        # Add chunk_count column if missing
        if 'chunk_count' not in existing_columns:
            conn.execute("ALTER TABLE session_memories ADD COLUMN chunk_count INTEGER DEFAULT 0")

        # Add auto_chunked column if missing
        if 'auto_chunked' not in existing_columns:
            conn.execute("ALTER TABLE session_memories ADD COLUMN auto_chunked INTEGER DEFAULT 0")

        # Create embeddings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_embeddings (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES session_memories(id) ON DELETE CASCADE
            )
        """)

        # Create chunks table with ALL required columns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER NOT NULL,
                parent_title TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'text',
                start_char INTEGER,
                end_char INTEGER,
                token_count INTEGER,
                header_path TEXT,
                level INTEGER DEFAULT 0,
                prev_chunk_id INTEGER,
                next_chunk_id INTEGER,
                content_hash TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                section_hierarchy TEXT,
                granularity_level TEXT DEFAULT 'medium',
                chunk_position_ratio REAL,
                sibling_count INTEGER,
                depth_level INTEGER,
                contains_code INTEGER DEFAULT 0,
                contains_table INTEGER DEFAULT 0,
                keywords TEXT DEFAULT '[]',
                original_content TEXT,
                is_contextually_enriched INTEGER DEFAULT 0,
                FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE,
                UNIQUE(parent_id, chunk_index)
            )
        """)

        # Create chunk embeddings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                id INTEGER PRIMARY KEY,
                chunk_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
            )
        """)

        # Create vector search virtual table for chunks (requires sqlite-vec extension)
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunk_search
                USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[384]
                )
            """)
        except Exception:
            # If vec0 extension not available, skip vector search table
            # This allows tests to run even without the extension
            pass

        # Create vector search virtual table for session memories (requires sqlite-vec extension)
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_session_search
                USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding float[384]
                )
            """)
        except Exception:
            # If vec0 extension not available, skip vector search table
            pass

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session ON session_memories(agent_id, session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session_iter ON session_memories(agent_id, session_id, session_iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session_task ON session_memories(agent_id, session_id, task_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON session_memories(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON session_memories(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_iter ON session_memories(session_iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_parent ON memory_chunks(parent_id)")

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to run migrations: {e}")
    finally:
        conn.close()
