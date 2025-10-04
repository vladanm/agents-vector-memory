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

        # Create embeddings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_embeddings (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES session_memories(id) ON DELETE CASCADE
            )
        """)

        # Create chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER NOT NULL,
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
