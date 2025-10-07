"""
Session Memory Store with Vector Search
========================================

Session-scoped memory management with semantic search via sqlite-vec.
"""

import os
import re
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import tempfile

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from .chunking import DocumentChunker, ChunkingConfig
from .memory_types import ContentFormat, ChunkEntry, get_memory_type_config

# Valid memory types
VALID_MEMORY_TYPES = [
    "session_context", "input_prompt", "system_memory", "reports",
    "report_observations", "working_memory", "knowledge_base"
]


class SessionMemoryStore:
    """
    Session-scoped memory storage with vector search.

    Provides:
    - Automatic document chunking with metadata enrichment
    - Semantic search across memories (coarse, medium, fine granularity)
    - Session and task code organization
    - Content hash deduplication
    """

    def __init__(self, db_path: str = None, embedding_model_name: str = None):
        """
        Initialize session memory store.

        Args:
            db_path: Path to SQLite database (default: memory/agent_session_memory.db)
            embedding_model_name: Name of sentence-transformers model for embeddings
        """
        # Set up database path
        if db_path is None:
            memory_dir = Path(__file__).parent.parent / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(memory_dir / "agent_session_memory.db")

        self.db_path = db_path

        # Initialize embedding model configuration
        self.embedding_model_name = embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._embedding_model = None  # Lazy loading

        # Initialize chunking
        self.chunker = DocumentChunker(ChunkingConfig())

        # Initialize database
        self._init_db()

    @property
    def embedding_model(self):
        """Lazy load embedding model to reduce memory usage."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                print("✅ Embedding model loaded successfully")
            except ImportError:
                print("⚠️  sentence-transformers not installed - semantic search unavailable")
                self._embedding_model = False  # Mark as unavailable
            except Exception as e:
                print(f"⚠️  Could not load embedding model: {e}")
                self._embedding_model = False

        return self._embedding_model if self._embedding_model is not False else None

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with sqlite-vec loaded.

        Configures busy_timeout to handle concurrent access from multiple processes.
        This is critical for MCP server usage where the server holds connections
        while clients make queries.

        Returns:
            sqlite3.Connection with busy_timeout=5000ms and sqlite-vec loaded
        """
        conn = sqlite3.connect(self.db_path)

        # CRITICAL FIX: Set busy timeout to handle concurrent access
        # Without this, any lock contention causes immediate "database is locked" error
        # With 5000ms timeout, SQLite waits for locks instead of failing
        conn.execute("PRAGMA busy_timeout = 5000")

        # Load sqlite-vec extension if available
        if SQLITE_VEC_AVAILABLE:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        return conn
