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
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile

# Initialize logger
logger = logging.getLogger(__name__)

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

    Supports hierarchical chunking, embedding, and multi-granularity search.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize session memory store.

        Args:
            db_path: Path to SQLite database (defaults to ./memory/agent_session_memory.db)
        """
        if db_path is None:
            # Default to ./memory/agent_session_memory.db
            memory_dir = Path.cwd() / "memory" / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(memory_dir / "agent_session_memory.db")

        self.db_path = db_path

        # Initialize database schema
        self._init_schema()

        # Initialize chunker (lazily, only when needed)
        self._chunker = None

        # Initialize embedding model (lazily, only when needed)
        self._embedding_model = None

    @property
    def chunker(self):
        """Lazy initialization of chunker"""
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def token_encoder(self):
        """Cached token encoder instance"""
        if not hasattr(self, '_token_encoder'):
            if TIKTOKEN_AVAILABLE:
                try:
                    self._token_encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._token_encoder = None
            else:
                self._token_encoder = None
        return self._token_encoder

    @property
    def embedding_model(self):
        """Lazy-load embedding model for semantic search."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device='cpu'
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = False
        return self._embedding_model if self._embedding_model is not False else None

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text using cached tokenizer.
        Falls back to character-based estimation if tokenizer unavailable.
        """
        if self.token_encoder:
            try:
                return len(self.token_encoder.encode(text))
            except Exception:
                pass

        # Fallback: ~4 characters per token
        return len(text) // 4

    def _format_bytes_human_readable(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with extensions and optimizations configured.

        Production-grade settings:
        - WAL mode for better concurrency
        - NORMAL synchronous mode for performance
        - 64MB cache for frequently accessed data
        - Foreign keys enabled
        - 5-second busy timeout for lock contention

        Returns:
            sqlite3.Connection with production optimizations
        """
        conn = sqlite3.connect(self.db_path)

        # STEP 1: Enable WAL mode for better concurrency
        # WAL allows readers to read while writer writes (no blocking)
        # Must be set before other PRAGMAs for maximum effect
        conn.execute("PRAGMA journal_mode=WAL")

        # STEP 2: Set synchronous to NORMAL (balance of safety and speed)
        # NORMAL is safe for most applications and much faster than FULL
        # FULL would fsync after every transaction (too slow for MCP server)
        conn.execute("PRAGMA synchronous=NORMAL")

        # STEP 3: Increase cache size to 64MB for performance
        # Negative value means size in KB (-64000 = 64MB)
        # Default is usually only ~2MB, this improves read performance significantly
        conn.execute("PRAGMA cache_size=-64000")

        # STEP 4: Enable foreign keys for referential integrity
        # SQLite doesn't enable this by default for backward compatibility
        # We need this for CASCADE deletes to work properly
        conn.execute("PRAGMA foreign_keys=ON")

        # STEP 5: Set busy timeout to handle concurrent access
        # 5-second timeout allows SQLite to wait for locks instead of failing immediately
        conn.execute("PRAGMA busy_timeout=5000")

        # Load sqlite-vec extension
        try:
            conn.enable_load_extension(True)

            # Build list of paths to try
            vec_paths = []

            # First, try to find sqlite_vec in Python's site-packages
            try:
                import sqlite_vec
                import os
                sqlite_vec_dir = os.path.dirname(sqlite_vec.__file__)
                vec_paths.append(os.path.join(sqlite_vec_dir, "vec0"))
            except ImportError:
                pass

            # Add common installation paths
            vec_paths.extend([
                "./vec0",
                "/usr/local/lib/vec0",
                "/opt/homebrew/lib/vec0",
            ])

            loaded = False
            for vec_path in vec_paths:
                try:
                    conn.load_extension(vec_path)
                    loaded = True
                    logger.info(f"Loaded vec0 extension from: {vec_path}")
                    break
                except sqlite3.OperationalError:
                    continue

            if not loaded:
                # Try loading without path (system-installed)
                try:
                    conn.load_extension("vec0")
                    loaded = True
                    logger.info("Loaded vec0 extension from system")
                except sqlite3.OperationalError:
                    pass

            conn.enable_load_extension(False)

            if not loaded:
                # Vector search will be unavailable but basic operations work
                logger.warning("Failed to load vec0 extension - vector search unavailable")

        except Exception as e:
            # Extension loading failed - continue without vector search
            logger.warning(f"Exception loading vec0 extension: {e}")

        return conn

    def __enter__(self):
        """Context manager entry - no-op, connections managed per operation"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no-op, connections managed per operation"""
        return False
