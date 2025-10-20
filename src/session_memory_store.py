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
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from .chunking import DocumentChunker, ChunkingConfig
from .memory_types import ContentFormat, ChunkEntry, get_memory_type_config
from .config import Config

# Import modular operations
from .storage import StorageOperations
from .search import SearchOperations
from .maintenance import MaintenanceOperations
from .chunking_storage import ChunkingStorageOperations
from .exceptions import (
    VectorMemoryException, ValidationError, MemoryError,
    SearchError, ChunkingError, DatabaseError, DatabaseLockError
)
from .retry_utils import exponential_backoff, retry_on_lock


# Valid memory types
VALID_MEMORY_TYPES = [
    "session_context", "input_prompt", "system_memory", "reports",
    "report_observations", "working_memory", "knowledge_base"
]


# ======================
# TASK 7: PERFORMANCE MONITORING
# ======================

def log_timing(operation_name: str):
    """
    Decorator to log operation timing and detect slow queries.

    Logs query execution time and adds timing metadata to result dict.
    Warns if query exceeds LOG_SLOW_QUERY_THRESHOLD.

    Args:
        operation_name: Human-readable name for the operation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start

                # Log timing if enabled
                if Config.LOG_QUERY_TIMING:
                    logger.info(f"[TIMING] {operation_name}: {elapsed:.3f}s")

                # Warn about slow queries
                if elapsed > Config.LOG_SLOW_QUERY_THRESHOLD:
                    logger.warning(
                        f"SLOW QUERY: {operation_name} took {elapsed:.3f}s "
                        f"(threshold: {Config.LOG_SLOW_QUERY_THRESHOLD}s)"
                    )

                # Add timing to result if it's a dict and config enabled
                if isinstance(result, dict) and Config.LOG_TIMING_TO_RESPONSE:
                    result["_timing"] = {
                        "operation": operation_name,
                        "elapsed_seconds": round(elapsed, 3)
                    }

                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[TIMING] {operation_name} FAILED after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


class SearchStatistics:
    """
    Track search performance statistics for monitoring and optimization.

    Maintains a rolling window of recent queries with timing and result metrics.
    Calculates P50, P95, P99 latencies and slow query rates.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize search statistics tracker.

        Args:
            max_history: Maximum number of queries to track (rolling window)
        """
        self.queries = []
        self.total_queries = 0
        self.slow_queries = 0
        self.max_history = max_history

    def record_query(self, query_type: str, elapsed: float, result_count: int,
                    metadata: dict = None):
        """
        Record query execution statistics.

        Args:
            query_type: Type of query (e.g., "search_fine", "search_medium")
            elapsed: Query execution time in seconds
            result_count: Number of results returned
            metadata: Optional additional metadata (filters, limits, etc.)
        """
        self.total_queries += 1

        if elapsed > Config.LOG_SLOW_QUERY_THRESHOLD:
            self.slow_queries += 1

        query_record = {
            "type": query_type,
            "elapsed": elapsed,
            "result_count": result_count,
            "timestamp": time.time(),
            "is_slow": elapsed > Config.LOG_SLOW_QUERY_THRESHOLD
        }

        if metadata:
            query_record["metadata"] = metadata

        self.queries.append(query_record)

        # Keep only last N queries (rolling window)
        if len(self.queries) > self.max_history:
            self.queries = self.queries[-self.max_history:]

    def get_stats(self) -> dict:
        """
        Get statistics summary with latency percentiles.

        Returns:
            Dict with query statistics including P50, P95, P99 latencies,
            slow query rate, and overall query metrics.
        """
        if not self.queries:
            return {
                "total_queries": self.total_queries,
                "recent_queries": 0,
                "average_time": 0.0,
                "median_time": 0.0,
                "p50_time": 0.0,
                "p95_time": 0.0,
                "p99_time": 0.0,
                "slow_queries": self.slow_queries,
                "slow_query_rate": 0.0,
                "queries_in_window": 0
            }

        times = sorted([q["elapsed"] for q in self.queries])
        n = len(times)

        # Calculate percentiles
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return {
            "total_queries": self.total_queries,
            "recent_queries": len(self.queries),
            "average_time": round(sum(times) / n, 3),
            "median_time": round(times[n // 2], 3),
            "p50_time": round(times[p50_idx], 3),
            "p95_time": round(times[p95_idx], 3),
            "p99_time": round(times[p99_idx], 3),
            "slow_queries": self.slow_queries,
            "slow_query_rate": round(self.slow_queries / self.total_queries, 3) if self.total_queries > 0 else 0.0,
            "queries_in_window": len(self.queries),
            "query_types": self._count_by_type()
        }

    def _count_by_type(self) -> dict:
        """Count queries by type."""
        counts = {}
        for q in self.queries:
            qtype = q["type"]
            counts[qtype] = counts.get(qtype, 0) + 1
        return counts

    def get_slow_queries(self, limit: int = 10) -> list:
        """
        Get the slowest queries for debugging.

        Args:
            limit: Maximum number of slow queries to return

        Returns:
            List of slowest query records
        """
        slow = [q for q in self.queries if q["is_slow"]]
        slow.sort(key=lambda x: x["elapsed"], reverse=True)
        return slow[:limit]


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

        # TASK 7: Initialize search statistics tracker
        self.search_stats = SearchStatistics()

        # Initialize operation modules
        self.storage = StorageOperations(self)
        self.search = SearchOperations(self)
        self.maintenance = MaintenanceOperations(self)
        self.chunking = ChunkingStorageOperations(self)

        # Warm-start embedding model (ensures it's ready before first use)
        if Config.WARM_START_EMBEDDING_MODEL:
            logger.info("Warming up embedding model...")
            _ = self.embedding_model  # Trigger lazy load

        # Check for chunks missing embeddings
        missing_count = self._count_chunks_without_embeddings()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} chunks without embeddings")

            # Auto-backfill if count is reasonable (<AUTO_BACKFILL_THRESHOLD)
            if missing_count < Config.AUTO_BACKFILL_THRESHOLD:
                logger.info("Auto-backfilling embeddings...")
                result = self.backfill_embeddings()
                if result["success"]:
                    logger.info(f"Auto-backfill complete: {result['chunks_processed']} chunks")
                else:
                    logger.error(f"Auto-backfill failed: {result.get('error', 'Unknown error')}")
            else:
                logger.warning(f"Too many missing embeddings ({missing_count}), manual backfill required")
                logger.info("Run store.backfill_embeddings() manually")

    # ======================
    # LAZY-LOADED PROPERTIES
    # ======================

    @property
    def token_encoder(self):
        """Cached token encoder instance."""
        if not hasattr(self, '_token_encoder'):
            if TIKTOKEN_AVAILABLE:
                try:
                    import tiktoken
                    self._token_encoder = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    logger.warning(f"Failed to initialize tiktoken: {e}")
                    self._token_encoder = None
            else:
                self._token_encoder = None
        return self._token_encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.token_encoder:
            return len(self.token_encoder.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

    # ======================
    # DATABASE CONNECTION & SCHEMA
    # ======================

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new database connection with vector search enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row

        # CRITICAL: Set foreign_keys FIRST before any other operations
        conn.execute("PRAGMA foreign_keys=ON")

        # Then set other PRAGMAs
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")

        # Performance optimizations
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        conn.execute("PRAGMA temp_store = MEMORY")

        # Enable vector search (load sqlite-vec)
        if SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec: {e}")

        return conn

    def _init_schema(self):
        """Initialize database schema with retry logic."""
        from .db_migrations import ensure_schema_up_to_date
        ensure_schema_up_to_date(self.db_path)

    # ======================
    # LAZY-LOADED COMPONENTS
    # ======================

    @property
    def chunker(self) -> 'DocumentChunker':
        """Lazy initialization of chunker"""
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def embedding_model(self) -> Any:
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

    def _count_chunks_without_embeddings(self) -> int:
        """
        Count chunks that don't have embeddings.

        Returns:
            Number of chunks without embeddings
        """
        try:
            conn = self._get_connection()
            result = conn.execute("""
                SELECT COUNT(*)
                FROM memory_chunks c
                LEFT JOIN vec_chunk_search v ON c.id = v.chunk_id
                WHERE v.chunk_id IS NULL
            """).fetchone()
            conn.close()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count chunks without embeddings: {e}")
            return 0

    def backfill_embeddings(self, batch_size: int = 32) -> dict[str, Any]:
        """
        Backfill embeddings for chunks that don't have them.

        This method finds all chunks without embeddings and generates
        embeddings for them in batches. It's designed to be safe to run
        multiple times (idempotent).

        Args:
            batch_size: Number of chunks to process per batch

        Returns:
            Dict with backfill statistics:
            {
                "success": bool,
                "chunks_processed": int,
                "total_chunks": int,
                "message": str,
                "error": str (optional)
            }
        """
        try:
            model = self.embedding_model
            if model is None:
                return {
                    "success": False,
                    "chunks_processed": 0,
                    "error": "Embedding model not available"
                }

            conn = self._get_connection()

            # Get chunks without embeddings
            chunks = conn.execute("""
                SELECT c.id, c.content
                FROM memory_chunks c
                LEFT JOIN vec_chunk_search v ON c.id = v.chunk_id
                WHERE v.chunk_id IS NULL
                ORDER BY c.id
            """).fetchall()

            total_chunks = len(chunks)
            if total_chunks == 0:
                conn.close()
                return {
                    "success": True,
                    "chunks_processed": 0,
                    "total_chunks": 0,
                    "message": "No chunks need backfilling"
                }

            logger.info(f"Backfilling embeddings for {total_chunks} chunks")
            processed = 0

            # Process in batches
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i+batch_size]
                chunk_ids = [c[0] for c in batch]
                contents = [c[1] for c in batch]

                # Generate embeddings
                embeddings = model.encode(
                    contents,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                # Insert into vec_chunk_search
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    conn.execute("""
                        INSERT INTO vec_chunk_search (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding.tobytes()))

                conn.commit()
                processed += len(batch)

                if processed % 100 == 0:
                    logger.info(f"Backfilled {processed}/{total_chunks} chunks")

            conn.close()

            logger.info(f"Backfill complete: {processed} chunks processed")

            return {
                "success": True,
                "chunks_processed": processed,
                "total_chunks": total_chunks,
                "message": f"Successfully backfilled {processed} embeddings"
            }

        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            return {
                "success": False,
                "chunks_processed": 0,
                "error": str(e)
            }

    # ======================
    # TASK 7: PERFORMANCE STATISTICS API
    # ======================

    def get_search_statistics(self) -> dict:
        """
        Get search performance statistics.

        Returns:
            Dict with query statistics including P50, P95, P99 latencies,
            slow query rate, and query type breakdown.
        """
        return self.search_stats.get_stats()

    def get_slow_queries(self, limit: int = 10) -> list:
        """
        Get the slowest queries for debugging.

        Args:
            limit: Maximum number of slow queries to return

        Returns:
            List of slowest query records with timing and metadata
        """
        return self.search_stats.get_slow_queries(limit)

    # ======================
    # PUBLIC API (Delegates to modules)
    # ======================

    def store_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Store a memory entry."""
        return self.storage.store_memory(*args, **kwargs)

    def search_memories(self, *args, **kwargs) -> dict[str, Any]:
        """Search memories with filters."""
        return self.search.search_memories(*args, **kwargs)

    def search_with_granularity(self, *args, **kwargs) -> dict[str, Any]:
        """Search with specific granularity."""
        return self.search.search_with_granularity(*args, **kwargs)

    def expand_chunk_context(self, *args, **kwargs) -> dict[str, Any]:
        """Expand chunk context."""
        return self.chunking.expand_chunk_context(*args, **kwargs)

    def get_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Get memory by ID."""
        return self.storage.get_memory(*args, **kwargs)

    def get_memory_by_id(self, *args, **kwargs) -> dict[str, Any]:
        """Get memory by ID (alias for get_memory)."""
        return self.storage.get_memory(*args, **kwargs)

    def delete_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Delete memory."""
        return self.storage.delete_memory(*args, **kwargs)

    def reconstruct_document(self, *args, **kwargs) -> dict[str, Any]:
        """Reconstruct document from chunks."""
        return self.storage.reconstruct_document(*args, **kwargs)

    def write_document_to_file(self, *args, **kwargs) -> dict[str, Any]:
        """Write document to file."""
        return self.storage.write_document_to_file(*args, **kwargs)

    def load_session_context_for_task(self, *args, **kwargs) -> dict[str, Any]:
        """Load session context."""
        return self.search.load_session_context_for_task(*args, **kwargs)

    def list_memories(self, *args, **kwargs) -> dict[str, Any]:
        """List memories."""
        return self.maintenance.list_memories(*args, **kwargs)

    def get_session_stats(self, *args, **kwargs) -> dict[str, Any]:
        """Get session statistics."""
        return self.maintenance.get_session_stats(*args, **kwargs)

    def list_sessions(self, *args, **kwargs) -> dict[str, Any]:
        """List recent sessions."""
        return self.maintenance.list_sessions(*args, **kwargs)

    def load_session_context(self, session_id: str, session_iter: str) -> dict[str, Any]:
        """Load all relevant session context for continuation."""
        return self._load_session_context_impl(session_id, session_iter)

    def cleanup_old_memories(self, *args, **kwargs) -> dict[str, Any]:
        """Cleanup old memories."""
        return self.maintenance.cleanup_old_memories(*args, **kwargs)

    # ======================
    # CORE IMPLEMENTATION METHODS
    # ======================

    def _store_memory_impl(
        self,
        memory_type: str,
        agent_id: str,
        session_id: str,
        content: str,
        session_iter: int = 1,
        task_code: str = None,
        title: str = None,
        description: str = None,
        tags: list[str] = None,
        metadata: dict[str, Any] = None,
        auto_chunk: bool = None,
        embedding: list[float] = None
    ) -> dict[str, Any]:
        """
        Store a memory entry with optional chunking and embedding.

        Args:
            memory_type: Type of memory (session_context, input_prompt, etc.)
            agent_id: Agent identifier
            session_id: Session identifier
            content: Memory content
            session_iter: Session iteration number
            task_code: Optional task code
            title: Optional title
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dict
            auto_chunk: Enable automatic chunking (defaults based on memory_type)
            embedding: Optional pre-computed embedding vector

        Returns:
            Dict with success status and memory details
        """
        try:
            # Validate memory type
            if memory_type not in VALID_MEMORY_TYPES:
                return {
                    "success": False,
                    "memory_id": None,
                    "memory_type": None,
                    "agent_id": None,
                    "session_id": None,
                    "content_hash": None,
                    "chunks_created": None,
                    "created_at": None,
                    "error": "Invalid memory type",
                    "message": f"Memory type must be one of: {VALID_MEMORY_TYPES}"
                }

            # Validate agent_id (must not be empty)
            if not agent_id or agent_id.strip() == "":
                return {
                    "success": False,
                    "memory_id": None,
                    "memory_type": None,
                    "agent_id": None,
                    "session_id": None,
                    "content_hash": None,
                    "chunks_created": None,
                    "created_at": None,
                    "error": "Invalid agent_id",
                    "message": "agent_id cannot be empty"
                }

            # Validate session_id (must not be empty, except for knowledge_base)
            if memory_type == "knowledge_base":
                # For knowledge_base, if no session_id provided, use "global"
                if not session_id or session_id.strip() == "":
                    session_id = "global"
            else:
                # For other memory types, session_id is required
                if not session_id or session_id.strip() == "":
                    return {
                        "success": False,
                        "memory_id": None,
                        "memory_type": None,
                        "agent_id": None,
                        "session_id": None,
                        "content_hash": None,
                        "chunks_created": None,
                        "created_at": None,
                        "error": "Invalid session_id",
                        "message": "session_id cannot be empty for this memory type"
                    }

            # Get memory type config
            config = get_memory_type_config(memory_type)

            # Determine if we should chunk (default based on memory type)
            if auto_chunk is None:
                auto_chunk = config.get('default_auto_chunk', False)

            # Generate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Prepare timestamps
            now = datetime.now(timezone.utc).isoformat()

            # Prepare tags and metadata as JSON
            tags_json = json.dumps(tags if tags else [])
            metadata_json = json.dumps(metadata if metadata else {})

            # CRITICAL FIX: Do expensive operations (chunking, embedding) BEFORE opening connection
            # This prevents holding database locks during long-running operations
            chunks = []
            chunks_created = 0
            if auto_chunk:
                chunk_metadata = {
                    'memory_type': memory_type,
                    'title': title or 'Untitled',
                    'enable_enrichment': True
                }

                # Chunk document (use placeholder memory_id=0, will update after insert)
                chunks = self.chunker.chunk_document(content, 0, chunk_metadata)

                # Generate embeddings for all chunks in batch (10-50x faster than sequential)
                # FIX: Always try to generate embeddings if chunks exist
                # The embedding_model property will handle lazy loading and error cases
                if chunks:
                    chunk_texts = [chunk.content for chunk in chunks]
                    try:
                        # Use the property (not _embedding_model) to trigger lazy loading
                        model = self.embedding_model
                        if model is not None:
                            embeddings = model.encode(chunk_texts, batch_size=Config.EMBEDDING_BATCH_SIZE, show_progress_bar=False)
                            # Convert embeddings to bytes and attach to chunks
                            for i, chunk in enumerate(chunks):
                                chunk.embedding = embeddings[i].tobytes()
                            logger.info(f"Generated {len(embeddings)} chunk embeddings")
                        else:
                            logger.warning("Embedding model not available, chunks will not have embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings for chunks: {e}")
                        # Continue without embeddings

            # NOW open connection - all expensive operations are done
            conn = self._get_connection()

            try:
                cursor = conn.execute("""
                    INSERT INTO session_memories
                    (memory_type, agent_id, session_id, session_iter, task_code, content,
                     title, description, tags, metadata, content_hash, created_at, updated_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_type, agent_id, session_id, session_iter, task_code, content,
                    title, description, tags_json, metadata_json, content_hash, now, now, now
                ))

                memory_id = cursor.lastrowid

                # Store embedding if provided
                if embedding:
                    conn.execute("""
                        INSERT INTO vec_session_search (memory_id, embedding)
                        VALUES (?, ?)
                    """, (memory_id, json.dumps(embedding).encode()))

                # Insert pre-computed chunks
                for chunk in chunks:
                    # Update parent_id to actual memory_id
                    chunk.parent_id = memory_id
                    conn.execute("""
                        INSERT INTO memory_chunks
                        (parent_id, chunk_index, content, chunk_type, start_char, end_char,
                         token_count, header_path, level, content_hash, created_at,
                         parent_title, section_hierarchy, granularity_level, chunk_position_ratio,
                         sibling_count, depth_level, contains_code, contains_table, keywords,
                         original_content, is_contextually_enriched, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.parent_id, chunk.chunk_index, chunk.content, chunk.chunk_type,
                        chunk.start_char, chunk.end_char, chunk.token_count, chunk.header_path,
                        chunk.level, chunk.content_hash, chunk.created_at,
                        chunk.parent_title, chunk.section_hierarchy, chunk.granularity_level,
                        chunk.chunk_position_ratio, chunk.sibling_count, chunk.depth_level,
                        chunk.contains_code, chunk.contains_table, json.dumps(chunk.keywords),
                        chunk.original_content, chunk.is_contextually_enriched,
                        chunk.embedding
                    ))

                    # Store chunk embedding in vector search if available
                    if chunk.embedding:
                        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        conn.execute("""
                            INSERT INTO vec_chunk_search (chunk_id, embedding)
                            VALUES (?, ?)
                        """, (chunk_id, chunk.embedding))

                chunks_created = len(chunks)

                conn.commit()
                conn.close()

                return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "content_hash": content_hash,
                    "chunks_created": chunks_created,
                    "created_at": now,
                    "message": f"Memory stored successfully with ID: {memory_id}",
                    "error": None
                }

            except Exception as e:
                conn.rollback()
                conn.close()
                raise

        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "memory_id": None,
                "memory_type": None,
                "agent_id": None,
                "session_id": None,
                "content_hash": None,
                "chunks_created": None,
                "created_at": None,
                "error": "Storage failed",
                "message": str(e)
            }

    def _search_memories_impl(
        self,
        memory_type: str = None,
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        query: str = None,
        limit: int = 3,
        latest_first: bool = True
    ) -> dict[str, Any]:
        """
        Search memories with optional filters.

        Args:
            memory_type: Optional memory type filter
            agent_id: Optional agent filter
            session_id: Optional session filter
            session_iter: Optional iteration filter
            task_code: Optional task code filter
            query: Optional semantic search query
            limit: Maximum results
            latest_first: Sort by newest first

        Returns:
            Dict with search results
        """
        try:
            conn = self._get_connection()

            # Build query
            conditions = []
            params = []

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type)
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if session_iter is not None:
                conditions.append("session_iter = ?")
                params.append(session_iter)
            if task_code:
                conditions.append("task_code = ?")
                params.append(task_code)

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            order_clause = "ORDER BY session_iter DESC, created_at DESC" if latest_first else "ORDER BY session_iter ASC, created_at ASC"

            sql = f"""
                SELECT * FROM session_memories
                {where_clause}
                {order_clause}
                LIMIT ?
            """
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            conn.close()

            # Format results
            # Column indices from PRAGMA table_info:
            # 0=id, 1=memory_type, 2=agent_id, 3=session_id, 4=session_iter(INT),
            # 5=task_code, 6=content, 7=original_content, 8=title, 9=description,
            # 10=tags, 11=metadata, 12=content_hash, 13=embedding, 14=created_at,
            # 15=updated_at, 16=accessed_at, 17=access_count(INT), 18=auto_chunk,
            # 19=chunk_count, 20=auto_chunked
            results = []
            for row in rows:
                # Convert session_iter from "v1" string to 1 integer if needed
                session_iter_val = row[4]
                if isinstance(session_iter_val, str) and session_iter_val.startswith('v'):
                    try:
                        session_iter_val = int(session_iter_val[1:])
                    except (ValueError, IndexError):
                        session_iter_val = 1  # Default fallback
                elif session_iter_val is None:
                    session_iter_val = 1  # Default fallback

                results.append({
                    "id": row[0],
                    "memory_type": row[1],
                    "agent_id": row[2],
                    "session_id": row[3],
                    "session_iter": session_iter_val,  # Converted to INTEGER
                    "task_code": row[5],
                    "content": row[6],
                    "title": row[8],  # Fixed: was row[7], skipped original_content
                    "description": row[9],  # Fixed: was row[8]
                    "tags": json.loads(row[10]) if row[10] else [],  # Fixed: was row[9]
                    "metadata": json.loads(row[11]) if row[11] else {},  # Fixed: was row[10]
                    "content_hash": row[12],  # Fixed: was row[11]
                    "created_at": row[14],  # Fixed: was row[12], skipped embedding
                    "updated_at": row[15] or row[14],  # Fixed: was row[13], fallback to created_at
                    "accessed_at": row[16],  # Fixed: was row[14]
                    "access_count": row[17],  # Fixed: was row[15], already INTEGER
                    "similarity": 2.0,  # Placeholder for scoped match
                    "source_type": "scoped"
                })

            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "filters": {
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code
                },
                "limit": limit,
                "latest_first": latest_first,
                "error": None,
                "message": None
            }

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {
                "success": False,
                "results": [],
                "total_results": 0,
                "query": query,
                "filters": {},
                "limit": limit,
                "latest_first": latest_first,
                "error": "Search failed",
                "message": str(e)
            }

    # ======================
    # TASK 2: ITERATIVE POST-FILTER FETCHING HELPER METHODS
    # ======================

    def _normalize_session_iter(self, value: str | int | None) -> int:
        """
        Normalize session_iter to integer for comparison.

        Converts string "v1" -> 1, "v2" -> 2, etc.
        Handles None as default iteration 1.

        Args:
            value: session_iter as string ("v1"), integer (1), or None

        Returns:
            Integer iteration number
        """
        if value is None:
            return 1

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            # Handle "v1" -> 1, "v2" -> 2 format
            if value.startswith('v'):
                try:
                    return int(value[1:])
                except (ValueError, IndexError):
                    return 1
            # Handle plain string "1" -> 1
            try:
                return int(value)
            except ValueError:
                return 1

        return 1

    def _passes_metadata_filters(self, row: tuple, filters: dict) -> bool:
        """
        Check if row passes all metadata filters.

        Args:
            row: Database row tuple
            filters: Dict with memory_type, agent_id, session_id, session_iter, task_code

        Returns:
            True if row passes all filters, False otherwise
        """
        # row indices: [8]=memory_type, [9]=agent_id, [10]=session_id, [11]=session_iter, [12]=task_code

        if filters.get("memory_type") and row[8] != filters["memory_type"]:
            return False
        if filters.get("agent_id") and row[9] != filters["agent_id"]:
            return False
        if filters.get("session_id") and row[10] != filters["session_id"]:
            return False

        # FIX: Normalize session_iter for type-safe comparison
        if filters.get("session_iter") is not None:
            row_iter = self._normalize_session_iter(row[11])
            filter_iter = self._normalize_session_iter(filters["session_iter"])
            if row_iter != filter_iter:
                return False

        if filters.get("task_code") and row[12] != filters["task_code"]:
            return False

        return True

    def _enrich_results_with_parent_metadata(self, conn: sqlite3.Connection, results: list) -> None:
        """
        Enrich search results with tags and metadata from parent memories.
        Modifies results list in-place by adding 'tags' and 'metadata' fields.

        Args:
            conn: Database connection
            results: List of result dicts with 'memory_id' field
        """
        if not results:
            return

        # Get unique memory_ids
        memory_ids = list(set([r["memory_id"] for r in results]))

        # Batch fetch tags and metadata for all memory_ids
        placeholders = ','.join('?' * len(memory_ids))
        rows = conn.execute(f"""
            SELECT id, tags, metadata
            FROM session_memories
            WHERE id IN ({placeholders})
        """, memory_ids).fetchall()

        # Build lookup dict
        metadata_lookup = {}
        for row in rows:
            metadata_lookup[row[0]] = {
                "tags": json.loads(row[1]) if row[1] else [],
                "metadata": json.loads(row[2]) if row[2] else {}
            }

        # Enrich results in-place
        for result in results:
            mid = result["memory_id"]
            if mid in metadata_lookup:
                result["tags"] = metadata_lookup[mid]["tags"]
                result["metadata"] = metadata_lookup[mid]["metadata"]

    def _iterative_vector_search(
        self,
        conn: sqlite3.Connection,
        query_bytes: bytes,
        limit: int,
        metadata_filters: dict
    ) -> list:
        """
        Iteratively fetch vector results until enough pass metadata filters.

        This solves the "250→3 after filters" starvation problem by continuing
        to fetch batches until we have enough filtered results.

        Args:
            conn: Database connection
            query_bytes: Query embedding as bytes
            limit: Target number of results after filtering
            metadata_filters: Dict with memory_type, agent_id, session_id, etc.

        Returns:
            List of filtered results (sorted by distance)
        """
        start_time = time.time()
        want = limit
        batch_size = max(Config.VECTOR_SEARCH_BATCH_SIZE, limit * 50)  # Start large for low selectivity
        offset = 0
        kept = []
        max_offset = Config.VECTOR_SEARCH_MAX_OFFSET  # Safety limit to prevent infinite loops

        logger.info(f"Starting iterative search: want={want}, initial_batch={batch_size}")

        while len(kept) < want and offset < max_offset:
            # Fetch next batch (no metadata filters in SQL - sqlite-vec limitation)
            sql = """
                SELECT
                    vc.chunk_id,
                    mc.parent_id as memory_id,
                    mc.chunk_index,
                    mc.content,
                    mc.chunk_type,
                    mc.header_path,
                    mc.level,
                    distance,
                    m.memory_type,
                    m.agent_id,
                    m.session_id,
                    m.session_iter,
                    m.task_code
                FROM vec_chunk_search vc
                JOIN memory_chunks mc ON vc.chunk_id = mc.id
                JOIN session_memories m ON mc.parent_id = m.id
                WHERE vc.embedding MATCH ?
                    AND k = ?
                ORDER BY distance
                LIMIT ? OFFSET ?
            """

            rows = conn.execute(sql, [query_bytes, batch_size + offset, batch_size, offset]).fetchall()

            if not rows:
                # No more results available
                logger.info(f"No more results available at offset={offset}")
                break

            if Config.LOG_FILTER_STATS:
                logger.info(f"Fetched batch: offset={offset}, size={len(rows)}, total_fetched={offset + len(rows)}")

            # Apply metadata filters in Python
            before_filter = len(kept)
            for row in rows:
                if not self._passes_metadata_filters(row, metadata_filters):
                    continue

                kept.append(row)
                if len(kept) >= want:
                    break

            passed_in_batch = len(kept) - before_filter
            if Config.LOG_FILTER_STATS:
                logger.info(f"  Batch passed filters: {passed_in_batch}/{len(rows)} ({100*passed_in_batch/len(rows):.1f}%)")

            # Update for next iteration
            offset += batch_size

            # Adaptively grow batch size if selectivity is very low
            if len(kept) < want * 0.1 and passed_in_batch < len(rows) * 0.05:
                # Less than 10% of target and less than 5% passing filters
                old_batch_size = batch_size
                batch_size = min(batch_size * Config.VECTOR_SEARCH_GROWTH_FACTOR, 5000)
                logger.info(f"Low selectivity detected, growing batch_size: {old_batch_size} → {batch_size}")

        elapsed = time.time() - start_time
        logger.info(
            f"Iterative search performance: "
            f"elapsed={elapsed:.3f}s, "
            f"fetched={offset}, "
            f"kept={len(kept)}, "
            f"target={want}, "
            f"batches={offset//Config.VECTOR_SEARCH_BATCH_SIZE}"
        )

        if len(kept) < want:
            logger.warning(f"Could not find enough results: wanted={want}, got={len(kept)}")

        # Sort by distance and limit
        kept.sort(key=lambda x: x[7])  # Sort by distance (index 7)
        return kept[:limit]

    # ======================
    # SEARCH WITH GRANULARITY
    # ======================

    @log_timing("vector_search_fine")
    def _search_with_granularity_impl(
        self,
        memory_type: str,
        granularity: str,
        query: str,
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        limit: int = 3,
        similarity_threshold: float = 0.5,
        auto_merge_threshold: float = 0.6
    ) -> dict[str, Any]:
        """
        Search with specific granularity level.

        Granularity levels:
        - fine: Individual chunks (<400 tokens)
        - medium: Section-level with auto-merging (400-1200 tokens)
        - coarse: Full documents

        Args:
            memory_type: Type of memory to search
            granularity: Search granularity (fine/medium/coarse)
            query: Semantic search query
            agent_id: Optional agent filter
            session_id: Optional session filter
            session_iter: Optional iteration filter
            task_code: Optional task code filter
            limit: Maximum results
            similarity_threshold: Minimum similarity (0.0-1.0) - KEPT FOR BACKWARD COMPATIBILITY, NOT USED FOR FILTERING
            auto_merge_threshold: For medium, merge if >=X siblings match

        Returns:
            Dict with search results at specified granularity
        """
        search_start = time.time()

        if granularity == "coarse":
            # COARSE GRANULARITY: Full document search via chunk aggregation
            logger.info("=" * 80)
            logger.info("COARSE GRANULARITY SEARCH STARTING (CHUNK AGGREGATION STRATEGY)")
            logger.info(f"  memory_type: {memory_type}")
            logger.info(f"  query: {query[:100]}...")
            logger.info(f"  agent_id: {agent_id}")
            logger.info(f"  session_id: {session_id}")
            logger.info(f"  session_iter: {session_iter}")
            logger.info(f"  task_code: {task_code}")
            logger.info(f"  limit: {limit} documents")
            logger.info("=" * 80)

            try:
                # Check embedding model
                model = self.embedding_model
                if model is None:
                    return {
                        "success": True,
                        "results": [],
                        "total_results": 0,
                        "granularity": "coarse",
                        "message": "Embedding model not available",
                        "error": None
                    }

                # Step 1: Get many fine-grained chunks (20x requested documents)
                chunk_fetch_limit = limit * 20
                logger.info(f"Step 1: Fetching {chunk_fetch_limit} fine-grained chunks")

                # Generate query embedding
                query_embedding = model.encode([query], show_progress_bar=False)[0]
                query_bytes = query_embedding.tobytes()

                conn = self._get_connection()

                # Build metadata filters dict
                metadata_filters = {
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code
                }

                # Fetch fine-grained chunks with iterative search
                chunk_rows = self._iterative_vector_search(
                    conn, query_bytes, chunk_fetch_limit, metadata_filters
                )

                logger.info(f"  → Retrieved {len(chunk_rows)} chunks after filtering")

                if not chunk_rows:
                    conn.close()
                    logger.info("No chunks found matching filters")
                    return {
                        "success": True,
                        "results": [],
                        "total_results": 0,
                        "granularity": "coarse",
                        "message": "No matching chunks found",
                        "error": None
                    }

                # Step 2: Aggregate chunks by parent document (memory_id)
                logger.info("Step 2: Aggregating chunks by document")
                doc_chunks = {}  # memory_id -> list of (chunk_id, distance, content)

                for row in chunk_rows:
                    chunk_id = row[0]
                    memory_id = row[1]
                    content = row[3]
                    distance = row[7]

                    if memory_id not in doc_chunks:
                        doc_chunks[memory_id] = []
                    doc_chunks[memory_id].append((chunk_id, distance, content))

                logger.info(f"  → Found {len(doc_chunks)} unique documents")

                # Step 3: Rank documents by best chunk similarity
                logger.info("Step 3: Ranking documents by best chunk similarity")
                doc_scores = []

                for memory_id, chunks in doc_chunks.items():
                    # Best similarity = lowest distance
                    best_distance = min(c[1] for c in chunks)
                    best_similarity = 1.0 - (best_distance**2 / 2.0)
                    matching_chunks = len(chunks)

                    doc_scores.append({
                        "memory_id": memory_id,
                        "best_similarity": best_similarity,
                        "best_distance": best_distance,
                        "matching_chunks": matching_chunks,
                        "chunk_details": chunks
                    })

                # Sort by best similarity (highest first)
                doc_scores.sort(key=lambda x: x["best_similarity"], reverse=True)
                logger.info(f"  → Top document similarity: {doc_scores[0]['best_similarity']:.4f}")

                # Step 4: Fetch full document content for top N documents
                top_docs = doc_scores[:limit]
                logger.info(f"Step 4: Fetching full content for top {len(top_docs)} documents")

                results = []
                for doc_score in top_docs:
                    memory_id = doc_score["memory_id"]

                    # Get full document metadata
                    doc_row = conn.execute("""
                        SELECT id, memory_type, agent_id, session_id, session_iter,
                               task_code, content, title, description, tags, metadata,
                               content_hash, created_at, updated_at
                        FROM session_memories
                        WHERE id = ?
                    """, (memory_id,)).fetchone()

                    if not doc_row:
                        continue

                    # Get all chunks for this document (ordered by chunk_index)
                    all_chunks = conn.execute("""
                        SELECT chunk_index, content, chunk_type, header_path
                        FROM memory_chunks
                        WHERE parent_id = ?
                        ORDER BY chunk_index ASC
                    """, (memory_id,)).fetchall()

                    # Build full document content from chunks
                    full_content = "\n\n".join([c[1] for c in all_chunks])

                    results.append({
                        "memory_id": doc_row[0],
                        "memory_type": doc_row[1],
                        "agent_id": doc_row[2],
                        "session_id": doc_row[3],
                        "session_iter": doc_row[4],
                        "task_code": doc_row[5],
                        "content": full_content,
                        "title": doc_row[7],
                        "description": doc_row[8],
                        "tags": json.loads(doc_row[9]) if doc_row[9] else [],
                        "metadata": json.loads(doc_row[10]) if doc_row[10] else {},
                        "content_hash": doc_row[11],
                        "created_at": doc_row[12],
                        "updated_at": doc_row[13],
                        "similarity": float(doc_score["best_similarity"]),
                        "matching_chunks": doc_score["matching_chunks"],
                        "total_chunks": len(all_chunks),
                        "source": "full_document",
                        "granularity": "coarse"
                    })

                conn.close()

                logger.info("=" * 80)
                logger.info("COARSE GRANULARITY RESULTS:")
                logger.info(f"  Total documents returned: {len(results)}")
                if results:
                    logger.info(f"  Best similarity: {results[0]['similarity']:.4f}")
                    logger.info(f"  Best document: {results[0]['title'] or 'Untitled'}")
                    logger.info(f"  Matching chunks: {results[0]['matching_chunks']}/{results[0]['total_chunks']}")
                logger.info("=" * 80)

                # Record statistics
                elapsed = time.time() - search_start
                self.search_stats.record_query(
                    f"search_coarse_{memory_type}",
                    elapsed,
                    len(results),
                    {
                        "memory_type": memory_type,
                        "filters": metadata_filters,
                        "chunk_fetch_limit": chunk_fetch_limit,
                        "documents_found": len(doc_chunks)
                    }
                )

                return {
                    "success": True,
                    "results": results,
                    "total_results": len(results),
                    "granularity": "coarse",
                    "message": None,
                    "error": None
                }

            except Exception as e:
                elapsed = time.time() - search_start
                logger.error(f"Coarse granularity search failed after {elapsed:.3f}s: {e}", exc_info=True)

                # Record failed query in stats
                self.search_stats.record_query(
                    f"search_coarse_{memory_type}_FAILED",
                    elapsed,
                    0,
                    {"error": str(e)}
                )

                return {
                    "success": False,
                    "results": [],
                    "total_results": 0,
                    "granularity": "coarse",
                    "message": str(e),
                    "error": "Search failed"
                }

        elif granularity == "fine":
            # Return individual chunks using vector search with TASK 2 iterative fetching
            logger.info("=" * 80)
            logger.info("FINE GRANULARITY SEARCH STARTING (TASK 2: ITERATIVE FETCHING)")
            logger.info(f"  memory_type: {memory_type}")
            logger.info(f"  query: {query[:100]}...")
            logger.info(f"  agent_id: {agent_id}")
            logger.info(f"  session_id: {session_id}")
            logger.info(f"  session_iter: {session_iter}")
            logger.info(f"  task_code: {task_code}")
            logger.info(f"  limit: {limit}")
            logger.info(f"  similarity_threshold: {similarity_threshold} (NOT USED FOR FILTERING)")
            logger.info("=" * 80)
            try:
                # Get embedding model
                model = self.embedding_model
                if model is None:
                    return {
                        "success": True,
                        "results": [],
                        "total_results": 0,
                        "granularity": "fine",
                        "message": "Embedding model not available",
                        "error": None
                    }

                # Generate query embedding
                query_embedding = model.encode([query], show_progress_bar=False)[0]
                query_bytes = query_embedding.tobytes()

                # TASK 2 FIX: Use iterative fetching instead of fixed k×10 multiplier
                conn = self._get_connection()

                # Build metadata filters dict
                metadata_filters = {
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code
                }

                # Use iterative search to handle low selectivity filters
                rows = self._iterative_vector_search(conn, query_bytes, limit, metadata_filters)

                logger.info(f"Iterative search returned {len(rows)} filtered chunks")

                # Convert rows to results (no more filtering needed - already done in iterative search)
                results = []
                for row in rows:
                    distance = row[7]
                    similarity = 1.0 - (distance**2 / 2.0)  # Convert L2 distance to similarity

                    results.append({
                        "chunk_id": row[0],
                        "memory_id": row[1],
                        "chunk_index": row[2],
                        "chunk_content": row[3],
                        "chunk_type": row[4],
                        "header_path": row[5],
                        "level": row[6],
                        "similarity": float(similarity),
                        "source": "chunk",
                        "granularity": "fine"
                    })

                # Enrich results with parent memory tags and metadata
                self._enrich_results_with_parent_metadata(conn, results)

                conn.close()

                logger.info("=" * 80)
                logger.info("FINAL RESULTS:")
                logger.info(f"  Total results: {len(results)} chunks")
                if results:
                    logger.info(f"  Best similarity: {results[0]['similarity']}")
                    if len(results) > 1:
                        logger.info(f"  Worst similarity: {results[-1]['similarity']}")
                logger.info("=" * 80)

                # Record statistics
                elapsed = time.time() - search_start
                self.search_stats.record_query(
                    f"search_fine_{memory_type}",
                    elapsed,
                    len(results),
                    {
                        "memory_type": memory_type,
                        "limit": limit,
                        "filters": metadata_filters
                    }
                )

                return {
                    "success": True,
                    "results": results,
                    "total_results": len(results),
                    "granularity": "fine",
                    "message": None,
                    "error": None
                }

            except Exception as e:
                elapsed = time.time() - search_start
                logger.error(f"Fine granularity search failed after {elapsed:.3f}s: {e}", exc_info=True)

                # Record failed query in stats
                self.search_stats.record_query(
                    f"search_fine_{memory_type}_FAILED",
                    elapsed,
                    0,
                    {"error": str(e)}
                )

                return {
                    "success": False,
                    "results": [],
                    "total_results": 0,
                    "granularity": "fine",
                    "message": str(e),
                    "error": "Search failed"
                }

        else:  # medium granularity
            # Return section-level results with auto-merging
            try:
                # First do fine search to find matching chunks
                fine_result = self._search_with_granularity_impl(
                    memory_type, "fine", query, agent_id, session_id,
                    session_iter, task_code, limit * 5,  # Get more chunks for section grouping
                    similarity_threshold, auto_merge_threshold
                )

                if not fine_result["success"] or not fine_result["results"]:
                    return {
                        "success": True,
                        "results": [],
                        "total_results": 0,
                        "granularity": "medium",
                        "message": fine_result.get("message", "No matching chunks found"),
                        "error": None
                    }

                # Group chunks by section (using header_path parent)
                conn = self._get_connection()
                sections = {}

                for chunk_result in fine_result["results"]:
                    memory_id = chunk_result["memory_id"]
                    header_path = chunk_result["header_path"]

                    # Get section key (parent header, e.g., "## Natural Language Processing")
                    if not header_path or header_path == "/":
                        section_key = (memory_id, "/")
                    else:
                        # Extract parent section (last level-2 heading or root)
                        parts = [p.strip() for p in header_path.split(">") if p.strip()]
                        # Find the highest level heading (usually level 2: ##)
                        # BUG FIX: Extract H1 > H2 section header (first TWO parts), not just H1 root
                        section_key = (memory_id, " > ".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else "/")

                    if section_key not in sections:
                        sections[section_key] = {
                            "memory_id": memory_id,
                            "section_header": section_key[1],
                            "matching_chunks": [],
                            "all_chunk_ids": set()
                        }

                    sections[section_key]["matching_chunks"].append(chunk_result)
                    sections[section_key]["all_chunk_ids"].add(chunk_result["chunk_id"])

                # For each section, get all chunks and determine if we should merge
                section_results = []

                for section_key, section_data in sections.items():
                    memory_id = section_data["memory_id"]
                    section_header = section_data["section_header"]

                    # Get all chunks in this section
                    if section_header == "/":
                        # Root level - get all chunks
                        section_chunks = conn.execute("""
                            SELECT id, chunk_index, content, chunk_type, header_path, level
                            FROM memory_chunks
                            WHERE parent_id = ?
                            ORDER BY chunk_index
                        """, (memory_id,)).fetchall()
                    else:
                        # Specific section - get chunks with matching header prefix
                        section_chunks = conn.execute("""
                            SELECT id, chunk_index, content, chunk_type, header_path, level
                            FROM memory_chunks
                            WHERE parent_id = ?
                                AND (header_path = ? OR header_path LIKE ?)
                            ORDER BY chunk_index
                        """, (memory_id, section_header, f"{section_header} >%")).fetchall()

                    if not section_chunks:
                        continue

                    # Calculate match ratio
                    total_chunks = len(section_chunks)
                    matched_chunks = len(section_data["matching_chunks"])
                    match_ratio = matched_chunks / total_chunks if total_chunks > 0 else 0

                    # Build section content
                    section_content = "\n\n".join([chunk[2] for chunk in section_chunks])

                    # Get average similarity of matching chunks
                    avg_similarity = sum(c["similarity"] for c in section_data["matching_chunks"]) / matched_chunks

                    section_results.append({
                        "memory_id": memory_id,
                        "section_header": section_header,
                        "section_content": section_content,
                        "header_path": section_header,
                        "chunks_in_section": total_chunks,
                        "matched_chunks": matched_chunks,
                        "match_ratio": float(match_ratio),
                        "auto_merged": match_ratio >= auto_merge_threshold,
                        "similarity": float(avg_similarity),
                        "source": "expanded_section",
                        "granularity": "medium"
                    })

                # Enrich results with parent memory tags and metadata
                self._enrich_results_with_parent_metadata(conn, section_results)

                conn.close()

                # Sort by similarity and limit
                section_results.sort(key=lambda x: x["similarity"], reverse=True)
                section_results = section_results[:limit]

                # Record statistics
                elapsed = time.time() - search_start
                self.search_stats.record_query(
                    f"search_medium_{memory_type}",
                    elapsed,
                    len(section_results),
                    {
                        "memory_type": memory_type,
                        "limit": limit,
                        "auto_merge_threshold": auto_merge_threshold
                    }
                )

                return {
                    "success": True,
                    "results": section_results,
                    "total_results": len(section_results),
                    "granularity": "medium",
                    "message": None,
                    "error": None
                }

            except Exception as e:
                elapsed = time.time() - search_start
                logger.error(f"Medium granularity search failed after {elapsed:.3f}s: {e}", exc_info=True)

                # Record failed query in stats
                self.search_stats.record_query(
                    f"search_medium_{memory_type}_FAILED",
                    elapsed,
                    0,
                    {"error": str(e)}
                )

                return {
                    "success": False,
                    "results": [],
                    "total_results": 0,
                    "granularity": "medium",
                    "message": str(e),
                    "error": "Search failed"
                }
    def _expand_chunk_context_impl(
        self,
        chunk_id: int,
        surrounding_chunks: int = 2
    ) -> dict[str, Any]:
        """
        Expand chunk context by retrieving surrounding chunks.

        Args:
            chunk_id: Target chunk ID
            surrounding_chunks: Number of chunks before/after to retrieve

        Returns:
            Dict with target chunk and surrounding context
        """
        try:
            conn = self._get_connection()

            # Get target chunk and find its parent_id and chunk_index
            target = conn.execute("""
                SELECT * FROM memory_chunks
                WHERE id = ?
            """, (chunk_id,)).fetchone()

            if not target:
                conn.close()
                return {
                    "success": False,
                    "memory_id": None,
                    "target_chunk_index": None,
                    "context_window": None,
                    "chunks_returned": None,
                    "expanded_content": None,
                    "chunks": None,
                    "error": "Chunk not found",
                    "message": f"No chunk found with ID: {chunk_id}"
                }

            # Column indices: 0=id, 1=parent_id, 2=parent_title, 3=chunk_index, 4=content,
            # 5=chunk_type, 6=start_char, 7=end_char, 8=token_count, 9=header_path, 10=level
            memory_id = target[1]  # parent_id
            chunk_index = target[3]  # chunk_index (correct column)

            # Get surrounding chunks
            start_index = max(0, chunk_index - surrounding_chunks)
            end_index = chunk_index + surrounding_chunks

            chunks = conn.execute("""
                SELECT * FROM memory_chunks
                WHERE parent_id = ? AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index ASC
            """, (memory_id, start_index, end_index)).fetchall()

            conn.close()

            # Format results
            all_chunks = []
            for chunk in chunks:
                all_chunks.append({
                    "chunk_id": chunk[0],
                    "chunk_index": chunk[3],  # Fixed: correct column index
                    "content": chunk[4],       # Fixed: correct column index
                    "chunk_type": chunk[5],    # Fixed: correct column index
                    "header_path": chunk[9],
                    "level": chunk[10]
                })

            # Build expanded content
            expanded_content = "\n\n".join([c["content"] for c in all_chunks])

            return {
                "success": True,
                "memory_id": memory_id,
                "target_chunk_index": chunk_index,
                "context_window": surrounding_chunks,
                "chunks_returned": len(all_chunks),
                "expanded_content": expanded_content,
                "chunks": all_chunks,
                "error": None,
                "message": None
            }

        except Exception as e:
            return {
                "success": False,
                "memory_id": None,
                "target_chunk_index": None,
                "context_window": None,
                "chunks_returned": None,
                "expanded_content": None,
                "chunks": None,
                "error": "Context expansion failed",
                "message": str(e)
            }

    def _load_session_context_for_task_impl(
        self,
        agent_id: str,
        session_id: str,
        current_task_code: str
    ) -> dict[str, Any]:
        """
        Load session context only if agent previously worked on the same task_code.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            current_task_code: Current task code to match

        Returns:
            Dict with session context if found, empty otherwise
        """
        try:
            conn = self._get_connection()

            # Find latest session_context memory matching agent, session, and task_code
            row = conn.execute("""
                SELECT * FROM session_memories
                WHERE memory_type = 'session_context'
                AND agent_id = ?
                AND session_id = ?
                AND task_code = ?
                ORDER BY session_iter DESC, created_at DESC
                LIMIT 1
            """, (agent_id, session_id, current_task_code)).fetchone()

            conn.close()

            if not row:
                return {
                    "success": True,
                    "has_context": False,
                    "context": None,
                    "message": f"No previous context found for task_code: {current_task_code}",
                    "error": None
                }

            return {
                "success": True,
                "has_context": True,
                "context": {
                    "id": row[0],
                    "memory_type": row[1],
                    "agent_id": row[2],
                    "session_id": row[3],
                    "session_iter": row[4],
                    "task_code": row[5],
                    "content": row[6],
                    "title": row[8],
                    "description": row[9],
                    "tags": json.loads(row[10]) if row[10] else [],
                    "metadata": json.loads(row[11]) if row[11] else {},
                    "created_at": row[14],
                    "updated_at": row[15]
                },
                "message": f"Found context for task_code: {current_task_code}",
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to load session context: {e}", exc_info=True)
            return {
                "success": False,
                "has_context": False,
                "context": None,
                "message": str(e),
                "error": "Failed to load session context"
            }

    def _get_memory_impl(self, memory_id: int) -> dict[str, Any]:
        """
        Retrieve specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Dict with memory details or error
        """
        try:
            conn = self._get_connection()

            row = conn.execute("""
                SELECT * FROM session_memories WHERE id = ?
            """, (memory_id,)).fetchone()

            conn.close()

            if not row:
                return {
                    "success": False,
                    "memory": None,
                    "error": "Memory not found",
                    "message": f"No memory with ID: {memory_id}"
                }

            return {
                "success": True,
                "memory": {
                    "id": row[0],
                    "memory_type": row[1],
                    "agent_id": row[2],
                    "session_id": row[3],
                    "session_iter": row[4],
                    "task_code": row[5],
                    "content": row[6],
                    "title": row[8],
                    "description": row[9],
                    "tags": json.loads(row[10]) if row[10] else [],
                    "metadata": json.loads(row[11]) if row[11] else {},
                    "content_hash": row[12],
                    "created_at": row[14],
                    "updated_at": row[15],
                    "accessed_at": row[16],
                    "access_count": row[17]
                },
                "error": None,
                "message": None
            }

        except Exception as e:
            logger.error(f"Failed to get memory: {e}", exc_info=True)
            return {
                "success": False,
                "memory": None,
                "error": "Retrieval failed",
                "message": str(e)
            }

    def _delete_memory_impl(self, memory_id: int) -> dict[str, Any]:
        """
        Delete a memory and all associated data.

        Args:
            memory_id: Memory identifier

        Returns:
            Dict with success status
        """
        try:
            conn = self._get_connection()

            # Check if memory exists
            row = conn.execute("""
                SELECT id FROM session_memories WHERE id = ?
            """, (memory_id,)).fetchone()

            if not row:
                conn.close()
                return {
                    "success": False,
                    "memory_id": None,
                    "error": "Memory not found",
                    "message": f"No memory with ID: {memory_id}"
                }

            # Delete (cascades to chunks)
            conn.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "memory_id": memory_id,
                "error": None,
                "message": f"Memory {memory_id} deleted successfully"
            }

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}", exc_info=True)
            return {
                "success": False,
                "memory_id": None,
                "error": "Deletion failed",
                "message": str(e)
            }

    def _reconstruct_document_impl(self, memory_id: int) -> dict[str, Any]:
        """
        Reconstruct a document from its chunks.

        Args:
            memory_id: Memory identifier

        Returns:
            Dict with reconstructed content
        """
        try:
            conn = self._get_connection()

            # Get parent memory
            parent = conn.execute("""
                SELECT content, title, memory_type FROM session_memories WHERE id = ?
            """, (memory_id,)).fetchone()

            if not parent:
                conn.close()
                return {
                    "success": False,
                    "memory_id": None,
                    "content": None,
                    "title": None,
                    "memory_type": None,
                    "chunk_count": 0,
                    "error": "Memory not found",
                    "message": f"No memory with ID: {memory_id}"
                }

            # Get chunks
            chunks = conn.execute("""
                SELECT chunk_index, content, chunk_type, header_path, level, original_content
                FROM memory_chunks
                WHERE parent_id = ?
                ORDER BY chunk_index ASC
            """, (memory_id,)).fetchall()

            conn.close()

            if not chunks:
                # No chunks, return original content
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "content": parent[0],
                    "title": parent[1],
                    "memory_type": parent[2],
                    "chunk_count": 0,
                    "message": "No chunks found, returning original content",
                    "error": None
                }

            # Reconstruct from chunks using original_content (index 5) if available,
            # fallback to enriched content (index 1) for backward compatibility
            reconstructed_parts = []
            for chunk in chunks:
                # Use original_content if available (not None), otherwise use content
                clean_content = chunk[5] if chunk[5] is not None else chunk[1]
                reconstructed_parts.append(clean_content)

            reconstructed_content = '\n\n'.join(reconstructed_parts)

            return {
                "success": True,
                "memory_id": memory_id,
                "content": reconstructed_content,
                "title": parent[1],
                "memory_type": parent[2],
                "chunk_count": len(chunks),
                "message": f"Document reconstructed from {len(chunks)} chunks",
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "memory_id": None,
                "content": None,
                "title": None,
                "memory_type": None,
                "chunk_count": 0,
                "error": "Reconstruction failed",
                "message": str(e)
            }

    def _write_document_to_file_impl(
        self,
        memory_id: int,
        output_path: str = None,
        include_metadata: bool = True,
        format: str = "markdown"
    ) -> dict[str, Any]:
        """
        Write a reconstructed document to disk.

        Args:
            memory_id: Memory identifier
            output_path: Path to write file (defaults to temp file)
            include_metadata: Include YAML frontmatter metadata
            format: Output format (markdown, text)

        Returns:
            Dict with file path and status
        """
        try:
            # Reconstruct document
            doc = self.reconstruct_document(memory_id)
            if not doc["success"]:
                return doc

            # Generate output path if not provided
            if output_path is None:
                import tempfile
                suffix = ".md" if format == "markdown" else ".txt"
                fd, output_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)

            # Build content
            content_parts = []

            if include_metadata and format == "markdown":
                # Add YAML frontmatter
                metadata = {
                    "memory_id": memory_id,
                    "title": doc["title"],
                    "memory_type": doc["memory_type"],
                    "chunk_count": doc["chunk_count"]
                }
                if YAML_AVAILABLE:
                    import yaml
                    content_parts.append("---")
                    content_parts.append(yaml.dump(metadata, default_flow_style=False).strip())
                    content_parts.append("---")
                    content_parts.append("")

            content_parts.append(doc["content"])

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_parts))

            # Calculate file size
            file_size_bytes = len('\n'.join(content_parts).encode('utf-8'))
            file_size_mb = file_size_bytes / (1024 * 1024)
            file_size_human = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_bytes / 1024:.2f} KB"

            # Estimate tokens
            estimated_tokens = self.count_tokens('\n'.join(content_parts)) if hasattr(self, 'count_tokens') else None

            return {
                "success": True,
                "file_path": output_path,
                "file_size_bytes": file_size_bytes,
                "file_size_human": file_size_human,
                "estimated_tokens": estimated_tokens,
                "memory_id": memory_id,
                "created_at": doc.get("created_at"),
                "message": f"Document written to {output_path}",
                "error_code": None,
                "error_message": None
            }

        except Exception as e:
            return {
                "success": False,
                "file_path": None,
                "file_size_bytes": None,
                "file_size_human": None,
                "estimated_tokens": None,
                "memory_id": None,
                "created_at": None,
                "message": str(e),
                "error_code": "WRITE_FAILED",
                "error_message": str(e)
            }

    def _get_session_stats_impl(self, agent_id: str | None, session_id: str | None) -> dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            agent_id: Optional agent filter
            session_id: Optional session identifier

        Returns:
            Dict with session statistics
        """
        try:
            conn = self._get_connection()

            # Build WHERE clause dynamically
            where_parts = []
            params = []

            if session_id:
                where_parts.append("session_id = ?")
                params.append(session_id)

            if agent_id:
                where_parts.append("agent_id = ?")
                params.append(agent_id)

            where_clause = " AND ".join(where_parts) if where_parts else "1=1"

            # Get overall stats
            stats = conn.execute(f"""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    COUNT(DISTINCT agent_id) as unique_agents,
                    COUNT(DISTINCT task_code) as unique_tasks,
                    MAX(session_iter) as max_session_iter,
                    AVG(LENGTH(content)) as avg_content_length,
                    SUM(access_count) as total_access_count
                FROM session_memories
                WHERE {where_clause}
            """, tuple(params)).fetchone()

            # Get breakdown by memory type
            breakdown_rows = conn.execute(f"""
                SELECT memory_type, COUNT(*) as count
                FROM session_memories
                WHERE {where_clause}
                GROUP BY memory_type
                ORDER BY count DESC
            """, tuple(params)).fetchall()

            conn.close()

            # Build memory type breakdown dict
            memory_type_breakdown = {row[0]: row[1] for row in breakdown_rows}

            # Convert max_session_iter from "v1" string to 1 integer if needed
            max_session_iter_val = stats[4]
            if isinstance(max_session_iter_val, str) and max_session_iter_val and max_session_iter_val.startswith('v'):
                try:
                    max_session_iter_val = int(max_session_iter_val[1:])
                except (ValueError, IndexError):
                    max_session_iter_val = 1  # Default fallback
            elif max_session_iter_val is None:
                max_session_iter_val = 1

            return {
                "success": True,
                "total_memories": stats[0],
                "memory_types": stats[1],
                "unique_agents": stats[2],
                "unique_sessions": 1 if session_id else stats[0],
                "unique_tasks": stats[3],
                "max_session_iter": max_session_iter_val,
                "avg_content_length": round(stats[5], 2) if stats[5] else 0.0,
                "total_access_count": stats[6] or 0,
                "memory_type_breakdown": memory_type_breakdown,
                "filters": {"agent_id": agent_id, "session_id": session_id},
                "error": None,
                "message": f"Found {stats[0]} memories"
            }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}", exc_info=True)
            return {
                "success": False,
                "total_memories": None,
                "memory_types": None,
                "unique_agents": None,
                "unique_sessions": None,
                "unique_tasks": None,
                "max_session_iter": None,
                "avg_content_length": None,
                "total_access_count": None,
                "memory_type_breakdown": None,
                "filters": {"agent_id": agent_id, "session_id": session_id},
                "error": "Stats retrieval failed",
                "message": str(e)
            }

    def _list_sessions_impl(self, agent_id: str | None, limit: int) -> dict[str, Any]:
        """
        List recent sessions with activity counts.

        Args:
            agent_id: Optional agent filter
            limit: Maximum number of sessions to return

        Returns:
            Dict with list of sessions ordered by most recent first
        """
        try:
            conn = self._get_connection()

            # Build WHERE clause
            where_clause = "agent_id = ?" if agent_id else "1=1"
            params = [agent_id] if agent_id else []

            # Query sessions with aggregated info
            rows = conn.execute(f"""
                SELECT
                    agent_id,
                    session_id,
                    COUNT(*) as memory_count,
                    MAX(session_iter) as latest_iter,
                    MAX(created_at) as latest_activity,
                    MIN(created_at) as first_activity,
                    GROUP_CONCAT(DISTINCT memory_type) as memory_types
                FROM session_memories
                WHERE {where_clause}
                GROUP BY agent_id, session_id
                ORDER BY latest_activity DESC
                LIMIT ?
            """, tuple(params + [limit])).fetchall()

            conn.close()

            # Build session info list
            sessions = []
            for row in rows:
                # Convert latest_iter from "v1" to 1 if needed
                latest_iter_val = row[2]
                if isinstance(latest_iter_val, str) and latest_iter_val and latest_iter_val.startswith('v'):
                    try:
                        latest_iter_val = int(latest_iter_val[1:])
                    except (ValueError, IndexError):
                        latest_iter_val = 1
                elif latest_iter_val is None:
                    latest_iter_val = 1

                sessions.append({
                    "agent_id": row[0],
                    "session_id": row[1],
                    "memory_count": row[2],
                    "latest_iter": latest_iter_val,
                    "latest_activity": row[4],
                    "first_activity": row[5],
                    "memory_types": row[6].split(',') if row[6] else []
                })

            return {
                "success": True,
                "sessions": sessions,
                "total_sessions": len(sessions),
                "agent_filter": agent_id,
                "limit": limit,
                "error": None,
                "message": f"Found {len(sessions)} sessions"
            }

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}", exc_info=True)
            return {
                "success": False,
                "sessions": [],
                "total_sessions": 0,
                "agent_filter": agent_id,
                "limit": limit,
                "error": "List sessions failed",
                "message": str(e)
            }

    def _load_session_context_impl(self, session_id: str, session_iter: str) -> dict[str, Any]:
        """
        Load all relevant session context for task continuation.

        Args:
            session_id: Session identifier
            session_iter: Session iteration (e.g., "v1", "v2")

        Returns:
            Dict with session context if found
        """
        try:
            conn = self._get_connection()

            # Convert session_iter to integer for comparison
            session_iter_int = None
            if isinstance(session_iter, str) and session_iter.startswith('v'):
                try:
                    session_iter_int = int(session_iter[1:])
                except (ValueError, IndexError):
                    session_iter_int = 1
            elif isinstance(session_iter, int):
                session_iter_int = session_iter
            else:
                session_iter_int = 1

            # Find latest session_context memory for this session
            row = conn.execute("""
                SELECT * FROM session_memories
                WHERE memory_type = 'session_context'
                AND session_id = ?
                AND session_iter = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (session_id, session_iter)).fetchone()

            conn.close()

            if not row:
                return {
                    "success": True,
                    "found_previous_context": False,
                    "context": None,
                    "message": f"No session context found for session {session_id} iteration {session_iter}",
                    "error": None
                }

            return {
                "success": True,
                "found_previous_context": True,
                "context": {
                    "id": row[0],
                    "memory_type": row[1],
                    "agent_id": row[2],
                    "session_id": row[3],
                    "session_iter": session_iter_int,
                    "task_code": row[5],
                    "content": row[6],
                    "title": row[8],
                    "description": row[9],
                    "tags": json.loads(row[10]) if row[10] else [],
                    "metadata": json.loads(row[11]) if row[11] else {},
                    "created_at": row[14],
                    "updated_at": row[15] or row[14]
                },
                "message": f"Found session context for {session_id}:{session_iter}",
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to load session context: {e}", exc_info=True)
            return {
                "success": False,
                "found_previous_context": False,
                "context": None,
                "message": str(e),
                "error": "Failed to load session context"
            }

    def _cleanup_old_memories_impl(self, days_old: int, dry_run: bool) -> dict[str, Any]:
        """
        Cleanup memories older than specified days.

        Args:
            days_old: Delete memories older than this many days
            dry_run: If True, only count what would be deleted (don't actually delete)

        Returns:
            Dict with cleanup statistics
        """
        try:
            from datetime import datetime, timedelta, timezone

            conn = self._get_connection()

            # Calculate cutoff date with overflow protection
            try:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
            except OverflowError:
                # If days_old is too large, use min datetime (will match nothing)
                cutoff_date = datetime.min.replace(tzinfo=timezone.utc).isoformat()

            # Find memories to delete
            to_delete = conn.execute("""
                SELECT id, created_at
                FROM session_memories
                WHERE created_at < ?
                ORDER BY created_at ASC
            """, (cutoff_date,)).fetchall()

            if not to_delete:
                conn.close()
                return {
                    "success": True,
                    "deleted_count": 0,
                    "oldest_deleted": None,
                    "newest_deleted": None,
                    "message": f"No memories found older than {days_old} days",
                    "error": None
                }

            oldest_deleted = to_delete[0][1]
            newest_deleted = to_delete[-1][1]
            count = len(to_delete)

            if not dry_run:
                # Actually delete memories and their chunks
                for row in to_delete:
                    memory_id = row[0]
                    # Delete associated chunks first (foreign keys may not be enabled)
                    conn.execute("DELETE FROM memory_chunks WHERE parent_id = ?", (memory_id,))
                    # Then delete the memory
                    conn.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))
                conn.commit()

            conn.close()

            action = "Would delete" if dry_run else "Deleted"
            return {
                "success": True,
                "deleted_count": count,
                "oldest_deleted": oldest_deleted,
                "newest_deleted": newest_deleted,
                "message": f"{action} {count} memories older than {days_old} days",
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}", exc_info=True)
            return {
                "success": False,
                "deleted_count": None,
                "oldest_deleted": None,
                "newest_deleted": None,
                "message": str(e),
                "error": "Cleanup failed"
            }
