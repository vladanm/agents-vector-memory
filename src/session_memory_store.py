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


        # Initialize operation modules
        self.storage = StorageOperations(self)
        self.search = SearchOperations(self)
        self.maintenance = MaintenanceOperations(self)
        self.chunking = ChunkingStorageOperations(self)

    def _init_schema(self):
        """Initialize database schema using migrations."""
        from .db_migrations import run_migrations
        run_migrations(self.db_path)


    @property
    def token_encoder(self):
        """Cached token encoder instance."""
        if not hasattr(self, '_token_encoder'):
            if TIKTOKEN_AVAILABLE:
                try:
                    self._token_encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._token_encoder = None
            else:
                self._token_encoder = None
        return self._token_encoder

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
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with WAL mode, optimizations, and sqlite-vec loaded.

        Returns:
            sqlite3.Connection: Database connection configured with:
                - WAL (Write-Ahead Logging) mode for better concurrency
                - NORMAL synchronous mode for performance
                - Foreign keys enabled for referential integrity
                - Memory-based temp storage for performance
                - Memory-mapped I/O for large databases
                - sqlite-vec extension loaded for vector search

        Note:
            This method creates a new connection each time it is called.
            The caller is responsible for closing the connection.
        """
        conn = sqlite3.connect(self.db_path)

        # Load sqlite-vec extension if available
        if SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                import sqlite_vec
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                pass  # Continue without vector search if it fails

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

        # Set synchronous mode to NORMAL (faster than FULL, still safe with WAL)
        conn.execute("PRAGMA synchronous=NORMAL")

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys=ON")

        # Use memory for temporary storage (faster)
        conn.execute("PRAGMA temp_store=MEMORY")

        # Enable memory-mapped I/O (faster for large databases)
        conn.execute("PRAGMA mmap_size=30000000000")

        return conn


    @property
    def chunker(self) -> 'DocumentChunker':
        """Lazy initialization of chunker"""
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def token_encoder(self) -> Any:
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

    def load_session_context_for_task(self, *args, **kwargs) -> dict[str, Any]:
        """Load session context for task."""
        return self.search.load_session_context_for_task(*args, **kwargs)

    def get_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Get memory by ID."""
        return self.storage.get_memory(*args, **kwargs)

    def get_session_stats(self, *args, **kwargs) -> dict[str, Any]:
        """Get session statistics."""
        return self.maintenance.get_session_stats(*args, **kwargs)

    def list_sessions(self, *args, **kwargs) -> dict[str, Any]:
        """List sessions."""
        return self.maintenance.list_sessions(*args, **kwargs)

    def reconstruct_document(self, *args, **kwargs) -> dict[str, Any]:
        """Reconstruct document from chunks."""
        return self.storage.reconstruct_document(*args, **kwargs)

    def write_document_to_file(self, *args, **kwargs) -> dict[str, Any]:
        """Write document to file."""
        return self.storage.write_document_to_file(*args, **kwargs)

    def delete_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Delete memory."""
        return self.storage.delete_memory(*args, **kwargs)

    # ======================
    # IMPLEMENTATION METHODS
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

            # Validate session_id (must not be empty)
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
                    "message": "session_id cannot be empty"
                }

            # Get memory type config
            config = get_memory_type_config(memory_type)

            # Determine if we should chunk (default based on memory type)
            if auto_chunk is None:
                auto_chunk = config.get('auto_chunk', False)

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
                if self.embedding_model and chunks:
                    chunk_texts = [chunk.content for chunk in chunks]
                    try:
                        embeddings = self.embedding_model.encode(chunk_texts, batch_size=32, show_progress_bar=False)
                        # Convert embeddings to bytes and attach to chunks
                        for i, chunk in enumerate(chunks):
                            chunk.embedding = embeddings[i].tobytes()
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

            finally:
                # CRITICAL: Always close connection, even if an error occurs
                conn.close()

        except Exception as e:
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

        For semantic search (query parameter), this requires embeddings.
        Without query, performs filtered listing.

        Args:
            memory_type: Filter by memory type
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            session_iter: Filter by specific iteration
            task_code: Filter by task code
            query: Semantic search query (requires embeddings)
            limit: Maximum results to return
            latest_first: Order by latest iteration/creation first

        Returns:
            Dict with search results
        """
        try:
            conn = self._get_connection()

            # Build WHERE clause
            where_conditions = []
            params = []

            if memory_type:
                where_conditions.append("memory_type = ?")
                params.append(memory_type)

            if agent_id:
                where_conditions.append("agent_id = ?")
                params.append(agent_id)

            if session_id:
                where_conditions.append("session_id = ?")
                params.append(session_id)

            if session_iter is not None:
                where_conditions.append("session_iter = ?")
                params.append(session_iter)

            if task_code:
                where_conditions.append("task_code = ?")
                params.append(task_code)

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # Build ORDER BY clause
            if latest_first:
                order_clause = "ORDER BY session_iter DESC, created_at DESC"
            else:
                order_clause = "ORDER BY session_iter ASC, created_at ASC"

            # Execute query
            params.append(limit)
            rows = conn.execute(f"""
                SELECT * FROM session_memories
                {where_clause}
                {order_clause}
                LIMIT ?
            """, params).fetchall()

            conn.close()

            # Format results
            results = []
            for row in rows:
                results.append({
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
                    "access_count": row[17],
                    "similarity": 2.0,  # Perfect match for filtered results
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
            return {
                "success": False,
                "results": [],
                "total_results": 0,
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
                "error": "Search failed",
                "message": str(e)
            }

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
        similarity_threshold: float = 0.7,
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
            similarity_threshold: Minimum similarity (0.0-1.0)
            auto_merge_threshold: For medium, merge if >=X siblings match

        Returns:
            Dict with search results at specified granularity
        """
        # For now, this is a placeholder that calls the filtered search
        # Full semantic search requires embedding infrastructure

        if granularity == "coarse":
            # Return full documents - CRITICAL FIX: Add granularity field
            result = self.search_memories(
                memory_type=memory_type,
                agent_id=agent_id,
                session_id=session_id,
                session_iter=session_iter,
                task_code=task_code,
                query=query,
                limit=limit
            )
            # Add granularity field to match GranularSearchResult schema
            result["granularity"] = "coarse"
            return result
        elif granularity == "fine":
            # Return individual chunks
            # TODO: Implement chunk-level search
            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "fine",
                "message": "Chunk-level search requires embedding infrastructure",
                "error": None
            }
        else:  # medium
            # Return section-level results
            # TODO: Implement section-level search with auto-merging
            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "medium",
                "message": "Section-level search requires embedding infrastructure",
                "error": None
            }

    def _expand_chunk_context_impl(
        self,
        memory_id: int,
        chunk_index: int,
        context_window: int = 2
    ) -> dict[str, Any]:
        """
        Expand chunk context by retrieving surrounding chunks.

        Args:
            memory_id: Parent memory ID
            chunk_index: Target chunk index
            context_window: Number of chunks before/after to retrieve

        Returns:
            Dict with target chunk and surrounding context
        """
        try:
            conn = self._get_connection()

            # Get target chunk
            target = conn.execute("""
                SELECT * FROM memory_chunks
                WHERE parent_id = ? AND chunk_index = ?
            """, (memory_id, chunk_index)).fetchone()

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
                    "message": f"No chunk found at index {chunk_index} for memory {memory_id}"
                }

            # Get surrounding chunks
            start_index = max(0, chunk_index - context_window)
            end_index = chunk_index + context_window

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
                    "chunk_index": chunk[2],
                    "content": chunk[3],
                    "chunk_type": chunk[4],
                    "header_path": chunk[8],
                    "level": chunk[9]
                })

            # Build expanded content
            expanded_content = "\n\n".join([c["content"] for c in all_chunks])

            return {
                "success": True,
                "memory_id": memory_id,
                "target_chunk_index": chunk_index,
                "context_window": context_window,
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
            current_task_code: Current task being worked on

        Returns:
            Dict with session context if task match found
        """
        try:
            conn = self._get_connection()

            # Look for previous session context with matching task_code
            rows = conn.execute("""
                SELECT * FROM session_memories
                WHERE memory_type = 'session_context'
                AND agent_id = ?
                AND session_id = ?
                AND task_code = ?
                ORDER BY session_iter DESC, created_at DESC
                LIMIT 1
            """, (agent_id, session_id, current_task_code)).fetchall()

            conn.close()

            if rows:
                row = rows[0]
                context = {
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
                    "created_at": row[14]
                }

                return {
                    "success": True,
                    "found_previous_context": True,
                    "context": context,
                    "message": f"Found previous context for task: {current_task_code}",
                    "error": None
                }
            else:
                return {
                    "success": True,
                    "found_previous_context": False,
                    "context": None,
                    "message": f"No previous context found for task: {current_task_code}",
                    "error": None
                }

        except Exception as e:
            return {
                "success": False,
                "found_previous_context": False,
                "context": None,
                "error": "Context loading failed",
                "message": str(e)
            }

    def _get_memory_impl(self, memory_id: int) -> dict[str, Any]:
        """Retrieve specific memory by ID."""
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
                    "message": f"No memory found with ID: {memory_id}"
                }

            memory = {
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
            }

            return {
                "success": True,
                "memory": memory,
                "error": None,
                "message": None
            }

        except Exception as e:
            return {
                "success": False,
                "memory": None,
                "error": "Retrieval failed",
                "message": str(e)
            }

    def _get_session_stats_impl(
        self,
        agent_id: str = None,
        session_id: str = None
    ) -> dict[str, Any]:
        """Get statistics about session memory usage."""
        try:
            conn = self._get_connection()

            # Build WHERE conditions for filtering
            where_conditions = []
            params = []

            if agent_id:
                where_conditions.append("agent_id = ?")
                params.append(agent_id)

            if session_id:
                where_conditions.append("session_id = ?")
                params.append(session_id)

            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

            # Get overall stats
            stats_query = f"""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    COUNT(DISTINCT agent_id) as unique_agents,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    COUNT(DISTINCT task_code) as unique_tasks,
                    MAX(session_iter) as max_session_iter,
                    AVG(LENGTH(content)) as avg_content_length,
                    SUM(access_count) as total_access_count
                FROM session_memories
                {where_clause}
            """

            stats_row = conn.execute(stats_query, params).fetchone()

            # Get memory type breakdown
            type_query = f"""
                SELECT memory_type, COUNT(*) as count
                FROM session_memories
                {where_clause}
                GROUP BY memory_type
                ORDER BY count DESC
            """

            type_rows = conn.execute(type_query, params).fetchall()

            conn.close()

            return {
                "success": True,
                "total_memories": stats_row[0],
                "memory_types": stats_row[1],
                "unique_agents": stats_row[2],
                "unique_sessions": stats_row[3],
                "unique_tasks": stats_row[4],
                "max_session_iter": stats_row[5] or 0,
                "avg_content_length": round(stats_row[6] or 0, 2),
                "total_access_count": stats_row[7] or 0,
                "memory_type_breakdown": {row[0]: row[1] for row in type_rows},
                "filters": {
                    "agent_id": agent_id,
                    "session_id": session_id
                },
                "error": None,
                "message": None
            }

        except Exception as e:
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
                "filters": None,
                "error": "Stats retrieval failed",
                "message": str(e)
            }

    def _list_sessions_impl(
        self,
        agent_id: str = None,
        limit: int = 20
    ) -> dict[str, Any]:
        """List recent sessions with basic info."""
        try:
            conn = self._get_connection()

            where_clause = "WHERE agent_id = ?" if agent_id else ""
            params = [agent_id] if agent_id else []
            params.append(limit)

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
                {where_clause}
                GROUP BY agent_id, session_id
                ORDER BY latest_activity DESC
                LIMIT ?
            """, params).fetchall()

            conn.close()

            sessions = []
            for row in rows:
                sessions.append({
                    "agent_id": row[0],
                    "session_id": row[1],
                    "memory_count": row[2],
                    "latest_iter": row[3],
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
                "message": None
            }

        except Exception as e:
            return {
                "success": False,
                "sessions": [],
                "total_sessions": 0,
                "agent_filter": agent_id,
                "limit": limit,
                "error": "Session listing failed",
                "message": str(e)
            }

    def _reconstruct_document_impl(self, memory_id: int) -> dict[str, Any]:
        """
        Reconstruct a document from its chunks.

        Args:
            memory_id: The ID of the parent memory to reconstruct

        Returns:
            Dict with reconstructed content and chunk info
        """
        try:
            conn = self._get_connection()

            # Get the parent memory
            parent = conn.execute("""
                SELECT content, title, memory_type, created_at
                FROM session_memories
                WHERE id = ?
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
                    "message": f"No memory found with ID: {memory_id}"
                }

            # Get all chunks for this memory
            # CRITICAL FIX: Select original_content (field index will be 23) to get clean content
            # without metadata headers that were added for embedding purposes
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
        Write a reconstructed document from memory to disk as a markdown file.

        Use this when documents are too large for MCP response (>20k tokens).
        After writing, use standard file read operations to access the content.

        Args:
            memory_id: The ID of the memory to reconstruct and write
            output_path: Absolute path where to write the file (auto-generated if None)
            include_metadata: Whether to include YAML frontmatter with metadata
            format: Output format ("markdown" or "plain")

        Returns:
            Dict with success status and file information
        """
        try:
            # ===== STEP 1: VALIDATE INPUTS =====
            if memory_id <= 0:
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": memory_id,
                    "created_at": None,
                    "message": None,
                    "error_code": "INVALID_PARAMETER",
                    "error_message": "memory_id must be a positive integer"
                }

            if format not in ["markdown", "plain"]:
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": None,
                    "created_at": None,
                    "message": None,
                    "error_code": "INVALID_PARAMETER",
                    "error_message": "format must be 'markdown' or 'plain'"
                }

            if output_path and not Path(output_path).is_absolute():
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": None,
                    "created_at": None,
                    "message": None,
                    "error_code": "INVALID_PATH",
                    "error_message": "output_path must be an absolute path"
                }

            # ===== STEP 2: FETCH MEMORY FROM DATABASE =====
            memory_result = self.get_memory(memory_id)

            if not memory_result.get("success"):
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": memory_id,
                    "created_at": None,
                    "message": None,
                    "error_code": "MEMORY_NOT_FOUND",
                    "error_message": f"Memory with ID {memory_id} does not exist"
                }

            memory = memory_result["memory"]

            # ===== STEP 3: RECONSTRUCT FULL DOCUMENT =====
            reconstruct_result = self.reconstruct_document(memory_id)

            if not reconstruct_result.get("success"):
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": memory_id,
                    "created_at": None,
                    "message": None,
                    "error_code": "RECONSTRUCTION_FAILED",
                    "error_message": f"Failed to reconstruct document: {reconstruct_result.get('message', 'Unknown error')}"
                }

            full_content = reconstruct_result["content"]

            # ===== STEP 4: ADD METADATA IF REQUESTED =====
            if include_metadata and YAML_AVAILABLE:
                # Generate YAML frontmatter
                frontmatter_data = {
                    "memory_id": memory["id"],
                    "title": memory.get("title") or "Untitled Document",
                    "memory_type": memory.get("memory_type"),
                    "created_at": memory.get("created_at"),
                    "updated_at": memory.get("updated_at"),
                    "session_id": memory.get("session_id"),
                    "agent_id": memory.get("agent_id"),
                    "task_code": memory.get("task_code"),
                    "tags": memory.get("tags", [])
                }

                # Remove None values
                frontmatter_data = {k: v for k, v in frontmatter_data.items() if v is not None}

                if memory.get("description"):
                    frontmatter_data["description"] = memory["description"]

                try:
                    frontmatter_yaml = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False)
                    frontmatter = f"---\n{frontmatter_yaml}---\n\n"
                    full_content = frontmatter + full_content
                except Exception:
                    # If YAML generation fails, continue without frontmatter
                    pass

            # ===== STEP 5: DETERMINE OUTPUT PATH =====
            if output_path is None:
                # Generate automatic temp path
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                filename = f"memory_{memory_id}_{timestamp}.md"

                temp_dir = Path(tempfile.gettempdir()) / "vector_memory"
                temp_dir.mkdir(parents=True, exist_ok=True)

                output_path = temp_dir / filename
            else:
                output_path = Path(output_path)

                # Validate provided path and create directory if needed
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return {
                        "success": False,
                        "file_path": None,
                        "file_size_bytes": None,
                        "file_size_human": None,
                        "estimated_tokens": None,
                        "memory_id": None,
                        "created_at": None,
                        "message": None,
                        "error_code": "PERMISSION_DENIED",
                        "error_message": f"Cannot create directory: {output_path.parent}"
                    }
                except OSError as e:
                    return {
                        "success": False,
                        "file_path": None,
                        "file_size_bytes": None,
                        "file_size_human": None,
                        "estimated_tokens": None,
                        "memory_id": None,
                        "created_at": None,
                        "message": None,
                        "error_code": "PERMISSION_DENIED",
                        "error_message": f"Cannot create directory: {str(e)}"
                    }

            # ===== STEP 6: WRITE TO FILE =====
            try:
                output_path.write_text(full_content, encoding="utf-8")
            except PermissionError:
                return {
                    "success": False,
                    "file_path": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "estimated_tokens": None,
                    "memory_id": None,
                    "created_at": None,
                    "message": None,
                    "error_code": "PERMISSION_DENIED",
                    "error_message": f"No write permission for path: {output_path}"
                }
            except OSError as e:
                if "No space left" in str(e) or "Disk quota exceeded" in str(e):
                    return {
                        "success": False,
                        "file_path": None,
                        "file_size_bytes": None,
                        "file_size_human": None,
                        "estimated_tokens": None,
                        "memory_id": None,
                        "created_at": None,
                        "message": None,
                        "error_code": "DISK_FULL",
                        "error_message": "Insufficient disk space"
                    }
                else:
                    return {
                        "success": False,
                        "file_path": None,
                        "file_size_bytes": None,
                        "file_size_human": None,
                        "estimated_tokens": None,
                        "memory_id": None,
                        "created_at": None,
                        "message": None,
                        "error_code": "WRITE_FAILED",
                        "error_message": f"Failed to write file: {str(e)}"
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
                    "message": None,
                    "error_code": "WRITE_FAILED",
                    "error_message": f"Unexpected error writing file: {str(e)}"
                }

            # ===== STEP 7: GATHER STATISTICS =====
            file_size_bytes = output_path.stat().st_size
            file_size_human = self._format_bytes_human_readable(file_size_bytes)
            estimated_tokens = self._estimate_tokens(full_content)

            # ===== STEP 8: RETURN SUCCESS =====
            return {
                "success": True,
                "file_path": str(output_path.absolute()),
                "file_size_bytes": file_size_bytes,
                "file_size_human": file_size_human,
                "estimated_tokens": estimated_tokens,
                "memory_id": memory_id,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "message": "Document successfully written to disk",
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
                "memory_id": memory_id,
                "created_at": None,
                "message": None,
                "error_code": "WRITE_FAILED",
                "error_message": f"Unexpected error: {str(e)}"
            }

    def _delete_memory_impl(self, memory_id: int) -> dict[str, Any]:
        """
        Delete a memory and all associated data.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            Dict with deletion status
        """
        try:
            conn = self._get_connection()

            try:
                # Check if memory exists
                existing = conn.execute(
                    "SELECT memory_type FROM session_memories WHERE id = ?",
                    (memory_id,)
                ).fetchone()

                if not existing:
                    return {
                        "success": False,
                        "memory_id": None,
                        "error": "Memory not found",
                        "message": f"No memory found with ID: {memory_id}"
                    }

                # Delete from session_memories (cascades to embeddings, chunks)
                conn.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))

                # Delete from vector search index
                conn.execute("DELETE FROM vec_session_search WHERE memory_id = ?", (memory_id,))

                conn.commit()

                return {
                    "success": True,
                    "memory_id": memory_id,
                    "message": f"Memory {memory_id} and all associated data deleted successfully",
                    "error": None
                }

            except Exception as delete_error:
                conn.rollback()
                return {
                    "success": False,
                    "memory_id": None,
                    "error": "Deletion failed",
                    "message": str(delete_error)
                }

            finally:
                # CRITICAL: Always close connection, even if an error occurs
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "memory_id": None,
                "error": "Deletion failed",
                "message": str(e)
            }
