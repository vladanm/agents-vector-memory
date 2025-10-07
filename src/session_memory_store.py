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
        Get database connection with extensions loaded.

        Configures busy_timeout to handle concurrent access from multiple processes.
        This is critical for MCP server usage where the server holds connections
        while clients make queries.

        Returns:
            sqlite3.Connection with busy_timeout=5000ms and extensions loaded
        """
        conn = sqlite3.connect(self.db_path)

        # STEP 1: Enable WAL mode for better concurrency
        # WAL allows readers to read while writer writes (no blocking)
        conn.execute("PRAGMA journal_mode=WAL")

        # STEP 2: Set synchronous to NORMAL (balance of safety and speed)
        conn.execute("PRAGMA synchronous=NORMAL")

        # STEP 3: Increase cache size to 64MB for performance
        conn.execute("PRAGMA cache_size=-64000")

        # STEP 4: Enable foreign keys for referential integrity
        conn.execute("PRAGMA foreign_keys=ON")

        # STEP 5: Set busy timeout to handle concurrent access
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

    def __enter__(self) -> 'SessionMemoryStore':
        """Context manager entry - no-op, connections managed per operation"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - no-op, connections managed per operation"""
        return False


    def _migrate_schema(self, conn: sqlite3.Connection) -> int:
        """
        Migrate database schema to latest version by adding missing columns.

        This method is idempotent - safe to run multiple times. It detects
        which columns are missing and only adds those, preserving all existing data.

        Args:
            conn: Active database connection

        Returns:
            Number of migrations applied (0 if schema already up-to-date)

        Raises:
            sqlite3.Error: If migration fails
        """
        try:
            # Check if memory_chunks table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
            )
            if not cursor.fetchone():
                # Table doesn't exist yet, will be created by CREATE TABLE statement
                return 0

            # Get existing columns
            cursor = conn.execute("PRAGMA table_info(memory_chunks)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Define required columns with their ALTER TABLE statements
            # Format: column_name: (alter_statement, column_type, default_value)
            required_columns = {
                'granularity_level': (
                    "ALTER TABLE memory_chunks ADD COLUMN granularity_level TEXT DEFAULT 'medium'",
                    "TEXT",
                    "'medium'"
                ),
                'parent_title': (
                    "ALTER TABLE memory_chunks ADD COLUMN parent_title TEXT DEFAULT NULL",
                    "TEXT",
                    "NULL"
                ),
                'section_hierarchy': (
                    "ALTER TABLE memory_chunks ADD COLUMN section_hierarchy TEXT DEFAULT NULL",
                    "TEXT",
                    "NULL"
                ),
                'chunk_position_ratio': (
                    "ALTER TABLE memory_chunks ADD COLUMN chunk_position_ratio REAL DEFAULT 0.5",
                    "REAL",
                    "0.5"
                ),
                'sibling_count': (
                    "ALTER TABLE memory_chunks ADD COLUMN sibling_count INTEGER DEFAULT 1",
                    "INTEGER",
                    "1"
                ),
                'depth_level': (
                    "ALTER TABLE memory_chunks ADD COLUMN depth_level INTEGER DEFAULT 0",
                    "INTEGER",
                    "0"
                ),
                'contains_code': (
                    "ALTER TABLE memory_chunks ADD COLUMN contains_code BOOLEAN DEFAULT 0",
                    "BOOLEAN",
                    "0"
                ),
                'contains_table': (
                    "ALTER TABLE memory_chunks ADD COLUMN contains_table BOOLEAN DEFAULT 0",
                    "BOOLEAN",
                    "0"
                ),
                'keywords': (
                    "ALTER TABLE memory_chunks ADD COLUMN keywords TEXT DEFAULT '[]'",
                    "TEXT",
                    "'[]'"
                ),
                'original_content': (
                    "ALTER TABLE memory_chunks ADD COLUMN original_content TEXT DEFAULT NULL",
                    "TEXT",
                    "NULL"
                ),
                'is_contextually_enriched': (
                    "ALTER TABLE memory_chunks ADD COLUMN is_contextually_enriched BOOLEAN DEFAULT 0",
                    "BOOLEAN",
                    "0"
                )
            }

            # Determine which columns need to be added
            migrations_to_apply = []
            for column_name, (alter_statement, col_type, default_val) in required_columns.items():
                if column_name not in existing_columns:
                    migrations_to_apply.append({
                        'column': column_name,
                        'statement': alter_statement,
                        'type': col_type,
                        'default': default_val
                    })

            # If no migrations needed, return early
            if not migrations_to_apply:
                logger.info("Database schema is up-to-date (no migrations needed)")
                return 0

            # Apply migrations
            logger.info(f"Applying {len(migrations_to_apply)} schema migration(s)...")

            for migration in migrations_to_apply:
                try:
                    conn.execute(migration['statement'])
                    logger.info(f"Added column: {migration['column']} ({migration['type']}, default={migration['default']})") 
                except sqlite3.Error as e:
                    # If column already exists (race condition), continue
                    if "duplicate column" in str(e).lower():
                        logger.warning(f"Column {migration['column']} already exists, skipping")
                        continue
                    else:
                        # Re-raise other errors
                        raise

            # Commit all migrations as a transaction
            conn.commit()

            # Verify migrations were applied
            cursor = conn.execute("PRAGMA table_info(memory_chunks)")
            updated_columns = {row[1] for row in cursor.fetchall()}

            # Check if all required columns now exist
            missing_after_migration = set(required_columns.keys()) - updated_columns
            if missing_after_migration:
                raise sqlite3.Error(
                    f"Migration incomplete: columns still missing after migration: {missing_after_migration}"
                )

            logger.info(f"Successfully applied {len(migrations_to_apply)} schema migration(s)")
            logger.info(f"Database schema updated from {len(existing_columns)} to {len(updated_columns)} columns")

            return len(migrations_to_apply)

        except sqlite3.Error as e:
            logger.error(f"Schema migration failed: {e}")
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema"""
        conn = self._get_connection()

        # Main session memories table
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
                tags TEXT,
                metadata TEXT,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)

        # Memory chunks table (for hierarchical chunking)
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
                created_at TEXT NOT NULL,
                parent_title TEXT DEFAULT NULL,
                section_hierarchy TEXT DEFAULT NULL,
                granularity_level TEXT DEFAULT 'medium',
                chunk_position_ratio REAL DEFAULT 0.5,
                sibling_count INTEGER DEFAULT 1,
                depth_level INTEGER DEFAULT 0,
                contains_code BOOLEAN DEFAULT 0,
                contains_table BOOLEAN DEFAULT 0,
                keywords TEXT DEFAULT '[]',
                original_content TEXT DEFAULT NULL,
                is_contextually_enriched BOOLEAN DEFAULT 0,
                embedding BLOB,
                FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE,
                UNIQUE(parent_id, chunk_index)
            )
        """)

        # ========================================
        # MIGRATION: Add missing columns to existing databases
        # ========================================
        try:
            migrations_applied = self._migrate_schema(conn)
            if migrations_applied > 0:
                logger.info(f"Database migration completed: {migrations_applied} column(s) added")
        except sqlite3.Error as e:
            logger.warning(f"Schema migration failed: {e}")
            logger.warning("This may cause issues with existing databases.")
            logger.warning("Please check database integrity and consider manual migration.")
            # Don't raise - allow server to continue (may fail later if schema incompatible)
        # ========================================

        # Vector search table using sqlite-vec
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_session_search (
                rowid INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                chunk_id INTEGER,
                embedding BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES session_memories(id) ON DELETE CASCADE
            )
        """)

        # Try to create vector index (will fail gracefully if vec0 not loaded)
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_session_index
                USING vec0(
                    rowid INTEGER PRIMARY KEY,
                    embedding FLOAT[1536]
                )
            """)
        except sqlite3.OperationalError:
            # Vector index unavailable - semantic search will be disabled
            pass

        # Indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent_session ON session_memories(agent_id, session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_session_iter ON session_memories(session_id, session_iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_task_code ON session_memories(task_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON session_memories(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_parent ON memory_chunks(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_granularity ON memory_chunks(granularity_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON memory_chunks(section_hierarchy)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_title ON memory_chunks(parent_title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_contains_code ON memory_chunks(contains_code)")

        conn.commit()
        conn.close()

    def store_memory(
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
                    "error": "Invalid memory type",
                    "message": f"Memory type must be one of: {VALID_MEMORY_TYPES}"
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
                    "message": f"Memory stored successfully with ID: {memory_id}"
                }

            finally:
                # CRITICAL: Always close connection, even if an error occurs
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "error": "Storage failed",
                "message": str(e)
            }

    def search_memories(
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
                    "title": row[7],
                    "description": row[8],
                    "tags": json.loads(row[9]) if row[9] else [],
                    "metadata": json.loads(row[10]) if row[10] else {},
                    "content_hash": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                    "accessed_at": row[14],
                    "access_count": row[15],
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
                "latest_first": latest_first
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Search failed",
                "message": str(e)
            }

    def search_with_granularity(
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
            # Return full documents
            return self.search_memories(
                memory_type=memory_type,
                agent_id=agent_id,
                session_id=session_id,
                session_iter=session_iter,
                task_code=task_code,
                query=query,
                limit=limit
            )
        elif granularity == "fine":
            # Return individual chunks
            # TODO: Implement chunk-level search
            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "fine",
                "message": "Chunk-level search requires embedding infrastructure"
            }
        else:  # medium
            # Return section-level results
            # TODO: Implement section-level search with auto-merging
            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "medium",
                "message": "Section-level search requires embedding infrastructure"
            }

    def expand_chunk_context(
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
                "chunks": all_chunks
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Context expansion failed",
                "message": str(e)
            }

    def load_session_context_for_task(
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
                    "title": row[7],
                    "description": row[8],
                    "tags": json.loads(row[9]) if row[9] else [],
                    "metadata": json.loads(row[10]) if row[10] else {},
                    "created_at": row[12]
                }

                return {
                    "success": True,
                    "found_previous_context": True,
                    "context": context,
                    "message": f"Found previous context for task: {current_task_code}"
                }
            else:
                return {
                    "success": True,
                    "found_previous_context": False,
                    "context": None,
                    "message": f"No previous context found for task: {current_task_code}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": "Context loading failed",
                "message": str(e)
            }

    def get_memory(self, memory_id: int) -> dict[str, Any]:
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
                "title": row[7],
                "description": row[8],
                "tags": json.loads(row[9]) if row[9] else [],
                "metadata": json.loads(row[10]) if row[10] else {},
                "content_hash": row[11],
                "created_at": row[12],
                "updated_at": row[13],
                "accessed_at": row[14],
                "access_count": row[15]
            }

            return {
                "success": True,
                "memory": memory
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Retrieval failed",
                "message": str(e)
            }

    def get_session_stats(
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
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Stats retrieval failed",
                "message": str(e)
            }

    def list_sessions(
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
                "limit": limit
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Session listing failed",
                "message": str(e)
            }

    def reconstruct_document(self, memory_id: int) -> dict[str, Any]:
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
                    "message": "No chunks found, returning original content"
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
                "message": f"Document reconstructed from {len(chunks)} chunks"
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Reconstruction failed",
                "message": str(e)
            }

    def write_document_to_file(
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
                    "error_code": "INVALID_PARAMETER",
                    "error_message": "memory_id must be a positive integer",
                    "memory_id": memory_id
                }

            if format not in ["markdown", "plain"]:
                return {
                    "success": False,
                    "error_code": "INVALID_PARAMETER",
                    "error_message": "format must be 'markdown' or 'plain'",
                    "format": format
                }

            if output_path and not Path(output_path).is_absolute():
                return {
                    "success": False,
                    "error_code": "INVALID_PATH",
                    "error_message": "output_path must be an absolute path",
                    "output_path": output_path
                }

            # ===== STEP 2: FETCH MEMORY FROM DATABASE =====
            memory_result = self.get_memory(memory_id)

            if not memory_result.get("success"):
                return {
                    "success": False,
                    "error_code": "MEMORY_NOT_FOUND",
                    "error_message": f"Memory with ID {memory_id} does not exist",
                    "memory_id": memory_id
                }

            memory = memory_result["memory"]

            # ===== STEP 3: RECONSTRUCT FULL DOCUMENT =====
            reconstruct_result = self.reconstruct_document(memory_id)

            if not reconstruct_result.get("success"):
                return {
                    "success": False,
                    "error_code": "RECONSTRUCTION_FAILED",
                    "error_message": f"Failed to reconstruct document: {reconstruct_result.get('message', 'Unknown error')}",
                    "memory_id": memory_id
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
                        "error_code": "PERMISSION_DENIED",
                        "error_message": f"Cannot create directory: {output_path.parent}",
                        "output_path": str(output_path)
                    }
                except OSError as e:
                    return {
                        "success": False,
                        "error_code": "PERMISSION_DENIED",
                        "error_message": f"Cannot create directory: {str(e)}",
                        "output_path": str(output_path)
                    }

            # ===== STEP 6: WRITE TO FILE =====
            try:
                output_path.write_text(full_content, encoding="utf-8")
            except PermissionError:
                return {
                    "success": False,
                    "error_code": "PERMISSION_DENIED",
                    "error_message": f"No write permission for path: {output_path}",
                    "output_path": str(output_path)
                }
            except OSError as e:
                if "No space left" in str(e) or "Disk quota exceeded" in str(e):
                    return {
                        "success": False,
                        "error_code": "DISK_FULL",
                        "error_message": "Insufficient disk space",
                        "output_path": str(output_path)
                    }
                else:
                    return {
                        "success": False,
                        "error_code": "WRITE_FAILED",
                        "error_message": f"Failed to write file: {str(e)}",
                        "output_path": str(output_path)
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error_code": "WRITE_FAILED",
                    "error_message": f"Unexpected error writing file: {str(e)}",
                    "output_path": str(output_path)
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
                "message": "Document successfully written to disk"
            }

        except Exception as e:
            return {
                "success": False,
                "error_code": "WRITE_FAILED",
                "error_message": f"Unexpected error: {str(e)}",
                "memory_id": memory_id
            }

    def delete_memory(self, memory_id: int) -> dict[str, Any]:
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
                    "message": f"Memory {memory_id} and all associated data deleted successfully"
                }

            except Exception as delete_error:
                conn.rollback()
                return {
                    "success": False,
                    "error": "Deletion failed",
                    "message": str(delete_error)
                }

            finally:
                # CRITICAL: Always close connection, even if an error occurs
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "error": "Deletion failed",
                "message": str(e)
            }
