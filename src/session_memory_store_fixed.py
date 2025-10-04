"""
Agent Session Memory Store
==========================

Core storage engine for agent session management with proper scoping,
ordering, and task continuity support.
"""

import sqlite3
import sqlite_vec
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# DEBUG: Log module load
with open("/tmp/vector_memory_debug.log", "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"session_memory_store.py LOADED at {datetime.now()}\n")
    f.write(f"{'='*60}\n")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .config import Config
from .security import (
    validate_agent_id, validate_session_id, validate_task_code,
    validate_memory_type, validate_content, validate_session_iter,
    validate_tags, generate_content_hash, SecurityError
)


class SessionMemoryStore:
    """
    Session-centric vector memory storage with agent scoping.
    """
    
    def __init__(self, db_path: Path, embedding_model_name: str = None):
        """
        Initialize session memory store.
        
        Args:
            db_path: Path to SQLite database file
            embedding_model_name: Name of embedding model to use
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model_name or Config.EMBEDDING_MODEL
        
        # Initialize database
        self._init_database()
        
        # Initialize embedding model (lazy loading)
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy load embedding model to reduce memory usage"""
        if self._embedding_model is None:
            # Import here to avoid circular dependencies
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with sqlite-vec enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Enable foreign key constraints for CASCADE DELETE
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _init_database(self) -> None:
        """Initialize database schema with session-centric design."""
        conn = self._get_connection()
        
        try:
            # Create main memory table with session scoping
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Create vector embeddings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_embeddings (
                    id INTEGER PRIMARY KEY,
                    memory_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES session_memories(id) ON DELETE CASCADE
                )
            """)
            
            # Create vector search index
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_session_search
                USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding float[{Config.EMBEDDING_DIM}]
                )
            """)

            # Create memory chunks table for document chunking support (NEW)
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
                    FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE,
                    UNIQUE(parent_id, chunk_index)
                )
            """)

            # Create chunk embeddings table for semantic search on chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id INTEGER PRIMARY KEY,
                    chunk_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
                )
            """)

            # Create vector search index for chunks
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunk_search
                USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{Config.EMBEDDING_DIM}]
                )
            """)

            # Create indexes for efficient scoped searches
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session ON session_memories(agent_id, session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session_iter ON session_memories(agent_id, session_id, session_iter)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_session_task ON session_memories(agent_id, session_id, task_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON session_memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON session_memories(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_iter ON session_memories(session_iter)")
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to initialize database: {e}")
        finally:
            conn.close()

    def _extract_yaml_frontmatter(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract YAML frontmatter from markdown content.

        Args:
            content: Content that may contain YAML frontmatter

        Returns:
            Tuple of (clean_content, frontmatter_dict)
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match and YAML_AVAILABLE:
            frontmatter_text = match.group(1)
            try:
                frontmatter_data = yaml.safe_load(frontmatter_text) or {}
                clean_content = content[match.end():]
                return clean_content, frontmatter_data
            except Exception:
                return content, {}
        return content, {}

    def _normalize_markdown(self, content: str) -> str:
        """
        Normalize markdown formatting for consistent storage.

        Args:
            content: Markdown content

        Returns:
            Normalized markdown content
        """
        # Remove excessive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Normalize header spacing
        content = re.sub(r'(#{1,6})\s+', r'\1 ', content)

        # Trim whitespace
        content = content.strip()

        return content


    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from content using simple frequency analysis.

        Args:
            content: Text content to analyze
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        # Remove markdown formatting and code blocks
        cleaned = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'`[^`]+`', '', cleaned)
        cleaned = re.sub(r'[#*_\[\](){}]', ' ', cleaned)

        # Extract words (lowercase, length > 3)
        words = re.findall(r'\b[a-z]{4,}\b', cleaned.lower())

        # Common stop words to exclude
        stop_words = {'this', 'that', 'with', 'from', 'have', 'will', 'your', 'they',
                      'been', 'were', 'their', 'what', 'when', 'where', 'which', 'who',
                      'about', 'after', 'before', 'could', 'should', 'would', 'there'}

        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]

        return keywords

    def _detect_code_blocks(self, content: str) -> bool:
        """
        Detect if content contains code blocks.

        Args:
            content: Content to check

        Returns:
            True if code blocks detected
        """
        # Check for fenced code blocks
        if '```' in content:
            return True

        # Check for indented code blocks (4+ spaces at line start)
        lines = content.split('\n')
        indented_lines = [line for line in lines if line.startswith('    ') and line.strip()]

        # If more than 2 consecutive indented lines, likely code
        return len(indented_lines) > 2

    def _detect_tables(self, content: str) -> bool:
        """
        Detect if content contains tables (markdown or HTML).

        Args:
            content: Content to check

        Returns:
            True if tables detected
        """
        # Markdown tables (pipe-separated)
        if '|' in content and content.count('|') > 3:
            lines = content.split('\n')
            table_lines = [line for line in lines if '|' in line]
            if len(table_lines) >= 2:  # At least header + one row
                return True

        # HTML tables
        if '<table' in content.lower() or '<tr' in content.lower():
            return True

        return False

    def _generate_document_summary(self, content: str, max_length: int = 200) -> str:
        """
        Generate a summary of document content.

        Args:
            content: Document content
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        # Remove code blocks and excess whitespace
        cleaned = re.sub(r'```.*?```', '[CODE]', content, flags=re.DOTALL)
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Extract first paragraph or sentences
        if len(cleaned) <= max_length:
            return cleaned

        # Try to break at sentence boundary
        truncated = cleaned[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')

        if last_period > max_length - 50:  # Period near end
            return cleaned[:last_period + 1]
        elif last_space > 0:  # Break at word
            return cleaned[:last_space] + '...'
        else:
            return truncated + '...'

    def _extract_document_structure(self, content: str) -> str:
        """
        Extract hierarchical structure from markdown content.

        Args:
            content: Markdown content

        Returns:
            JSON string representing document structure
        """
        lines = content.split('\n')
        structure = []

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.*)', line.strip())
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2)
                structure.append({
                    'level': level,
                    'text': text
                })

        return json.dumps(structure) if structure else None

    def _count_tokens_simple(self, text: str) -> int:
        """
        Simple token counting (fallback when tiktoken not available).

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        # Approximate: 1 token ≈ 0.75 words
        words = len(text.split())
        return int(words * 1.3)

    def _calculate_chunk_metadata(self, chunk, chunks: List, parent_title: str) -> Dict[str, Any]:
        """
        Calculate all metadata fields for a chunk.

        Args:
            chunk: ChunkEntry object
            chunks: All chunks in document
            parent_title: Title of parent memory

        Returns:
            Dictionary with all metadata fields
        """
        total_chunks = len(chunks)

        # Calculate position ratio
        chunk_position_ratio = (chunk.chunk_index + 1) / total_chunks if total_chunks > 0 else 0.5

        # Calculate sibling count (chunks at same header level)
        siblings = [c for c in chunks if c.header_path == chunk.header_path]
        sibling_count = len(siblings)

        # Depth level is same as chunk level
        depth_level = chunk.level if chunk.level is not None else 0

        # Content detection
        chunk_content = chunk.original_content if chunk.original_content else chunk.content
        contains_code = 1 if self._detect_code_blocks(chunk_content) else 0
        contains_table = 1 if self._detect_tables(chunk_content) else 0

        # Extract keywords
        keywords = self._extract_keywords(chunk_content, max_keywords=8)

        # Section hierarchy (same as header_path for now)
        section_hierarchy = chunk.header_path if chunk.header_path else None

        return {
            'parent_title': parent_title,
            'section_hierarchy': section_hierarchy,
            'chunk_position_ratio': chunk_position_ratio,
            'sibling_count': sibling_count,
            'depth_level': depth_level,
            'contains_code': contains_code,
            'contains_table': contains_table,
            'keywords': json.dumps(keywords)
        }


    def _store_chunks(self, parent_id: int, chunks: List, conn: sqlite3.Connection) -> None:
        """
        Store document chunks with embeddings in the memory_chunks table.

        Args:
            parent_id: Parent memory ID
            chunks: List of ChunkEntry objects
            conn: Database connection
        """
        from .memory_types import ChunkEntry

        now = datetime.now(timezone.utc).isoformat()

        # Generate embeddings for all chunks at once (batch processing)
        chunk_contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_contents)

        # Get parent memory title for chunk metadata
        parent_title = conn.execute(
            "SELECT title FROM session_memories WHERE id = ?", (parent_id,)
        ).fetchone()
        parent_title = parent_title[0] if parent_title and parent_title[0] else "Untitled"

        for i, chunk in enumerate(chunks):
            # Calculate all metadata fields
            chunk_meta = self._calculate_chunk_metadata(chunk, chunks, parent_title)

            # Insert chunk with ALL metadata fields
            cursor = conn.execute("""
                INSERT INTO memory_chunks (
                    parent_id, chunk_index, content, chunk_type,
                    start_char, end_char, token_count, header_path, level,
                    prev_chunk_id, next_chunk_id, content_hash, created_at,
                    original_content, is_contextually_enriched, granularity_level,
                    parent_title, section_hierarchy, chunk_position_ratio,
                    sibling_count, depth_level, contains_code, contains_table, keywords
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parent_id, chunk.chunk_index, chunk.content, chunk.chunk_type,
                chunk.start_char, chunk.end_char, chunk.token_count,
                chunk.header_path, chunk.level, chunk.prev_chunk_id,
                chunk.next_chunk_id, chunk.content_hash, now,
                chunk.original_content, chunk.is_contextually_enriched,
                chunk.granularity_level,
                chunk_meta['parent_title'], chunk_meta['section_hierarchy'],
                chunk_meta['chunk_position_ratio'], chunk_meta['sibling_count'],
                chunk_meta['depth_level'], chunk_meta['contains_code'],
                chunk_meta['contains_table'], chunk_meta['keywords']
            ))

            chunk_id = cursor.lastrowid
            embedding = embeddings[i]

            # Store chunk embedding
            conn.execute("""
                INSERT INTO chunk_embeddings (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk_id, embedding.tobytes()))

            # Store in vector search index
            conn.execute("""
                INSERT INTO vec_chunk_search (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk_id, embedding.tobytes()))

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
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        auto_chunk: bool = False
    ) -> Dict[str, Any]:
        """
        Store memory with session scoping and optional document chunking.

        Args:
            memory_type: Type of memory (session_context, input_prompt, etc.)
            agent_id: Agent identifier ("main" or "specialized-agent")
            session_id: Session identifier
            content: Memory content
            session_iter: Session iteration number
            task_code: Task identifier (optional)
            title: Memory title
            description: Brief description
            tags: List of tags
            metadata: Additional metadata
            auto_chunk: Enable automatic document chunking (default: False)

        Returns:
            Dict with success status and memory details
        """
        try:
            # Validate inputs
            memory_type = validate_memory_type(memory_type)
            agent_id = validate_agent_id(agent_id)
            session_id = validate_session_id(session_id)
            content = validate_content(content)
            session_iter = validate_session_iter(session_iter)
            task_code = validate_task_code(task_code) if task_code else None
            tags = validate_tags(tags or [])
            
            # Generate content hash for deduplication
            content_hash = generate_content_hash(f"{memory_type}:{agent_id}:{session_id}:{content}")
            
            # Create embedding
            embedding = self.embedding_model.encode([content])[0]
            
            # Current timestamp
            now = datetime.now(timezone.utc).isoformat()
            
            conn = self._get_connection()
            
            try:
                # Check for duplicate
                existing = conn.execute(
                    "SELECT id FROM session_memories WHERE content_hash = ?",
                    (content_hash,)
                ).fetchone()
                
                if existing:
                    return {
                        "success": False,
                        "error": "Duplicate content",
                        "message": f"Memory already exists with ID: {existing[0]}",
                        "existing_id": existing[0]
                    }
                
                # Calculate document-level metadata
                document_summary = self._generate_document_summary(content, max_length=200)
                estimated_tokens = self._count_tokens_simple(content)
                document_structure = self._extract_document_structure(content)
                chunk_strategy = "hierarchical" if auto_chunk else "none"

                # Insert memory with populated metadata fields
                cursor = conn.execute("""
                    INSERT INTO session_memories (
                        memory_type, agent_id, session_id, session_iter, task_code,
                        content, title, description, tags, metadata,
                        content_hash, created_at, updated_at,
                        document_summary, estimated_tokens, document_structure, chunk_strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_type, agent_id, session_id, session_iter, task_code,
                    content, title, description, json.dumps(tags), 
                    json.dumps(metadata or {}), content_hash, now, now,
                    document_summary, estimated_tokens, document_structure, chunk_strategy
                ))
                
                memory_id = cursor.lastrowid
                
                # Store embedding
                conn.execute("""
                    INSERT INTO session_embeddings (memory_id, embedding)
                    VALUES (?, ?)
                """, (memory_id, embedding.tobytes()))
                
                # Store in vector search index
                conn.execute("""
                    INSERT INTO vec_session_search (memory_id, embedding)
                    VALUES (?, ?)
                """, (memory_id, embedding.tobytes()))

                # Handle document chunking if enabled
                chunk_count = 0

                # DEBUG: Write to file
                debug_log = "/tmp/vector_memory_debug.log"
                with open(debug_log, "a") as f:
                    f.write(f"\n=== store_memory called ===\n")
                    f.write(f"memory_id: {memory_id}\n")
                    f.write(f"auto_chunk: {auto_chunk}\n")
                    f.write(f"memory_type: {memory_type}\n")
                    f.write(f"content_length: {len(content)}\n")

                if auto_chunk:
                    with open(debug_log, "a") as f:
                        f.write(f"ENTERING auto_chunk block!\n")
                    try:
                        from .chunking import DocumentChunker
                        from .memory_types import get_memory_type_config

                        # Get chunking config for memory type
                        type_config = get_memory_type_config(memory_type)

                        # Create chunker and process content
                        chunker = DocumentChunker()
                        chunk_metadata = {
                            "memory_type": memory_type,
                            "title": title or "Untitled Document"
                        }
                        if metadata:
                            chunk_metadata.update(metadata)

                        with open(debug_log, "a") as f:
                            f.write(f"About to call chunk_document with title: {title}\n")
                        chunks = chunker.chunk_document(content, memory_id, chunk_metadata)
                        with open(debug_log, "a") as f:
                            f.write(f"Chunking returned {len(chunks) if chunks else 0} chunks\n")

                        # Store chunks
                        if chunks:
                            with open(debug_log, "a") as f:
                                f.write(f"Storing {len(chunks)} chunks\n")
                            self._store_chunks(memory_id, chunks, conn)
                            chunk_count = len(chunks)
                            with open(debug_log, "a") as f:
                                f.write(f"Successfully stored {chunk_count} chunks\n")
                        else:
                            with open(debug_log, "a") as f:
                                f.write(f"No chunks to store (chunks list empty)\n")

                        # Cleanup chunker resources
                        chunker.cleanup_chunks()

                    except Exception as e:
                        # Log warning but don't fail the entire operation
                        import traceback
                        with open(debug_log, "a") as f:
                            f.write(f"EXCEPTION in chunking: {e}\n")
                            f.write(traceback.format_exc())

                conn.commit()

                result = {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code,
                    "content_hash": content_hash,
                    "created_at": now,
                    "message": f"Memory stored successfully with ID: {memory_id}"
                }

                if chunk_count > 0:
                    result["chunks_stored"] = chunk_count

                return result
                
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()
                
        except SecurityError as e:
            return {
                "success": False,
                "error": "Validation failed",
                "message": str(e)
            }
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
        limit: int = 10,
        latest_first: bool = True,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search memories with proper scoping and ordering.
        
        Args:
            memory_type: Filter by memory type
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            session_iter: Filter by specific iteration
            task_code: Filter by task code
            query: Semantic search query (optional)
            limit: Maximum results
            latest_first: Order by latest iteration/creation first
            similarity_threshold: Minimum similarity for semantic search
            
        Returns:
            Dict with search results ordered properly
        """
        try:
            conn = self._get_connection()
            
            # Build WHERE conditions
            where_conditions = []
            params = []
            
            if memory_type:
                where_conditions.append("m.memory_type = ?")
                params.append(memory_type)
            
            if agent_id:
                where_conditions.append("m.agent_id = ?")
                params.append(agent_id)
            
            if session_id:
                where_conditions.append("m.session_id = ?")
                params.append(session_id)
            
            if session_iter is not None:
                where_conditions.append("m.session_iter = ?")
                params.append(session_iter)
            
            if task_code:
                where_conditions.append("m.task_code = ?")
                params.append(task_code)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Handle semantic search vs. scoped search
            if query and query.strip():
                # Generate query embedding for semantic search
                query_embedding = self.embedding_model.encode([query.strip()])[0]

                # Build WHERE clause for the main query (not the subquery)
                where_clause_main = where_clause.replace("WHERE ", "AND ") if where_clause else ""

                distance_threshold = 1.8  # Allow semantic matches, filter out very dissimilar content

                # Search both document embeddings and chunk embeddings
                # 1. Search document embeddings
                doc_vector_query = f"""
                    SELECT m.*, v.distance, 'document' as source_type, NULL as chunk_index
                    FROM session_memories m
                    JOIN (
                        SELECT memory_id, distance
                        FROM vec_session_search
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance ASC
                    ) v ON m.id = v.memory_id
                    WHERE v.distance < ?
                    {where_clause_main}
                """

                # 2. Search chunk embeddings
                # Return chunk content instead of parent content to avoid duplication
                chunk_vector_query = f"""
                    SELECT
                        m.id, m.memory_type, m.agent_id, m.session_id,
                        m.session_iter, m.task_code,
                        mc.content as content,  -- Use chunk content, not parent content
                        m.title, m.description, m.tags, m.metadata,
                        m.content_hash, m.created_at, m.updated_at,
                        m.accessed_at, m.access_count,
                        v.distance, 'chunk' as source_type, mc.chunk_index
                    FROM session_memories m
                    JOIN memory_chunks mc ON m.id = mc.parent_id
                    JOIN (
                        SELECT chunk_id, distance
                        FROM vec_chunk_search
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance ASC
                    ) v ON mc.id = v.chunk_id
                    WHERE v.distance < ?
                    {where_clause_main}
                """

                # Combine both searches with UNION ALL
                combined_query = f"""
                    SELECT * FROM (
                        {doc_vector_query}
                        UNION ALL
                        {chunk_vector_query}
                    )
                    ORDER BY distance ASC
                    LIMIT ?
                """

                # Parameters: embedding, k, distance_threshold for both queries, then final limit
                final_params = (
                    [query_embedding.tobytes(), limit, distance_threshold] + params +
                    [query_embedding.tobytes(), limit * 2, distance_threshold] + params +
                    [limit]
                )
                rows = conn.execute(combined_query, final_params).fetchall()
                
            else:
                # Pure scoped search without semantic filtering
                order_clause = "ORDER BY m.created_at DESC"
                if latest_first:
                    order_clause = "ORDER BY m.session_iter DESC, m.created_at DESC"
                
                final_query = f"""
                    SELECT m.*, 0.0 as distance
                    FROM session_memories m
                    {where_clause}
                    {order_clause}
                    LIMIT ?
                """
                params.append(limit)
                
                rows = conn.execute(final_query, params).fetchall()
            
            # Format results
            results = []
            for row in rows:
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
                    "access_count": row[15],
                    "similarity": max(0.0, 2.0 - row[16]) if len(row) > 16 else 1.0  # Convert distance to similarity (sqlite-vec L2 distance)
                }

                # Add source information if available (from semantic search)
                if len(row) > 17:
                    memory["source_type"] = row[17]  # 'document' or 'chunk'
                    if row[18] is not None:  # chunk_index
                        memory["chunk_index"] = row[18]

                results.append(memory)

            # Filter results by similarity threshold (only for semantic search)
            if query and query.strip():
                # Filter out results with similarity below threshold
                # Note: similarity = 2.0 - distance (sqlite-vec returns L2 distance)
                # Lower distance = higher similarity
                filtered_results = [
                    result for result in results
                    if result['similarity'] >= similarity_threshold
                ]
                results = filtered_results

            # Update access counts
            if results:
                memory_ids = [r["id"] for r in results]
                placeholders = ",".join("?" * len(memory_ids))
                conn.execute(f"""
                    UPDATE session_memories 
                    SET access_count = access_count + 1, accessed_at = ?
                    WHERE id IN ({placeholders})
                """, [datetime.now(timezone.utc).isoformat()] + memory_ids)
                conn.commit()
            
            conn.close()
            
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
        query: str,
        memory_type: str,
        granularity: str,
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        auto_merge_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Three-tier granularity search for knowledge_base and reports.

        Args:
            query: Semantic search query
            memory_type: Memory type (knowledge_base or reports)
            granularity: 'fine' (<400 tokens), 'medium' (400-1200), 'coarse' (>1200)
            agent_id: Filter by agent ID (optional)
            session_id: Filter by session ID (optional)
            session_iter: Filter by iteration (optional)
            task_code: Filter by task code (optional)
            limit: Maximum results
            similarity_threshold: Minimum similarity score (0.0-1.0)
            auto_merge_threshold: For medium search, auto-merge if ≥60% siblings match

        Returns:
            Dict with search results at specified granularity
        """
        try:
            valid_granularities = ['fine', 'medium', 'coarse']
            if granularity not in valid_granularities:
                return {
                    "success": False,
                    "error": f"Invalid granularity. Must be one of: {', '.join(valid_granularities)}"
                }

            conn = self._get_connection()
            query_embedding = self.embedding_model.encode([query.strip()])[0]

            # Build WHERE conditions for filters
            where_conditions = ["m.memory_type = ?"]
            params = [memory_type]

            if agent_id:
                where_conditions.append("m.agent_id = ?")
                params.append(agent_id)

            if session_id:
                where_conditions.append("m.session_id = ?")
                params.append(session_id)

            if session_iter is not None:
                where_conditions.append("m.session_iter = ?")
                params.append(session_iter)

            if task_code:
                where_conditions.append("m.task_code = ?")
                params.append(task_code)

            where_clause_main = "AND " + " AND ".join(where_conditions)
            distance_threshold = 2.0 - similarity_threshold  # Convert similarity to distance

            if granularity == 'coarse':
                # Search full documents only
                vector_query = f"""
                    SELECT m.*, v.distance, 'document' as source_type,
                           NULL as chunk_index, NULL as chunk_content
                    FROM session_memories m
                    JOIN (
                        SELECT memory_id, distance
                        FROM vec_session_search
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance ASC
                    ) v ON m.id = v.memory_id
                    WHERE v.distance < ?
                    {where_clause_main}
                    ORDER BY v.distance ASC
                    LIMIT ?
                """
                final_params = [query_embedding.tobytes(), limit * 2, distance_threshold] + params + [limit]

            else:
                # Fine or medium: search ALL chunks (NO granularity_level filtering!)
                # Granularity controls expansion behavior, not what we search

                vector_query = f"""
                    SELECT
                        m.id, m.memory_type, m.agent_id, m.session_id,
                        m.session_iter, m.task_code,
                        m.title, m.description, m.tags, m.metadata,
                        m.content_hash, m.created_at, m.updated_at,
                        m.accessed_at, m.access_count,
                        v.distance, mc.chunk_index, mc.id as chunk_id,
                        COALESCE(mc.original_content, mc.content) as chunk_content,
                        mc.header_path, mc.token_count, mc.sibling_count,
                        mc.section_hierarchy, mc.chunk_position_ratio
                    FROM session_memories m
                    JOIN memory_chunks mc ON m.id = mc.parent_id
                    JOIN (
                        SELECT chunk_id, distance
                        FROM vec_chunk_search
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance ASC
                    ) v ON mc.id = v.chunk_id
                    WHERE v.distance < ?
                    {where_clause_main}
                    ORDER BY v.distance ASC
                    LIMIT ?
                """
                final_params = [query_embedding.tobytes(), limit * 3, distance_threshold] + params + [limit * 2]

            rows = conn.execute(vector_query, final_params).fetchall()

            # Handle coarse search (full documents)
            if granularity == 'coarse':
                results = []
                for row in rows:
                    # Column order from SELECT m.*, v.distance, 'document', NULL, NULL
                    # m.*: id(0), memory_type(1), agent_id(2), session_id(3), session_iter(4),
                    #      task_code(5), content(6), title(7), description(8), tags(9), metadata(10),
                    #      content_hash(11), created_at(12), updated_at(13), accessed_at(14), access_count(15),
                    #      document_summary(16), estimated_tokens(17), chunk_strategy(18)
                    # Then: distance(19), source_type(20), chunk_index(21), chunk_content(22)
                    result = {
                        "memory_id": row[0],
                        "memory_type": row[1],
                        "agent_id": row[2],
                        "session_id": row[3],
                        "session_iter": row[4],
                        "task_code": row[5],
                        "content": row[6],  # Full document content
                        "title": row[7],
                        "description": row[8],
                        "tags": json.loads(row[9]) if row[9] else [],
                        "metadata": json.loads(row[10]) if row[10] else {},
                        "content_hash": row[11],
                        "created_at": row[12],
                        "updated_at": row[13],
                        "accessed_at": row[14],
                        "access_count": row[15],
                        "similarity": max(0.0, 2.0 - row[19]) if row[19] is not None else 0.0,
                        "source_type": "document",
                        "granularity": "coarse"
                    }
                    results.append(result)
            else:
                # Format matched chunks for fine/medium
                matched_chunks = []
                for row in rows:
                    chunk = {
                        "memory_id": row[0],
                        "memory_type": row[1],
                        "agent_id": row[2],
                        "session_id": row[3],
                        "session_iter": row[4],
                        "task_code": row[5],
                        "title": row[6],
                        "description": row[7],
                        "tags": json.loads(row[8]) if row[8] else [],
                        "metadata": json.loads(row[9]) if row[9] else {},
                        "content_hash": row[10],
                        "created_at": row[11],
                        "updated_at": row[12],
                        "accessed_at": row[13],
                        "access_count": row[14],
                        "similarity": max(0.0, 2.0 - row[15]) if row[15] is not None else 0.0,
                        "chunk_index": row[16],
                        "chunk_id": row[17],
                        "chunk_content": row[18],
                        "header_path": row[19],
                        "token_count": row[20],
                        "sibling_count": row[21],
                        "section_hierarchy": row[22],
                        "chunk_position_ratio": row[23],
                        "source_type": "chunk",
                        "granularity": granularity
                    }
                    matched_chunks.append(chunk)

                # Apply granularity-specific behavior
                if granularity == 'fine':
                    # Fine: Return chunks as-is (no expansion)
                    results = []
                    for chunk in matched_chunks[:limit]:
                        result = chunk.copy()
                        result["content"] = chunk["chunk_content"]  # Content = actual chunk
                        results.append(result)

                elif granularity == 'medium':
                    # Medium: Expand to sections (auto-merge siblings)
                    results = self._expand_to_sections(matched_chunks, auto_merge_threshold, conn, limit)

            conn.close()

            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "granularity": granularity,
                "filters": {
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code
                },
                "similarity_threshold": similarity_threshold
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Granularity search failed",
                "message": str(e)
            }

    def _expand_to_sections(
        self,
        matched_chunks: List[Dict],
        merge_threshold: float,
        conn: sqlite3.Connection,
        limit: int
    ) -> List[Dict]:
        """
        Expand matched chunks to include their siblings (section-level context).

        Logic:
        1. Group matched chunks by (parent_id, header_path)
        2. For each section, fetch ALL sibling chunks
        3. Merge all siblings into section-level content
        4. Return expanded sections (limit applied)

        Args:
            matched_chunks: List of matching chunks from vector search
            merge_threshold: Unused (kept for compatibility)
            conn: Database connection
            limit: Maximum results to return

        Returns:
            List of expanded section results
        """
        # Group matched chunks by section
        sections = {}
        for chunk in matched_chunks:
            parent_id = chunk["memory_id"]
            section_path = chunk.get("header_path", "")
            key = (parent_id, section_path)

            if key not in sections:
                sections[key] = {
                    "matched_chunks": [],
                    "representative": chunk  # Keep one chunk as template
                }
            sections[key]["matched_chunks"].append(chunk)

        # Expand each section to include all siblings
        expanded_results = []
        for (parent_id, section_path), section_data in sections.items():
            # Fetch ALL chunks from this section
            all_section_chunks = conn.execute("""
                SELECT chunk_index, COALESCE(original_content, content) as content, token_count
                FROM memory_chunks
                WHERE parent_id = ? AND header_path = ?
                ORDER BY chunk_index ASC
            """, (parent_id, section_path)).fetchall()

            if not all_section_chunks:
                continue

            # Merge all sibling chunks
            merged_content = "\n\n".join([chunk[1] for chunk in all_section_chunks])
            total_tokens = sum([chunk[2] for chunk in all_section_chunks])

            # Create expanded result
            representative = section_data["representative"]
            result = {
                "memory_id": parent_id,
                "memory_type": representative["memory_type"],
                "agent_id": representative["agent_id"],
                "session_id": representative["session_id"],
                "session_iter": representative["session_iter"],
                "task_code": representative["task_code"],
                "title": representative["title"],
                "description": representative["description"],
                "tags": representative["tags"],
                "metadata": representative["metadata"],
                "content_hash": representative["content_hash"],
                "created_at": representative["created_at"],
                "updated_at": representative["updated_at"],
                "accessed_at": representative["accessed_at"],
                "access_count": representative["access_count"],
                "similarity": representative["similarity"],
                "section_content": merged_content,  # Merged section content from all sibling chunks
                "source_type": "expanded_section",
                "granularity": "medium",
                "header_path": section_path,
                "matched_chunk_count": len(section_data["matched_chunks"]),
                "total_chunk_count": len(all_section_chunks),
                "merged": True,
                "token_count": total_tokens
            }
            expanded_results.append(result)

        # Sort by similarity and limit
        expanded_results.sort(key=lambda x: x["similarity"], reverse=True)
        return expanded_results[:limit]

    def _auto_merge_medium_chunks(
        self,
        results: List[Dict],
        merge_threshold: float,
        conn: sqlite3.Connection
    ) -> List[Dict]:
        """
        Auto-merge medium chunks if ≥60% of siblings match.

        Returns section-level content when majority of section chunks match.
        """
        # Group results by parent_id and header_path (section grouping)
        sections = {}
        for result in results:
            if result.get("source_type") != "chunk":
                continue

            parent_id = result["memory_id"]
            # Use header_path for section grouping (section_hierarchy may be NULL)
            section = result.get("header_path", "")
            key = (parent_id, section)

            if key not in sections:
                sections[key] = {
                    "results": [],
                    "sibling_count": result.get("sibling_count", 1)
                }
            sections[key]["results"].append(result)

        # Check each section for merge opportunity
        merged_results = []
        processed_keys = set()

        for result in results:
            if result.get("source_type") != "chunk":
                merged_results.append(result)
                continue

            parent_id = result["memory_id"]
            # Use header_path for section grouping (section_hierarchy may be NULL)
            section = result.get("header_path", "")
            key = (parent_id, section)

            if key in processed_keys:
                continue

            section_data = sections.get(key)
            if not section_data:
                merged_results.append(result)
                continue

            matched_count = len(section_data["results"])
            sibling_count = section_data["sibling_count"]
            match_ratio = matched_count / sibling_count if sibling_count > 0 else 0

            if match_ratio >= merge_threshold:
                # Merge: fetch all sibling chunks and combine
                chunk_indices = [r["chunk_index"] for r in section_data["results"]]
                placeholders = ",".join("?" * len(chunk_indices))

                all_chunks = conn.execute(f"""
                    SELECT chunk_index, COALESCE(original_content, content) as content
                    FROM memory_chunks
                    WHERE parent_id = ? AND header_path = ?
                    ORDER BY chunk_index
                """, [parent_id, section]).fetchall()

                merged_content = "\n\n".join(chunk[1] for chunk in all_chunks)

                # Create merged result
                merged_result = result.copy()
                merged_result["chunk_content"] = merged_content
                merged_result["merged"] = True
                merged_result["merged_chunk_count"] = len(all_chunks)
                merged_result["match_ratio"] = match_ratio

                merged_results.append(merged_result)
                processed_keys.add(key)
            else:
                # Keep individual chunks
                for r in section_data["results"]:
                    merged_results.append(r)
                processed_keys.add(key)

        return merged_results

    def expand_chunk_context(
        self,
        memory_id: int,
        chunk_index: int,
        context_window: int = 2
    ) -> Dict[str, Any]:
        """
        Expand chunk context by retrieving surrounding sibling chunks.

        Args:
            memory_id: Parent memory ID
            chunk_index: Index of the target chunk
            context_window: Number of chunks before/after to include (default: 2)

        Returns:
            Dict with expanded context including prev/current/next chunks
        """
        try:
            conn = self._get_connection()

            # Get target chunk
            target_chunk = conn.execute("""
                SELECT
                    id, chunk_index, COALESCE(original_content, content) as content,
                    header_path, section_hierarchy, token_count,
                    chunk_type, level, granularity_level
                FROM memory_chunks
                WHERE parent_id = ? AND chunk_index = ?
            """, (memory_id, chunk_index)).fetchone()

            if not target_chunk:
                return {
                    "success": False,
                    "error": "Chunk not found"
                }

            # Get surrounding chunks
            min_index = max(0, chunk_index - context_window)
            max_index = chunk_index + context_window

            surrounding_chunks = conn.execute("""
                SELECT
                    chunk_index, COALESCE(original_content, content) as content,
                    header_path, token_count, chunk_type
                FROM memory_chunks
                WHERE parent_id = ? AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
            """, (memory_id, min_index, max_index)).fetchall()

            # Get parent metadata
            parent_info = conn.execute("""
                SELECT title, memory_type, description
                FROM session_memories
                WHERE id = ?
            """, (memory_id,)).fetchone()

            conn.close()

            # Build context
            previous_chunks = []
            next_chunks = []
            current_chunk = None

            for chunk in surrounding_chunks:
                chunk_dict = {
                    "chunk_index": chunk[0],
                    "content": chunk[1],
                    "header_path": chunk[2],
                    "token_count": chunk[3],
                    "chunk_type": chunk[4]
                }

                if chunk[0] < chunk_index:
                    previous_chunks.append(chunk_dict)
                elif chunk[0] == chunk_index:
                    current_chunk = chunk_dict
                else:
                    next_chunks.append(chunk_dict)

            return {
                "success": True,
                "memory_id": memory_id,
                "parent_title": parent_info[0] if parent_info else None,
                "memory_type": parent_info[1] if parent_info else None,
                "target_chunk": {
                    "chunk_index": target_chunk[1],
                    "content": target_chunk[2],
                    "header_path": target_chunk[3],
                    "section_hierarchy": target_chunk[4],
                    "token_count": target_chunk[5],
                    "chunk_type": target_chunk[6],
                    "level": target_chunk[7],
                    "granularity_level": target_chunk[8]
                },
                "previous_chunks": previous_chunks,
                "next_chunks": next_chunks,
                "context_window": context_window,
                "expanded_content": "\n\n---\n\n".join([
                    *[c["content"] for c in previous_chunks],
                    current_chunk["content"] if current_chunk else "",
                    *[c["content"] for c in next_chunks]
                ])
            }

        except Exception as e:
            return {
                "success": False,
                "error": "Failed to expand chunk context",
                "message": str(e)
            }

    def load_session_context_for_task(
        self,
        agent_id: str,
        session_id: str, 
        current_task_code: str
    ) -> Dict[str, Any]:
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
    
    def get_memory(self, memory_id: int) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

    def reconstruct_document(self, memory_id: int) -> Dict[str, Any]:
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
            chunks = conn.execute("""
                SELECT chunk_index, content, chunk_type, header_path, level
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

            # Reconstruct from chunks
            reconstructed_parts = []
            for chunk in chunks:
                reconstructed_parts.append(chunk[1])  # content

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

    def delete_memory(self, memory_id: int) -> Dict[str, Any]:
        """
        Delete a memory and all associated data.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            Dict with deletion status
        """
        try:
            conn = self._get_connection()

            # Check if memory exists
            existing = conn.execute(
                "SELECT memory_type FROM session_memories WHERE id = ?",
                (memory_id,)
            ).fetchone()

            if not existing:
                conn.close()
                return {
                    "success": False,
                    "error": "Memory not found",
                    "message": f"No memory found with ID: {memory_id}"
                }

            try:
                # Delete from session_memories (cascades to embeddings, chunks)
                conn.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))

                # Delete from vector search index
                conn.execute("DELETE FROM vec_session_search WHERE memory_id = ?", (memory_id,))

                conn.commit()

                return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": existing[0],
                    "message": f"Memory {memory_id} deleted successfully"
                }

            except Exception as e:
                conn.rollback()
                raise

            finally:
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "error": "Deletion failed",
                "message": str(e)
            }

    def cleanup_old_memories(self, older_than_days: int = 30, memory_type: str = None) -> Dict[str, Any]:
        """
        Clean up old memories older than specified days.

        Args:
            older_than_days: Delete memories older than this many days
            memory_type: Optional memory type filter

        Returns:
            Dict with cleanup statistics
        """
        try:
            from datetime import timedelta

            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()

            conn = self._get_connection()

            try:
                # Build query
                where_clause = "WHERE created_at < ?"
                params = [cutoff_date]

                if memory_type:
                    where_clause += " AND memory_type = ?"
                    params.append(memory_type)

                # Get IDs to delete
                rows = conn.execute(f"""
                    SELECT id FROM session_memories {where_clause}
                """, params).fetchall()

                deleted_ids = [row[0] for row in rows]

                if not deleted_ids:
                    conn.close()
                    return {
                        "success": True,
                        "deleted_count": 0,
                        "message": "No old memories found to clean up"
                    }

                # Delete memories (cascades to embeddings and chunks)
                conn.execute(f"""
                    DELETE FROM session_memories {where_clause}
                """, params)

                # Delete from vector index
                placeholders = ','.join(['?' for _ in deleted_ids])
                conn.execute(f"""
                    DELETE FROM vec_session_search WHERE memory_id IN ({placeholders})
                """, deleted_ids)

                conn.commit()

                return {
                    "success": True,
                    "deleted_count": len(deleted_ids),
                    "deleted_ids": deleted_ids,
                    "cutoff_date": cutoff_date,
                    "memory_type_filter": memory_type,
                    "message": f"Cleaned up {len(deleted_ids)} old memories"
                }

            except Exception as e:
                conn.rollback()
                raise

            finally:
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "error": "Cleanup failed",
                "message": str(e)
            }