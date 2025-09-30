"""
Agent Session Memory Store
==========================

Core storage engine for agent session management with proper scoping,
ordering, and task continuity support.
"""

import sqlite3
import sqlite_vec
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store memory with session scoping.
        
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
                
                # Insert memory
                cursor = conn.execute("""
                    INSERT INTO session_memories (
                        memory_type, agent_id, session_id, session_iter, task_code,
                        content, title, description, tags, metadata,
                        content_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_type, agent_id, session_id, session_iter, task_code,
                    content, title, description, json.dumps(tags), 
                    json.dumps(metadata or {}), content_hash, now, now
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
                
                conn.commit()
                
                return {
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
                
                # Execute vector search with proper LIMIT constraint
                # sqlite-vec requires LIMIT on the vector search subquery
                
                # Build WHERE clause for the main query (not the subquery)
                where_clause_main = where_clause.replace("WHERE ", "AND ") if where_clause else ""
                
                # Add vector search ordering
                order_clause = "ORDER BY v.distance ASC"
                if latest_first:
                    order_clause = "ORDER BY m.session_iter DESC, m.created_at DESC, v.distance ASC"
                
                vector_query = f"""
                    SELECT m.*, v.distance
                    FROM session_memories m
                    JOIN (
                        SELECT memory_id, distance 
                        FROM vec_session_search 
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance ASC
                    ) v ON m.id = v.memory_id
                    WHERE v.distance < ?
                    {where_clause_main}
                    {order_clause}
                """
                
                # Parameters: embedding, k (limit) for subquery, then similarity_threshold and additional params for main query
                # In sqlite-vec: distance 0.0 = identical, distance 2.0+ = very different
                # Based on testing, reasonable matches have distances around 1.1-1.5
                # Use a permissive threshold that allows semantic matches while filtering noise
                distance_threshold = 1.8  # Allow semantic matches, filter out very dissimilar content
                final_params = [query_embedding.tobytes(), limit, distance_threshold] + params
                rows = conn.execute(vector_query, final_params).fetchall()
                
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
                    "similarity": 1.0 - row[16] if len(row) > 16 else 1.0  # Convert distance to similarity
                }
                results.append(memory)
            
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