#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
# /// script
# dependencies = [
#     "mcp>=0.3.0",
#     "sqlite-vec>=0.1.6",
#     "sentence-transformers>=2.2.2",
#     "tiktoken>=0.5.0",
#     "pyyaml>=6.0",
#     "langchain-text-splitters>=0.3.0"
# ]
# requires-python = ">=3.8"
# ///

"""
Agent Session Memory MCP Server
===============================

Specialized vector-based memory server designed for agent session management.
Built specifically for hierarchical agent memory with session tracking.

Core Requirements:
- main agent: stores session_context and system_memory scoped by session_id/session_iter
- sub-agents: stores reports, observations, context memory, working memory, system memory
  scoped by agent_id + session_id + optional(session_iter, task_code)
- input_prompt: stores original prompts to prevent loss
- Proper ordering: session_iter DESC, created_at DESC

Supported Memory Types:
- session_context: Agent session snapshots for continuity
- input_prompt: Original user prompts for reference  
- reports: Agent-generated analysis and findings
- working_memory: Important information during task execution
- system_memory: System configs, commands, scripts for tasks
- report_observations: Additional notes on existing reports

Usage:
    python main.py --database-path /path/to/any/database.db
    python main.py --working-dir /path/to/project  # Legacy mode

Direct database path (preferred): Any SQLite file with agent_session_memory schema
Legacy mode: Memory files stored in {working_dir}/memory/agent_session_memory.db
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.fastmcp import FastMCP

# Import our modules
from src.config import Config
from src.security import validate_working_dir, SecurityError
from src.session_memory_store import SessionMemoryStore

# Initialize global objects
config = Config()
mcp = FastMCP("Agent Session Memory")
store = None

def initialize_store(working_dir: str = None, database_path: str = None) -> None:
    """Initialize the session memory store."""
    global store
    
    if database_path:
        # Use direct database path (new approach)
        db_path = Path(database_path)
        if not db_path.is_absolute():
            print(f"Database path must be absolute: {database_path}", file=sys.stderr)
            sys.exit(1)
        
        # Create parent directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SessionMemoryStore(db_path=db_path)
        print(f"Agent session memory store initialized with direct path: {store.db_path}")
        
    elif working_dir:
        # Use working directory approach (legacy)
        try:
            validated_dir = validate_working_dir(working_dir)
            config.working_dir = validated_dir
        except SecurityError as e:
            print(f"Security error: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Initialize store with traditional path
        db_path = Path(config.working_dir) / "memory" / "agent_session_memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SessionMemoryStore(db_path=db_path)
        print(f"Agent session memory store initialized with working dir: {store.db_path}")
        
    else:
        # Default behavior - use current working directory
        db_path = Path(config.working_dir) / "memory" / "agent_session_memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SessionMemoryStore(db_path=db_path)
        print(f"Agent session memory store initialized (default): {store.db_path}")

# ======================
# MAIN AGENT FUNCTIONS
# ======================

@mcp.tool()
def store_session_context(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None
) -> dict[str, Any]:
    """
    Store session context for main or sub-agents.
    Used to save session snapshots for continuity across iterations.
    
    Args:
        agent_id: Agent identifier ("main" or "specialized-agent")
        session_id: Session identifier 
        content: Session context content
        session_iter: Session iteration number (default: 1)
        title: Context title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
    
    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="session_context",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {}
    )

@mcp.tool()
def store_input_prompt(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None
) -> dict[str, Any]:
    """
    Store original input prompt to prevent loss during session.
    Can be used by both main and sub-agents.
    
    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Original input prompt content
        session_iter: Session iteration number
        task_code: Task identifier (optional)
        title: Prompt title
        description: Brief description
        tags: List of relevant tags  
        metadata: Additional metadata
        
    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="input_prompt",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {}
    )

@mcp.tool()
def store_system_memory(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None
) -> dict[str, Any]:
    """
    Store system information like script paths, endpoints, DB connections.
    Used by main agent for session-scoped system info, sub-agents for task-scoped.
    
    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: System information content
        session_iter: Session iteration number
        task_code: Task identifier (optional for sub-agents)
        title: System info title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
        
    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="system_memory",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {}
    )

# ======================
# SUB-AGENT FUNCTIONS
# ======================

@mcp.tool()
def store_report(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None,
    auto_chunk: bool = True
) -> dict[str, Any]:
    """
    Store agent reports (usually MD files produced at end of task).

    Args:
        agent_id: Agent identifier (required)
        session_id: Session identifier (required)
        content: Report content
        session_iter: Session iteration number (optional)
        task_code: Task identifier (optional)
        title: Report title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
        auto_chunk: Enable automatic document chunking (default: True for reports)

    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="reports",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {},
        auto_chunk=auto_chunk
    )

@mcp.tool()
def store_knowledge_base(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None,
    auto_chunk: bool = True
) -> dict[str, Any]:
    """
    Store knowledge base documents (long-term reference material, documentation, guides).

    Args:
        agent_id: Agent identifier (required)
        session_id: Session identifier (required)
        content: Knowledge base content (markdown, text, etc.)
        session_iter: Session iteration number (optional)
        task_code: Task identifier (optional)
        title: Document title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
        auto_chunk: Enable automatic document chunking (default: True for knowledge_base)

    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="knowledge_base",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {},
        auto_chunk=auto_chunk
    )

@mcp.tool()
def store_report_observation(
    agent_id: str,
    session_id: str,
    content: str,
    parent_report_id: int = None,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None
) -> dict[str, Any]:
    """
    Store additional info for report observations.
    
    Args:
        agent_id: Agent identifier (required)
        session_id: Session identifier (required)
        content: Observation content
        parent_report_id: ID of related report (optional)
        session_iter: Session iteration number (optional)
        task_code: Task identifier (optional)
        title: Observation title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
        
    Returns:
        Dict with success status and memory details
    """
    if metadata is None:
        metadata = {}
    if parent_report_id:
        metadata['parent_report_id'] = parent_report_id
        
    return store.store_memory(
        memory_type="report_observations",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata
    )

@mcp.tool()
def store_working_memory(
    agent_id: str,
    session_id: str,
    content: str,
    session_iter: int = 1,
    task_code: str = None,
    title: str = None,
    description: str = None,
    tags: list[str] = None,
    metadata: dict = None
) -> dict[str, Any]:
    """
    Store important information during task execution (gotcha moments).
    
    Args:
        agent_id: Agent identifier (required)
        session_id: Session identifier (required)  
        content: Working memory content
        session_iter: Session iteration number (optional)
        task_code: Task identifier (optional)
        title: Memory title
        description: Brief description
        tags: List of relevant tags
        metadata: Additional metadata
        
    Returns:
        Dict with success status and memory details
    """
    return store.store_memory(
        memory_type="working_memory",
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        session_iter=session_iter,
        task_code=task_code,
        title=title,
        description=description,
        tags=tags or [],
        metadata=metadata or {}
    )

# ======================
# SEARCH FUNCTIONS WITH SCOPING
# ======================

@mcp.tool()
def search_session_context(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True
) -> dict[str, Any]:
    """
    Search session context with proper scoping and ordering.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)  
        session_iter: Filter by specific iteration (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        
    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="session_context",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        query=query,
        limit=limit,
        latest_first=latest_first
    )

@mcp.tool()
def search_system_memory(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True
) -> dict[str, Any]:
    """
    Search system memory with proper scoping and ordering.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        session_iter: Filter by specific iteration (optional)
        task_code: Filter by task code (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        
    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="system_memory",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        query=query,
        limit=limit,
        latest_first=latest_first
    )

@mcp.tool()
def search_reports(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True
) -> dict[str, Any]:
    """
    Search reports with proper scoping and ordering.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        session_iter: Filter by specific iteration (optional)
        task_code: Filter by task code (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        
    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="reports",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        query=query,
        limit=limit,
        latest_first=latest_first
    )

@mcp.tool()
def search_knowledge_base(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True,
    similarity_threshold: float = 0.80
) -> dict[str, Any]:
    """
    Search knowledge base documents with proper scoping and ordering.

    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        session_iter: Filter by specific iteration (optional)
        task_code: Filter by task code (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        similarity_threshold: Minimum similarity score for results (default: 0.80)
                            Higher = stricter filtering. Range: 0.0 to 1.0

    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="knowledge_base",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        query=query,
        limit=limit,
        latest_first=latest_first,
        similarity_threshold=similarity_threshold
    )

@mcp.tool()
def search_working_memory(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True
) -> dict[str, Any]:
    """
    Search working memory with proper scoping and ordering.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        session_iter: Filter by specific iteration (optional)
        task_code: Filter by task code (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        
    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="working_memory",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        query=query,
        limit=limit,
        latest_first=latest_first
    )

@mcp.tool()
def search_input_prompts(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 10,
    latest_first: bool = True
) -> dict[str, Any]:
    """
    Search input prompts with proper scoping and ordering.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        session_iter: Filter by specific iteration (optional)
        task_code: Filter by task code (optional)
        query: Semantic search query (optional)
        limit: Maximum results (default: 10)
        latest_first: Order by latest iteration/creation first (default: True)
        
    Returns:
        Dict with search results ordered by session_iter DESC, created_at DESC
    """
    return store.search_memories(
        memory_type="input_prompt",
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        query=query,
        limit=limit,
        latest_first=latest_first
    )

# ======================
# CONDITIONAL LOADING FOR TASK CONTINUITY
# ======================

@mcp.tool()
def load_session_context_for_task(
    agent_id: str,
    session_id: str,
    current_task_code: str
) -> dict[str, Any]:
    """
    Load session context only if agent previously worked on the same task_code.
    Used by sub-agents for task continuity.
    
    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        current_task_code: Current task being worked on
        
    Returns:
        Dict with session context if task match found, empty if no match
    """
    return store.load_session_context_for_task(agent_id, session_id, current_task_code)

# ======================
# UTILITY FUNCTIONS
# ======================

@mcp.tool()
def get_memory_by_id(memory_id: int) -> dict[str, Any]:
    """
    Retrieve specific memory by ID.
    
    Args:
        memory_id: The memory ID to retrieve
        
    Returns:
        Dict with memory details or error
    """
    return store.get_memory(memory_id)

@mcp.tool()
def get_session_stats(
    agent_id: str = None,
    session_id: str = None
) -> dict[str, Any]:
    """
    Get statistics about session memory usage.
    
    Args:
        agent_id: Filter by agent ID (optional)
        session_id: Filter by session ID (optional)
        
    Returns:
        Dict with session statistics
    """
    return store.get_session_stats(agent_id, session_id)

@mcp.tool()
def list_sessions(
    agent_id: str = None,
    limit: int = 20
) -> dict[str, Any]:
    """
    List recent sessions with basic info.
    
    Args:
        agent_id: Filter by agent ID (optional)
        limit: Maximum sessions to return (default: 20)
        
    Returns:
        Dict with session list
    """
    return store.list_sessions(agent_id, limit)

@mcp.tool()
def delete_memory(memory_id: int) -> dict[str, Any]:
    """
    Delete a memory and all associated data (embeddings, chunks).

    Args:
        memory_id: The ID of the memory to delete

    Returns:
        Dict with deletion status
    """
    return store.delete_memory(memory_id)

@mcp.tool()
def cleanup_old_memories(
    older_than_days: int = 30,
    memory_type: str = None
) -> dict[str, Any]:
    """
    Clean up old memories older than specified days.

    Args:
        older_than_days: Delete memories older than this many days (default: 30)
        memory_type: Optional memory type filter

    Returns:
        Dict with cleanup statistics
    """
    return store.cleanup_old_memories(older_than_days, memory_type)

@mcp.tool()
def reconstruct_document(memory_id: int) -> dict[str, Any]:
    """
    Reconstruct a document from its stored chunks.

    Args:
        memory_id: The ID of the parent memory to reconstruct

    Returns:
        Dict with reconstructed content and chunk info
    """
    return store.reconstruct_document(memory_id)

# ======================
# SERVER INITIALIZATION
# ======================

if __name__ == "__main__":
    import argparse
    
    print("🤖 AGENT SESSION MEMORY MCP SERVER STARTING")
    print(f"📁 File: {__file__}")
    print("🎯 Specialized for agent session management with proper scoping")
    
    parser = argparse.ArgumentParser(description="Agent Session Memory MCP Server")
    parser.add_argument("--working-dir", help="Working directory for memory files (legacy)")
    parser.add_argument("--database-path", help="Direct path to SQLite database file (preferred)")
    args = parser.parse_args()
    
    try:
        # Initialize store with database path or working directory
        initialize_store(working_dir=args.working_dir, database_path=args.database_path)
        
        print("🔧 Available session-centric functions:")
        print("   📝 Storage: store_session_context, store_input_prompt, store_system_memory")
        print("   📊 Reports: store_report, store_report_observation, store_working_memory") 
        print("   🔍 Search: All with proper agent_id + session_id + session_iter scoping")
        print("   🔄 Continuity: load_session_context_for_task")
        print("   📈 Stats: get_session_stats, list_sessions")
        print("⚡ Proper ordering: session_iter DESC, created_at DESC")
        
        # Run the MCP server
        mcp.run()
        
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise