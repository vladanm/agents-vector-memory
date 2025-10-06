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
from typing import Dict, Any, List, Optional, Literal

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

# Granularity mapping for consolidated search functions
GRANULARITY_MAP = {
    "specific_chunks": "fine",
    "section_context": "medium",
    "full_documents": "coarse"
}

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
    """Store agent session snapshots for continuity across iterations.

    Args:
        agent_id: Agent identifier, e.g. 'main' or 'data-processor'
        session_id: Session identifier
        content: Session snapshot content
        session_iter: Iteration number (1-based)
        title: Snapshot title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data

    Returns:
        Success status and memory details
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
    """Store original prompts to prevent loss during session.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Original prompt text
        session_iter: Iteration number (1-based)
        task_code: Task identifier for sub-agents
        title: Prompt title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data

    Returns:
        Success status and memory details
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
    """Store system info: script paths, endpoints, DB connections.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: System configuration or command
        session_iter: Iteration number (1-based)
        task_code: Task identifier for sub-agents
        title: System info title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data

    Returns:
        Success status and memory details
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
    """Store agent reports with automatic chunking for large documents.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Report content (markdown recommended)
        session_iter: Iteration number (1-based)
        task_code: Task identifier
        title: Report title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data
        auto_chunk: Enable document chunking (default: True)

    Returns:
        Success status and memory details
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
    """Store long-term reference material: documentation, guides, tutorials.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Knowledge content (markdown recommended)
        session_iter: Iteration number (1-based)
        task_code: Task identifier
        title: Document title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data
        auto_chunk: Enable document chunking (default: True)

    Returns:
        Success status and memory details
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
    """Store additional notes and observations about existing reports.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Observation text
        parent_report_id: Related report ID
        session_iter: Iteration number (1-based)
        task_code: Task identifier
        title: Observation title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data

    Returns:
        Success status and memory details
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
    metadata: dict = None,
    auto_chunk: bool = True
) -> dict[str, Any]:
    """Store important insights during task execution (gotcha moments).

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        content: Working memory insight
        session_iter: Iteration number (1-based)
        task_code: Task identifier
        title: Memory title
        description: Brief summary
        tags: Categorization tags
        metadata: Additional key-value data
        auto_chunk: Enable document chunking (default: True)

    Returns:
        Success status and memory details
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
        metadata=metadata or {},
        auto_chunk=auto_chunk
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
    limit: int = 3,
    latest_first: bool = True
) -> dict[str, Any]:
    """Search session snapshots with filtering. Returns ordered by session_iter DESC, created_at DESC.

    Args:
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        query: Semantic search text
        limit: Max results (1-100)
        latest_first: Sort newest first (default: True)

    Returns:
        Search results with metadata
    """
    # Note: search_memories() performs scoped/filtered searches, not semantic search
    # For semantic search with similarity thresholds, use search_with_granularity()
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
    limit: int = 3,
    latest_first: bool = True
) -> dict[str, Any]:
    """Search system configurations with filtering. Returns ordered by session_iter DESC, created_at DESC.

    Args:
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        task_code: Filter by task
        query: Semantic search text
        limit: Max results (1-100)
        latest_first: Sort newest first (default: True)

    Returns:
        Search results with metadata
    """
    # Note: search_memories() performs scoped/filtered searches, not semantic search
    # For semantic search with similarity thresholds, use search_with_granularity()
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
def search_input_prompts(
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    query: str = None,
    limit: int = 3,
    latest_first: bool = True
) -> dict[str, Any]:
    """Search stored prompts with filtering. Returns ordered by session_iter DESC, created_at DESC.

    Args:
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        task_code: Filter by task
        query: Semantic search text
        limit: Max results (1-100)
        latest_first: Sort newest first (default: True)

    Returns:
        Search results with metadata
    """
    # Note: search_memories() performs scoped/filtered searches, not semantic search
    # For semantic search with similarity thresholds, use search_with_granularity()
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
    """Load session context only if agent worked on same task before. Used for task continuity.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier
        current_task_code: Current task code

    Returns:
        Session context if match found, empty otherwise
    """
    return store.load_session_context_for_task(agent_id, session_id, current_task_code)

# ======================
# UTILITY FUNCTIONS
# ======================

@mcp.tool()
def get_memory_by_id(memory_id: int) -> dict[str, Any]:
    """Retrieve memory by ID with complete details.

    Args:
        memory_id: Memory identifier

    Returns:
        Memory details or error
    """
    return store.get_memory(memory_id)

@mcp.tool()
def get_session_stats(
    agent_id: str = None,
    session_id: str = None
) -> dict[str, Any]:
    """Get memory usage statistics for sessions.

    Args:
        agent_id: Filter by agent
        session_id: Filter by session

    Returns:
        Usage statistics
    """
    return store.get_session_stats(agent_id, session_id)

@mcp.tool()
def list_sessions(
    agent_id: str = None,
    limit: int = 20
) -> dict[str, Any]:
    """List recent sessions with basic information.

    Args:
        agent_id: Filter by agent
        limit: Max sessions (1-100)

    Returns:
        Session list with metadata
    """
    return store.list_sessions(agent_id, limit)

@mcp.tool()
def delete_memory(memory_id: int) -> dict[str, Any]:
    """Delete memory and all associated data (embeddings, chunks).

    Args:
        memory_id: Memory identifier

    Returns:
        Deletion status
    """
    return store.delete_memory(memory_id)

@mcp.tool()
def cleanup_old_memories(
    older_than_days: int = 30,
    memory_type: str = None
) -> dict[str, Any]:
    """Delete memories older than specified days.

    Args:
        older_than_days: Age threshold in days (default: 30)
        memory_type: Filter by type, e.g. 'reports'

    Returns:
        Cleanup statistics
    """
    return store.cleanup_old_memories(older_than_days, memory_type)

@mcp.tool()
def reconstruct_document(memory_id: int) -> dict[str, Any]:
    """Reconstruct complete document from stored chunks.

    Args:
        memory_id: Parent memory identifier

    Returns:
        Reconstructed content and chunk info
    """
    return store.reconstruct_document(memory_id)

@mcp.tool()
def write_document_to_file(
    memory_id: int,
    output_path: str = None,
    include_metadata: bool = True,
    format: str = "markdown"
) -> dict[str, Any]:
    """Write reconstructed document to disk. Use for large documents (>20k tokens).

    Args:
        memory_id: Memory identifier
        output_path: Absolute file path (auto-generated if omitted)
        include_metadata: Add YAML frontmatter (default: True)
        format: Output format: 'markdown' or 'plain' (default: 'markdown')

    Returns:
        Status, file_path, file_size, estimated_tokens
    """
    return store.write_document_to_file(memory_id, output_path, include_metadata, format)

# ================================
# CONSOLIDATED THREE-TIER GRANULARITY SEARCH
# ================================

@mcp.tool()
def search_knowledge_base(
    query: str,
    granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    limit: int = 3,
    similarity_threshold: float = 0.7,
    auto_merge_threshold: float = 0.6
) -> dict[str, Any]:
    """Search knowledge base with configurable granularity.

    Args:
        query: Semantic search text
        granularity: Search granularity level
            - 'specific_chunks': Fine-grained (<400 tokens) for pinpoint queries, specific details, code snippets, definitions
            - 'section_context': Medium-grained (400-1200 tokens) for section-level understanding, concepts, procedures. Auto-merges when ≥60% siblings match
            - 'full_documents': Coarse-grained (full docs) for discovery, overviews, broad context. Docs: <5k (small), 5-50k (medium), 50k+ (large) tokens
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        task_code: Filter by task
        limit: Max results (1-100)
        similarity_threshold: Min similarity 0.0-1.0 (default: 0.7)
        auto_merge_threshold: Merge if ≥X siblings match (default: 0.6, applies to section_context)

    Returns:
        Search results at specified granularity. Source and granularity indicated in response.
        - specific_chunks: Chunk content, indices, header path, score. Source: 'chunk', granularity: 'fine'
        - section_context: Section content, header path, match stats, merge flag. Source: 'expanded_section', granularity: 'medium'
        - full_documents: Full content, title, description, tags. Source: 'document', granularity: 'coarse'
    """
    return store.search_with_granularity(
        query=query,
        memory_type="knowledge_base",
        granularity=GRANULARITY_MAP[granularity],
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        limit=limit,
        similarity_threshold=similarity_threshold,
        auto_merge_threshold=auto_merge_threshold
    )

@mcp.tool()
def search_reports(
    query: str,
    granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    limit: int = 3,
    similarity_threshold: float = 0.7,
    auto_merge_threshold: float = 0.6
) -> dict[str, Any]:
    """Search agent reports with configurable granularity.

    Args:
        query: Semantic search text
        granularity: Search granularity level
            - 'specific_chunks': Fine-grained (<400 tokens) for specific findings, data points, precise details
            - 'section_context': Medium-grained (400-1200 tokens) for section-level analysis, grouped findings, context. Auto-merges when ≥60% siblings match
            - 'full_documents': Coarse-grained (full docs) for report discovery, summaries, full context. Docs: <5k (small), 5-50k (medium), 50k+ (large) tokens
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        task_code: Filter by task
        limit: Max results (1-100)
        similarity_threshold: Min similarity 0.0-1.0 (default: 0.7)
        auto_merge_threshold: Merge if ≥X siblings match (default: 0.6, applies to section_context)

    Returns:
        Search results at specified granularity. Source and granularity indicated in response.
        - specific_chunks: Chunk content, indices, header path, score. Source: 'chunk', granularity: 'fine'
        - section_context: Section content, header path, match stats, merge flag. Source: 'expanded_section', granularity: 'medium'
        - full_documents: Full content, title, description, tags. Source: 'document', granularity: 'coarse'
    """
    return store.search_with_granularity(
        query=query,
        memory_type="reports",
        granularity=GRANULARITY_MAP[granularity],
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        limit=limit,
        similarity_threshold=similarity_threshold,
        auto_merge_threshold=auto_merge_threshold
    )

@mcp.tool()
def search_working_memory(
    query: str,
    granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    limit: int = 3,
    similarity_threshold: float = 0.7,
    auto_merge_threshold: float = 0.6
) -> dict[str, Any]:
    """Search working memory (insights, gotcha moments) with configurable granularity.

    Args:
        query: Semantic search text
        granularity: Search granularity level
            - 'specific_chunks': Fine-grained (<400 tokens) for specific insights, gotcha moments, precise details
            - 'section_context': Medium-grained (400-1200 tokens) for section-level understanding, grouped insights, context. Auto-merges when ≥60% siblings match
            - 'full_documents': Coarse-grained (full docs) for complete memory discovery, full context of insights. Docs: <5k (small), 5-50k (medium), 50k+ (large) tokens
        agent_id: Filter by agent
        session_id: Filter by session
        session_iter: Filter by iteration
        task_code: Filter by task
        limit: Max results (1-100)
        similarity_threshold: Min similarity 0.0-1.0 (default: 0.7)
        auto_merge_threshold: Merge if ≥X siblings match (default: 0.6, applies to section_context)

    Returns:
        Search results at specified granularity. Source and granularity indicated in response.
        - specific_chunks: Chunk content, indices, header path, score. Source: 'chunk', granularity: 'fine'
        - section_context: Section content, header path, match stats, merge flag. Source: 'expanded_section', granularity: 'medium'
        - full_documents: Full content, title, description, tags. Source: 'document', granularity: 'coarse'
    """
    return store.search_with_granularity(
        query=query,
        memory_type="working_memory",
        granularity=GRANULARITY_MAP[granularity],
        agent_id=agent_id,
        session_id=session_id,
        session_iter=session_iter,
        task_code=task_code,
        limit=limit,
        similarity_threshold=similarity_threshold,
        auto_merge_threshold=auto_merge_threshold
    )

@mcp.tool()
def expand_chunk_context(
    memory_id: int,
    chunk_index: int,
    context_window: int = 2
) -> dict[str, Any]:
    """Retrieve chunk with surrounding siblings. Universal tool for any granularity level.

    Args:
        memory_id: Parent memory identifier
        chunk_index: Target chunk index
        context_window: Chunks before/after (default: 2)

    Returns:
        Target chunk, previous/next chunks, expanded content
    """
    return store.expand_chunk_context(memory_id, chunk_index, context_window)

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

        # Log to stderr (stdout is reserved for MCP protocol)
        print("🔧 Available session-centric functions:", file=sys.stderr)
        print("   📝 Storage: store_session_context, store_input_prompt, store_system_memory", file=sys.stderr)
        print("   📊 Reports: store_report, store_report_observation, store_working_memory", file=sys.stderr)
        print("   🔍 Search: Consolidated 3-function search with granularity parameter", file=sys.stderr)
        print("   🔄 Continuity: load_session_context_for_task", file=sys.stderr)
        print("   📈 Stats: get_session_stats, list_sessions", file=sys.stderr)
        print("   💾 Document Export: write_document_to_file (for large documents)", file=sys.stderr)
        print("⚡ Proper ordering: session_iter DESC, created_at DESC", file=sys.stderr)

        # Run the MCP server
        mcp.run()

    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise
