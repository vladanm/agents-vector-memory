"""
MCP Structured Output Type Definitions
=======================================

TypedDict definitions for all MCP tool return values to enable structured outputs.
These types ensure consistent, machine-readable responses across all 25 tools.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


# ======================
# STORAGE OPERATION RESULTS
# ======================

class StoreMemoryResult(TypedDict):
    """Result from store_memory, store_report, store_knowledge_base, etc."""
    success: bool
    memory_id: int | None
    memory_type: str | None
    agent_id: str | None
    session_id: str | None
    content_hash: str | None
    chunks_created: int | None
    created_at: str | None
    message: str
    error: str | None  # Present when success=False


# ======================
# SEARCH RESULTS
# ======================

class MemorySearchResult(TypedDict):
    """Individual memory result from search operations"""
    id: int
    memory_type: str
    agent_id: str
    session_id: str
    session_iter: int
    task_code: str | None
    content: str
    title: str | None
    description: str | None
    tags: list[str]
    metadata: dict[str, Any]
    content_hash: str
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    similarity: float
    source_type: str


class SearchMemoriesResult(TypedDict):
    """Result from search_memories, search_session_context, etc."""
    success: bool
    results: list[MemorySearchResult]
    total_results: int
    query: str | None
    filters: dict[str, Any]
    limit: int
    latest_first: bool
    error: str | None
    message: str | None


class GranularSearchResult(TypedDict):
    """Result from search_with_granularity (knowledge_base, reports, working_memory)"""
    success: bool
    results: list[Any]  # Content varies by granularity
    total_results: int
    granularity: str
    message: str | None
    error: str | None


# ======================
# UTILITY OPERATION RESULTS
# ======================

class GetMemoryResult(TypedDict):
    """Result from get_memory_by_id"""
    success: bool
    memory: Optional[dict[str, Any]]
    error: str | None
    message: str | None


class ExpandChunkContextResult(TypedDict):
    """Result from expand_chunk_context"""
    success: bool
    memory_id: int | None
    target_chunk_index: int | None
    context_window: int | None
    chunks_returned: int | None
    expanded_content: str | None
    chunks: Optional[list[dict[str, Any]]]
    error: str | None
    message: str | None


class LoadSessionContextResult(TypedDict):
    """Result from load_session_context_for_task"""
    success: bool
    found_previous_context: bool
    context: Optional[dict[str, Any]]
    message: str
    error: str | None


class SessionStatsResult(TypedDict):
    """Result from get_session_stats"""
    success: bool
    total_memories: int | None
    memory_types: int | None
    unique_agents: int | None
    unique_sessions: int | None
    unique_tasks: int | None
    max_session_iter: int | None
    avg_content_length: float | None
    total_access_count: int | None
    memory_type_breakdown: Optional[dict[str, int]]
    filters: Optional[dict[str, str | None]]
    error: str | None
    message: str | None


class SessionInfo(TypedDict):
    """Individual session information"""
    agent_id: str
    session_id: str
    memory_count: int
    latest_iter: int
    latest_activity: str
    first_activity: str
    memory_types: list[str]


class ListSessionsResult(TypedDict):
    """Result from list_sessions"""
    success: bool
    sessions: list[SessionInfo]
    total_sessions: int
    agent_filter: str | None
    limit: int
    error: str | None
    message: str | None


class ReconstructDocumentResult(TypedDict):
    """Result from reconstruct_document"""
    success: bool
    memory_id: int | None
    content: str | None
    title: str | None
    memory_type: str | None
    chunk_count: int
    message: str
    error: str | None


class WriteDocumentResult(TypedDict):
    """Result from write_document_to_file"""
    success: bool
    file_path: str | None
    file_size_bytes: int | None
    file_size_human: str | None
    estimated_tokens: int | None
    memory_id: int | None
    created_at: str | None
    message: str
    error_code: str | None
    error_message: str | None


class DeleteMemoryResult(TypedDict):
    """Result from delete_memory"""
    success: bool
    memory_id: int | None
    message: str
    error: str | None


class CleanupMemoriesResult(TypedDict):
    """Result from cleanup_old_memories"""
    success: bool
    deleted_count: int | None
    oldest_deleted: str | None
    newest_deleted: str | None
    message: str
    error: str | None


# ======================
# ERROR RESULT (fallback for any failed operation)
# ======================

class ErrorResult(TypedDict):
    """Generic error result for any operation"""
    success: bool  # Always False
    error: str
    message: str
    error_code: str | None
