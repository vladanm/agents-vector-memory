"""
MCP Structured Output Type Definitions
=======================================

TypedDict definitions for all MCP tool return values to enable structured outputs.
These types ensure consistent, machine-readable responses across all 25 tools.
"""

from typing import TypedDict, List, Dict, Any, Optional


# ======================
# STORAGE OPERATION RESULTS
# ======================

class StoreMemoryResult(TypedDict):
    """Result from store_memory, store_report, store_knowledge_base, etc."""
    success: bool
    memory_id: Optional[int]
    memory_type: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]
    content_hash: Optional[str]
    chunks_created: Optional[int]
    created_at: Optional[str]
    message: str
    error: Optional[str]  # Present when success=False


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
    task_code: Optional[str]
    content: str
    title: Optional[str]
    description: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
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
    results: List[MemorySearchResult]
    total_results: int
    query: Optional[str]
    filters: Dict[str, Any]
    limit: int
    latest_first: bool
    error: Optional[str]
    message: Optional[str]


class GranularSearchResult(TypedDict):
    """Result from search_with_granularity (knowledge_base, reports, working_memory)"""
    success: bool
    results: List[Any]  # Content varies by granularity
    total_results: int
    granularity: str
    message: Optional[str]
    error: Optional[str]


# ======================
# UTILITY OPERATION RESULTS
# ======================

class GetMemoryResult(TypedDict):
    """Result from get_memory_by_id"""
    success: bool
    memory: Optional[Dict[str, Any]]
    error: Optional[str]
    message: Optional[str]


class ExpandChunkContextResult(TypedDict):
    """Result from expand_chunk_context"""
    success: bool
    memory_id: Optional[int]
    target_chunk_index: Optional[int]
    context_window: Optional[int]
    chunks_returned: Optional[int]
    expanded_content: Optional[str]
    chunks: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    message: Optional[str]


class LoadSessionContextResult(TypedDict):
    """Result from load_session_context_for_task"""
    success: bool
    found_previous_context: bool
    context: Optional[Dict[str, Any]]
    message: str
    error: Optional[str]


class SessionStatsResult(TypedDict):
    """Result from get_session_stats"""
    success: bool
    total_memories: Optional[int]
    memory_types: Optional[int]
    unique_agents: Optional[int]
    unique_sessions: Optional[int]
    unique_tasks: Optional[int]
    max_session_iter: Optional[int]
    avg_content_length: Optional[float]
    total_access_count: Optional[int]
    memory_type_breakdown: Optional[Dict[str, int]]
    filters: Optional[Dict[str, Optional[str]]]
    error: Optional[str]
    message: Optional[str]


class SessionInfo(TypedDict):
    """Individual session information"""
    agent_id: str
    session_id: str
    memory_count: int
    latest_iter: int
    latest_activity: str
    first_activity: str
    memory_types: List[str]


class ListSessionsResult(TypedDict):
    """Result from list_sessions"""
    success: bool
    sessions: List[SessionInfo]
    total_sessions: int
    agent_filter: Optional[str]
    limit: int
    error: Optional[str]
    message: Optional[str]


class ReconstructDocumentResult(TypedDict):
    """Result from reconstruct_document"""
    success: bool
    memory_id: Optional[int]
    content: Optional[str]
    title: Optional[str]
    memory_type: Optional[str]
    chunk_count: int
    message: str
    error: Optional[str]


class WriteDocumentResult(TypedDict):
    """Result from write_document_to_file"""
    success: bool
    file_path: Optional[str]
    file_size_bytes: Optional[int]
    file_size_human: Optional[str]
    estimated_tokens: Optional[int]
    memory_id: Optional[int]
    created_at: Optional[str]
    message: str
    error_code: Optional[str]
    error_message: Optional[str]


class DeleteMemoryResult(TypedDict):
    """Result from delete_memory"""
    success: bool
    memory_id: Optional[int]
    message: str
    error: Optional[str]


class CleanupMemoriesResult(TypedDict):
    """Result from cleanup_old_memories"""
    success: bool
    deleted_count: Optional[int]
    oldest_deleted: Optional[str]
    newest_deleted: Optional[str]
    message: str
    error: Optional[str]


# ======================
# ERROR RESULT (fallback for any failed operation)
# ======================

class ErrorResult(TypedDict):
    """Generic error result for any operation"""
    success: bool  # Always False
    error: str
    message: str
    error_code: Optional[str]
