"""
Transformation helper to fix Bugs #3 and #4 - Pydantic validation errors.

This module provides a function to transform database search results into
the format expected by MCP Pydantic models (SearchMemoriesResult).

Root Cause:
-----------
The database returns results with certain fields as strings (e.g., memory_id as TEXT,
session_iter as TEXT like "v1"), but the Pydantic TypedDict models expect different
types (e.g., id as int, session_iter as int).

Solution:
---------
Transform the search results to match the expected schema before returning them
from MCP tool functions.
"""

from typing import Any, Dict, List
import hashlib


def _format_search_result_for_mcp(db_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single database search result to match SearchMemoriesResult schema.

    Args:
        db_result: Raw result dict from database with fields like:
            - id or memory_id: Could be TEXT (e.g., "input_prompt:main:abc123")
            - session_iter: TEXT (e.g., "v1", "v2")
            - Other fields matching MemorySearchResult

    Returns:
        Dict formatted to match MemorySearchResult TypedDict expectations

    Type Conversions:
        - id: Convert string memory_id to integer hash
        - session_iter: Convert string like "v1" to integer (1)
        - Ensure all required fields are present with correct types
    """
    # Handle memory_id -> id conversion
    memory_id_str = db_result.get("id") or db_result.get("memory_id", "unknown")

    # Convert string memory_id to stable integer ID
    # Use first 8 bytes of SHA256 hash as integer
    id_hash = hashlib.sha256(str(memory_id_str).encode()).digest()[:8]
    id_int = int.from_bytes(id_hash, byteorder='big', signed=False)

    # Handle session_iter conversion (e.g., "v1" -> 1, "v2" -> 2)
    session_iter_str = db_result.get("session_iter", "v1")
    if isinstance(session_iter_str, str):
        # Extract number from "v1", "v2", etc.
        try:
            if session_iter_str.startswith("v"):
                session_iter_int = int(session_iter_str[1:])
            else:
                session_iter_int = int(session_iter_str)
        except (ValueError, IndexError):
            session_iter_int = 1  # Default to 1 if parsing fails
    else:
        session_iter_int = int(session_iter_str or 1)

    # Build the transformed result
    return {
        "id": id_int,
        "memory_type": db_result.get("memory_type", ""),
        "agent_id": db_result.get("agent_id", ""),
        "session_id": db_result.get("session_id", ""),
        "session_iter": session_iter_int,
        "task_code": db_result.get("task_code"),
        "content": db_result.get("content", ""),
        "title": db_result.get("title"),
        "description": db_result.get("description"),
        "tags": db_result.get("tags", []),
        "metadata": db_result.get("metadata", {}),
        "content_hash": db_result.get("content_hash", ""),
        "created_at": db_result.get("created_at", ""),
        "updated_at": db_result.get("updated_at", ""),
        "accessed_at": db_result.get("accessed_at", ""),
        "access_count": int(db_result.get("access_count", 0)),
        "similarity": float(db_result.get("similarity", 0.0)),
        "source_type": db_result.get("source_type", "scoped")
    }


def transform_search_memories_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform complete SearchMemoriesResult from database to MCP-compatible format.

    Args:
        raw_result: Raw result dict from store.search_memories() with structure:
            {
                "success": bool,
                "results": List[dict],  # Raw DB results
                "total_results": int,
                "query": str | None,
                "filters": dict,
                "limit": int,
                "latest_first": bool,
                "error": str | None,
                "message": str | None
            }

    Returns:
        Transformed dict matching SearchMemoriesResult TypedDict
    """
    # Transform each result in the list
    transformed_results = [
        _format_search_result_for_mcp(result)
        for result in raw_result.get("results", [])
    ]

    # Return the full structure with transformed results
    return {
        "success": raw_result.get("success", False),
        "results": transformed_results,
        "total_results": raw_result.get("total_results", 0),
        "query": raw_result.get("query"),
        "filters": raw_result.get("filters", {}),
        "limit": raw_result.get("limit", 0),
        "latest_first": raw_result.get("latest_first", True),
        "error": raw_result.get("error"),
        "message": raw_result.get("message")
    }
