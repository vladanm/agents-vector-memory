"""
Integration tests for TypedDict structured outputs.

Tests Phase 1 feature: TypedDict definitions for consistent return structures.
"""

import pytest
from src.mcp_types import StoreMemoryResult, SearchMemoriesResult


@pytest.mark.integration
def test_store_memory_returns_typed_dict(store, sample_memory):
    """Verify store_memory returns properly structured dict."""
    result = store.store_memory(**sample_memory)

    # Check all required fields present
    assert "success" in result
    assert "memory_id" in result
    assert "memory_type" in result
    assert "agent_id" in result
    assert "session_id" in result
    assert "content_hash" in result
    assert "chunks_created" in result
    assert "created_at" in result
    assert "message" in result

    # Check types
    assert isinstance(result["success"], bool)
    assert isinstance(result["memory_id"], int)
    assert isinstance(result["chunks_created"], int)
    assert isinstance(result["message"], str)


@pytest.mark.integration
def test_search_memories_returns_typed_dict(store, sample_memory):
    """Verify search_memories returns properly structured dict."""
    # Insert test data
    store.store_memory(**sample_memory)

    result = store.search_memories(
        memory_type="session_context",
        limit=5
    )

    # Check structure
    assert "success" in result
    assert "results" in result
    assert "total_results" in result
    assert "filters" in result
    assert "limit" in result

    # Check types
    assert isinstance(result["success"], bool)
    assert isinstance(result["results"], list)
    assert isinstance(result["total_results"], int)

    # Check result items structure
    if result["results"]:
        item = result["results"][0]
        assert "id" in item
        assert "memory_type" in item
        assert "agent_id" in item
        assert "session_id" in item
        assert "content" in item


@pytest.mark.integration
def test_error_response_structure(store):
    """Verify error responses have consistent structure."""
    # Trigger validation error
    result = store.store_memory(
        memory_type="invalid_type",
        agent_id="test",
        session_id="test",
        content="test"
    )

    # Check error structure
    assert result["success"] is False
    assert "error" in result
    assert "message" in result
    assert isinstance(result["error"], str)
    assert isinstance(result["message"], str)


@pytest.mark.integration
def test_all_store_operations_return_structured(store, sample_memory):
    """Test all major store operations return structured outputs."""
    # store_memory
    result1 = store.store_memory(**sample_memory)
    assert "success" in result1
    assert "memory_id" in result1

    # search_memories
    result2 = store.search_memories(memory_type="session_context")
    assert "success" in result2
    assert "results" in result2

    # get_session_stats
    result3 = store.get_session_stats()
    assert "success" in result3

    # All should have success boolean
    assert isinstance(result1["success"], bool)
    assert isinstance(result2["success"], bool)
    assert isinstance(result3["success"], bool)


@pytest.mark.integration
def test_empty_agent_id_validation(store):
    """Test validation error for empty agent_id (Phase 1 feature)."""
    result = store.store_memory(
        memory_type="session_context",
        agent_id="",
        session_id="test",
        content="test"
    )

    assert not result["success"]
    assert "agent_id" in result.get("error", "").lower()


@pytest.mark.integration
def test_empty_session_id_validation(store):
    """Test validation error for empty session_id (Phase 1 feature)."""
    result = store.store_memory(
        memory_type="session_context",
        agent_id="test",
        session_id="",
        content="test"
    )

    assert not result["success"]
    assert "session_id" in result.get("error", "").lower()
