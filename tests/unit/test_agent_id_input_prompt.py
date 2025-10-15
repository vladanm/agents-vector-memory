"""Tests for agent_id parameter in store_input_prompt and search_input_prompts."""

import pytest
import sys
from pathlib import Path

# Add main.py to path so we can import the tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main import (
    store_input_prompt,
    search_input_prompts,
    initialize_store
)


@pytest.fixture(scope="function")
def test_db(tmp_path):
    """Create a test database for each test."""
    db_path = tmp_path / "test_agent_id.db"
    initialize_store(database_path=str(db_path))
    yield db_path
    # Cleanup happens automatically with tmp_path


def test_store_input_prompt_with_different_agents(test_db):
    """Test that different agents can store their own input prompts."""

    # Store prompt from main orchestrator
    result1 = store_input_prompt(
        agent_id="main-orchestrator",
        session_id="test-session",
        session_iter="v1",
        content="Analyze the codebase"
    )
    assert result1["success"] is True
    assert result1["agent_id"] == "main-orchestrator"

    # Store prompt from sub-agent
    result2 = store_input_prompt(
        agent_id="code-explorer-agent",
        session_id="test-session",
        session_iter="v1",
        content="Find all function definitions"
    )
    assert result2["success"] is True
    assert result2["agent_id"] == "code-explorer-agent"

    # Verify both stored with different IDs
    assert result1["memory_id"] != result2["memory_id"]


def test_search_input_prompts_by_agent(test_db):
    """Test searching input prompts filtered by agent_id."""

    # Store prompts from two different agents
    store_input_prompt(
        agent_id="main-orchestrator",
        session_id="test-session",
        session_iter="v1",
        content="Prompt 1"
    )

    store_input_prompt(
        agent_id="code-explorer-agent",
        session_id="test-session",
        session_iter="v1",
        content="Prompt 2"
    )

    # Search for main-orchestrator prompts only
    results = search_input_prompts(
        session_id="test-session",
        agent_id="main-orchestrator",
        limit=10
    )

    assert results["success"] is True
    assert results["total_results"] == 1
    assert all(m["agent_id"] == "main-orchestrator" for m in results["results"])


def test_search_input_prompts_all_agents(test_db):
    """Test searching input prompts across all agents."""

    # Store prompts from two different agents
    store_input_prompt(
        agent_id="main-orchestrator",
        session_id="test-session",
        session_iter="v1",
        content="Prompt 1"
    )

    store_input_prompt(
        agent_id="code-explorer-agent",
        session_id="test-session",
        session_iter="v1",
        content="Prompt 2"
    )

    # Search without agent_id filter (all agents)
    results = search_input_prompts(
        session_id="test-session",
        agent_id=None,  # Search all agents
        limit=10
    )

    assert results["success"] is True
    assert results["total_results"] == 2

    # Verify both agents present
    agent_ids = {m["agent_id"] for m in results["results"]}
    assert "main-orchestrator" in agent_ids
    assert "code-explorer-agent" in agent_ids


def test_store_input_prompt_required_agent_id():
    """Test that agent_id is required for store_input_prompt."""
    # This test verifies the function signature requires agent_id
    import inspect
    sig = inspect.signature(store_input_prompt)
    params = list(sig.parameters.keys())

    # agent_id should be the first parameter
    assert params[0] == "agent_id"

    # agent_id should not have a default value (required)
    assert sig.parameters["agent_id"].default == inspect.Parameter.empty


def test_search_input_prompts_optional_agent_id():
    """Test that agent_id is optional for search_input_prompts."""
    import inspect
    sig = inspect.signature(search_input_prompts)

    # agent_id should have a default value of None
    assert sig.parameters["agent_id"].default is None


def test_multi_agent_session_isolation(test_db):
    """Test that different sessions are properly isolated even with same agent."""

    # Store prompts in session 1
    result1 = store_input_prompt(
        agent_id="main-orchestrator",
        session_id="session-1",
        session_iter="v1",
        content="Session 1 prompt"
    )

    # Store prompts in session 2
    result2 = store_input_prompt(
        agent_id="main-orchestrator",
        session_id="session-2",
        session_iter="v1",
        content="Session 2 prompt"
    )

    # Search session 1 only
    results1 = search_input_prompts(
        session_id="session-1",
        agent_id="main-orchestrator",
        limit=10
    )

    assert results1["success"] is True
    assert results1["total_results"] == 1
    assert results1["results"][0]["session_id"] == "session-1"

    # Search session 2 only
    results2 = search_input_prompts(
        session_id="session-2",
        agent_id="main-orchestrator",
        limit=10
    )

    assert results2["success"] is True
    assert results2["total_results"] == 1
    assert results2["results"][0]["session_id"] == "session-2"


def test_store_input_prompt_with_metadata(test_db):
    """Test storing input prompt with agent_id and custom metadata."""

    result = store_input_prompt(
        agent_id="test-agent",
        session_id="test-session",
        session_iter="v1",
        content="Test prompt with metadata",
        title="Test Title",
        description="Test description",
        tags=["test", "metadata"],
        metadata={"custom_field": "custom_value"}
    )

    assert result["success"] is True
    assert result["agent_id"] == "test-agent"
    assert result["memory_id"] is not None

    # Verify we can retrieve it
    results = search_input_prompts(
        session_id="test-session",
        agent_id="test-agent",
        limit=1
    )

    assert results["success"] is True
    assert results["total_results"] == 1
    memory = results["results"][0]
    assert memory["agent_id"] == "test-agent"
    assert memory["title"] == "Test Title"
    assert "test" in memory.get("tags", [])
