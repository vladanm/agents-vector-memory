"""
Shared pytest fixtures for all tests.

This module provides common fixtures for testing the vector memory MCP server,
including temporary databases, store instances, and sample data.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from src.session_memory_store import SessionMemoryStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    Yields:
        Path: Path to temporary database file

    Cleanup:
        Automatically removes the database after test completion
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        yield str(db_path)
        # Cleanup happens automatically when context exits


@pytest.fixture
def store(temp_db):
    """Create a SessionMemoryStore instance with temporary database.

    Args:
        temp_db: Temporary database path from temp_db fixture

    Yields:
        SessionMemoryStore: Store instance ready for testing
    """
    store = SessionMemoryStore(db_path=temp_db)
    yield store
    # Store will be garbage collected automatically


@pytest.fixture
def sample_memory():
    """Provide sample memory data for testing.

    Returns:
        dict: Sample memory entry with all fields
    """
    return {
        "memory_type": "session_context",
        "agent_id": "test-agent",
        "session_id": "test-session-001",
        "session_iter": 1,
        "task_code": "test-task",
        "content": "This is a test memory entry for unit testing.",
        "title": "Test Memory",
        "description": "A sample memory for testing purposes",
        "tags": ["test", "sample", "unit"],
        "metadata": {
            "test": True,
            "priority": "high",
            "created_by": "pytest"
        }
    }


@pytest.fixture
def sample_memory_large():
    """Provide large memory data for chunking tests.

    Returns:
        dict: Large memory entry that will trigger chunking
    """
    # Generate content > 1000 tokens to trigger chunking
    content = "\n\n".join([
        f"# Section {i}\n\nThis is section {i} with content. " * 50
        for i in range(10)
    ])

    return {
        "memory_type": "knowledge_base",
        "agent_id": "test-agent",
        "session_id": "test-session-002",
        "session_iter": 1,
        "task_code": "test-chunking",
        "content": content,
        "title": "Large Test Document",
        "description": "A large document for chunking tests",
        "tags": ["test", "large", "chunking"],
        "metadata": {"test": True},
        "auto_chunk": True
    }


@pytest.fixture
def multiple_memories():
    """Provide multiple memory entries for batch testing.

    Returns:
        list[dict]: List of memory entries for batch operations
    """
    base_memory = {
        "memory_type": "session_context",
        "agent_id": "test-agent",
        "session_id": "test-session-batch",
        "content": "Test content",
        "title": "Test Memory",
        "tags": ["test"],
        "metadata": {}
    }

    memories = []
    for i in range(5):
        memory = base_memory.copy()
        memory["session_iter"] = i + 1
        memory["task_code"] = f"task-{i}"
        memory["content"] = f"Test content for iteration {i + 1}"
        memory["title"] = f"Test Memory {i + 1}"
        memories.append(memory)

    return memories


@pytest.fixture
def search_query_data():
    """Provide sample data for search query tests.

    Returns:
        dict: Search query parameters and expected results
    """
    return {
        "query": "test semantic search",
        "filters": {
            "memory_type": "session_context",
            "agent_id": "test-agent",
            "session_id": "test-session-001"
        },
        "limit": 5,
        "similarity_threshold": 0.7
    }


@pytest.fixture(autouse=True)
def reset_test_environment():
    """Automatically reset environment before each test.

    This fixture runs automatically before each test to ensure
    a clean state. Add any global cleanup or setup here.
    """
    # Setup code (runs before each test)
    yield
    # Teardown code (runs after each test)
    pass


# Markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_embedding: marks tests that need embedding model"
    )
