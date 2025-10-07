"""
Unit tests for direct chunking functionality.

Tests that bypassing MCP and calling store_memory directly with auto_chunk=True
properly triggers chunking and stores chunks.
"""

import pytest
import sqlite3


TEST_CONTENT = """---
title: Direct Chunking Test
---

# Main Section

This is the main section with some content.

## Subsection 1

Content for subsection 1.

### Deep Subsection

More detailed content.

## Subsection 2

Final section content.
"""


@pytest.mark.unit
def test_direct_store_memory_with_chunking(store, temp_db):
    """Test that calling store_memory directly with auto_chunk=True works."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    assert result.get('success'), f"store_memory failed: {result}"
    assert 'memory_id' in result
    assert result['memory_id'] > 0


@pytest.mark.unit
def test_direct_chunking_creates_chunks(store, temp_db):
    """Test that direct chunking actually creates chunks in database."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    memory_id = result.get('memory_id')
    assert memory_id is not None

    # Query database
    conn = sqlite3.connect(temp_db)
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
        (memory_id,)
    ).fetchone()[0]
    conn.close()

    assert chunk_count > 0, "No chunks created"


@pytest.mark.unit
def test_direct_chunking_chunk_structure(store, temp_db):
    """Test that chunks have proper structure."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    memory_id = result.get('memory_id')

    conn = sqlite3.connect(temp_db)
    chunks = conn.execute("""
        SELECT chunk_index, chunk_type, header_path, level, token_count, content
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (memory_id,)).fetchall()
    conn.close()

    assert len(chunks) > 0, "No chunks found"

    for chunk_index, chunk_type, header_path, level, token_count, content in chunks:
        assert chunk_index >= 0, f"Invalid chunk_index: {chunk_index}"
        assert chunk_type is not None, f"Chunk {chunk_index} missing chunk_type"
        assert level >= 0, f"Chunk {chunk_index} has invalid level: {level}"
        assert token_count > 0, f"Chunk {chunk_index} has invalid token_count: {token_count}"
        assert content is not None and len(content) > 0, f"Chunk {chunk_index} has empty content"


@pytest.mark.unit
def test_direct_chunking_chunk_details(store, temp_db):
    """Test that chunks contain expected details."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    memory_id = result.get('memory_id')

    conn = sqlite3.connect(temp_db)
    chunks = conn.execute("""
        SELECT chunk_index, header_path, level, content
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (memory_id,)).fetchall()
    conn.close()

    # Combine all content
    all_content = ' '.join([chunk[3] for chunk in chunks])

    # Should contain key phrases from original
    assert ('Main Section' in all_content or 'Subsection' in all_content), \
        "Chunks should contain original document content"


@pytest.mark.unit
def test_direct_chunking_header_paths(store, temp_db):
    """Test that header paths are extracted."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    memory_id = result.get('memory_id')

    conn = sqlite3.connect(temp_db)
    header_paths = conn.execute("""
        SELECT header_path
        FROM memory_chunks
        WHERE parent_id = ?
    """, (memory_id,)).fetchall()
    conn.close()

    # All chunks should have header_path (even if empty string)
    assert all(path[0] is not None for path in header_paths), \
        "All chunks should have header_path field"


@pytest.mark.unit
def test_direct_chunking_token_counts(store, temp_db):
    """Test that token counts are reasonable."""
    result = store.store_memory(
        memory_type="reports",
        agent_id="direct-test",
        session_id="direct-session",
        session_iter=1,
        task_code="direct-test",
        content=TEST_CONTENT,
        title="Direct Test",
        auto_chunk=True
    )

    memory_id = result.get('memory_id')

    conn = sqlite3.connect(temp_db)
    token_counts = conn.execute("""
        SELECT token_count
        FROM memory_chunks
        WHERE parent_id = ?
    """, (memory_id,)).fetchall()
    conn.close()

    for token_count_tuple in token_counts:
        token_count = token_count_tuple[0]
        assert token_count > 0, "Token count should be positive"
        assert token_count < 5000, f"Token count {token_count} seems unreasonably high"
