"""
Integration tests for store_memory with auto_chunk functionality.

Tests that the auto_chunk parameter properly triggers chunking
and stores chunks in the database.
"""

import pytest
import sqlite3


TEST_CONTENT = """---
title: Direct Store Test
---

# Main Header

This is a test document with multiple sections.

## Section 1

Content for section 1 with substantial text to ensure chunking happens.

### Subsection 1.1

More detailed content in subsection 1.1.

## Section 2

Content for section 2.

### Subsection 2.1

Detailed content here.

### Subsection 2.2

More content in 2.2.

## Section 3

Final section with concluding remarks.
"""


@pytest.fixture
def chunked_memory_id(store):
    """Store a document with auto_chunk enabled.

    Returns:
        int: Memory ID of the stored document
    """
    result = store.store_memory(
        memory_type="reports",
        agent_id="test-agent",
        session_id="direct-test-session",
        session_iter=1,
        task_code="chunking-test",
        content=TEST_CONTENT,
        title="Direct Store Chunk Test",
        auto_chunk=True
    )

    assert result.get('success'), f"Failed to store document: {result}"
    return result['memory_id']


@pytest.mark.integration
def test_store_with_auto_chunk_returns_success(store, chunked_memory_id):
    """Test that store_memory with auto_chunk returns success."""
    assert chunked_memory_id is not None
    assert isinstance(chunked_memory_id, int)
    assert chunked_memory_id > 0


@pytest.mark.integration
def test_chunks_stored_in_database(store, chunked_memory_id, temp_db):
    """Test that chunks are actually stored in the database."""
    conn = sqlite3.connect(temp_db)

    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
        (chunked_memory_id,)
    ).fetchone()[0]

    conn.close()

    assert chunk_count > 0, "No chunks found in database"
    assert chunk_count >= 2, f"Expected at least 2 chunks, got {chunk_count}"


@pytest.mark.integration
def test_chunk_structure(store, chunked_memory_id, temp_db):
    """Test that chunks have proper structure and metadata."""
    conn = sqlite3.connect(temp_db)

    chunks = conn.execute("""
        SELECT chunk_index, chunk_type, header_path, level, content
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (chunked_memory_id,)).fetchall()

    conn.close()

    assert len(chunks) > 0, "No chunks retrieved"

    # Verify first chunk
    chunk_index, chunk_type, header_path, level, content = chunks[0]
    assert chunk_index == 0, "First chunk should have index 0"
    assert chunk_type is not None, "Chunk type should not be None"
    assert content is not None and len(content) > 0, "Chunk content should not be empty"


@pytest.mark.integration
def test_chunk_header_paths(store, chunked_memory_id, temp_db):
    """Test that chunks have header paths extracted."""
    conn = sqlite3.connect(temp_db)

    chunks = conn.execute("""
        SELECT chunk_index, header_path
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (chunked_memory_id,)).fetchall()

    conn.close()

    # At least one chunk should have a header path
    header_paths = [chunk[1] for chunk in chunks if chunk[1]]
    assert len(header_paths) > 0, "No header paths found in chunks"


@pytest.mark.integration
def test_chunk_levels(store, chunked_memory_id, temp_db):
    """Test that chunks have proper level values."""
    conn = sqlite3.connect(temp_db)

    chunks = conn.execute("""
        SELECT level
        FROM memory_chunks
        WHERE parent_id = ?
    """, (chunked_memory_id,)).fetchall()

    conn.close()

    levels = [chunk[0] for chunk in chunks]
    assert all(level >= 0 for level in levels), "All levels should be >= 0"


@pytest.mark.integration
def test_chunk_content_not_empty(store, chunked_memory_id, temp_db):
    """Test that all chunks have non-empty content."""
    conn = sqlite3.connect(temp_db)

    chunks = conn.execute("""
        SELECT chunk_index, content
        FROM memory_chunks
        WHERE parent_id = ?
    """, (chunked_memory_id,)).fetchall()

    conn.close()

    for chunk_index, content in chunks:
        assert content is not None, f"Chunk {chunk_index} has None content"
        assert len(content) > 0, f"Chunk {chunk_index} has empty content"


@pytest.mark.integration
def test_chunk_preview(store, chunked_memory_id, temp_db):
    """Test that chunk content contains expected text."""
    conn = sqlite3.connect(temp_db)

    chunks = conn.execute("""
        SELECT content
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (chunked_memory_id,)).fetchall()

    conn.close()

    # All chunk contents combined should contain key phrases from original
    all_content = ' '.join([chunk[0] for chunk in chunks])

    assert 'Main Header' in all_content or 'Section' in all_content, \
        "Chunks should contain content from original document"
