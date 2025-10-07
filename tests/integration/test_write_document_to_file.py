"""
Integration tests for write_document_to_file functionality.

Tests the document writing tool with various scenarios including:
- Auto-generated file paths
- Custom file paths
- Metadata (YAML frontmatter) inclusion
- Error handling
"""

import pytest
import os
from pathlib import Path


@pytest.fixture
def test_memory_id(store):
    """Create a test memory document for write tests.

    Returns:
        int: Memory ID of the created test document
    """
    content = "# Test Document\n\n" + ("This is test content. " * 100)

    result = store.store_memory(
        memory_type='reports',
        agent_id='verification-agent',
        session_id='test-write-tool',
        session_iter=1,
        content=content,
        title='Test Document',
        description='Test document for write_document_to_file'
    )

    assert result.get('success'), f"Failed to create test memory: {result}"
    return result['memory_id']


@pytest.mark.integration
def test_write_to_auto_generated_path_with_metadata(store, test_memory_id, tmp_path):
    """Test writing document to auto-generated path with metadata."""
    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        include_metadata=True
    )

    assert write_result.get('success'), f"Write failed: {write_result}"

    file_path = write_result['file_path']
    assert 'file_size_human' in write_result
    assert 'estimated_tokens' in write_result

    # Verify file exists
    assert os.path.exists(file_path), f"File not found: {file_path}"

    # Verify YAML frontmatter present
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    assert '---' in file_content[:100], "YAML frontmatter not found"
    assert 'Test Document' in file_content, "Title not in content"

    # Cleanup
    os.remove(file_path)


@pytest.mark.integration
def test_write_to_custom_path_without_metadata(store, test_memory_id, tmp_path):
    """Test writing document to custom path without metadata."""
    custom_path = str(tmp_path / 'test_custom_write.md')

    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        output_path=custom_path,
        include_metadata=False
    )

    assert write_result.get('success'), f"Write failed: {write_result}"
    assert write_result['file_path'] == custom_path, \
        f"Path mismatch: expected {custom_path}, got {write_result['file_path']}"

    # Verify file exists
    assert os.path.exists(custom_path), f"Custom path file not found: {custom_path}"

    # Verify no YAML frontmatter
    with open(custom_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Should not have frontmatter
    first_line = file_content.split('\n')[0]
    assert first_line != '---', "YAML frontmatter should not be present"


@pytest.mark.integration
def test_write_to_custom_path_with_metadata(store, test_memory_id, tmp_path):
    """Test writing document to custom path with metadata."""
    custom_path = str(tmp_path / 'test_with_metadata.md')

    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        output_path=custom_path,
        include_metadata=True
    )

    assert write_result.get('success'), f"Write failed: {write_result}"
    assert write_result['file_path'] == custom_path

    # Verify file has metadata
    with open(custom_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    assert '---' in file_content[:100], "YAML frontmatter not found"
    assert 'Test Document' in file_content


@pytest.mark.integration
def test_error_handling_invalid_memory_id(store):
    """Test error handling for invalid memory ID."""
    write_result = store.write_document_to_file(memory_id=999999999)

    assert not write_result.get('success'), "Should have failed with invalid memory ID"
    assert write_result.get('error_code') == 'MEMORY_NOT_FOUND', \
        f"Wrong error code: {write_result.get('error_code')}"


@pytest.mark.integration
def test_error_handling_invalid_parameter(store, test_memory_id):
    """Test error handling for invalid parameter."""
    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        format='invalid'
    )

    assert not write_result.get('success'), "Should have failed with invalid parameter"
    assert write_result.get('error_code') == 'INVALID_PARAMETER', \
        f"Wrong error code: {write_result.get('error_code')}"


@pytest.mark.integration
def test_error_handling_non_absolute_path(store, test_memory_id):
    """Test error handling for non-absolute path."""
    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        output_path='relative/path.md'
    )

    assert not write_result.get('success'), "Should have failed with non-absolute path"
    assert write_result.get('error_code') == 'INVALID_PATH', \
        f"Wrong error code: {write_result.get('error_code')}"


@pytest.mark.integration
def test_all_error_codes_implemented(store):
    """Verify all required error codes are in write_document_to_file implementation."""
    import inspect

    source = inspect.getsource(store.write_document_to_file)

    required_codes = [
        'MEMORY_NOT_FOUND',
        'INVALID_PATH',
        'INVALID_PARAMETER',
        'PERMISSION_DENIED',
        'DISK_FULL',
        'WRITE_FAILED',
        'RECONSTRUCTION_FAILED'
    ]

    for code in required_codes:
        assert code in source, f"Error code {code} not found in implementation"


@pytest.mark.integration
def test_file_size_and_token_estimation(store, test_memory_id, tmp_path):
    """Test that file size and token estimation are returned correctly."""
    custom_path = str(tmp_path / 'test_size_tokens.md')

    write_result = store.write_document_to_file(
        memory_id=test_memory_id,
        output_path=custom_path
    )

    assert write_result.get('success')
    assert 'file_size_human' in write_result
    assert 'estimated_tokens' in write_result

    # Verify file size is reasonable (>0)
    file_size = os.path.getsize(custom_path)
    assert file_size > 0, "File should have content"

    # Verify token estimation is reasonable
    assert write_result['estimated_tokens'] > 0, "Token count should be > 0"
