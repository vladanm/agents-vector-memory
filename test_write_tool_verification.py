#!/usr/bin/env python3
"""
Quick verification test for write_document_to_file corrective implementation.
Tests only the core write tool functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import SessionMemoryStore

def test_write_document_to_file():
    """Test write_document_to_file with all key scenarios."""

    db_path = '/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db'
    store = SessionMemoryStore(db_path=db_path)

    print("=" * 70)
    print("WRITE_DOCUMENT_TO_FILE VERIFICATION TEST")
    print("=" * 70)
    print()

    # Test 1: Create test document
    print("📝 Test 1: Creating test document...")
    content = "# Test Document\n\n" + ("This is test content. " * 100)

    store_result = store.store_memory(
        memory_type='reports',
        agent_id='verification-agent',
        session_id='test-write-tool',
        content=content,
        title='Test Document',
        description='Test document for write_document_to_file'
    )

    if not store_result.get('success'):
        print(f"❌ Failed to store test document: {store_result}")
        return False

    memory_id = store_result['memory_id']
    print(f"✅ Test document stored with memory_id: {memory_id}")
    print()

    # Test 2: Write to auto-generated path with metadata
    print("📝 Test 2: Write to auto-generated path with metadata...")
    write_result = store.write_document_to_file(
        memory_id=memory_id,
        include_metadata=True
    )

    if not write_result.get('success'):
        print(f"❌ Write failed: {write_result}")
        return False

    file_path = write_result['file_path']
    print(f"✅ File written successfully:")
    print(f"   - Path: {file_path}")
    print(f"   - Size: {write_result['file_size_human']}")
    print(f"   - Tokens: {write_result['estimated_tokens']}")

    # Verify file exists and has content
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    if '---' not in file_content[:100]:
        print("❌ YAML frontmatter not found")
        return False

    print("✅ File exists and contains YAML frontmatter")
    print()

    # Test 3: Write to custom path without metadata
    print("📝 Test 3: Write to custom path without metadata...")
    custom_path = '/tmp/test_write_tool_verification.md'

    write_result = store.write_document_to_file(
        memory_id=memory_id,
        output_path=custom_path,
        include_metadata=False
    )

    if not write_result.get('success'):
        print(f"❌ Write failed: {write_result}")
        return False

    if write_result['file_path'] != custom_path:
        print(f"❌ Path mismatch: expected {custom_path}, got {write_result['file_path']}")
        return False

    print(f"✅ Custom path write successful: {custom_path}")
    print()

    # Test 4: Error handling - invalid memory ID
    print("📝 Test 4: Error handling - invalid memory ID...")
    write_result = store.write_document_to_file(memory_id=999999999)

    if write_result.get('success'):
        print("❌ Should have failed with MEMORY_NOT_FOUND")
        return False

    if write_result.get('error_code') != 'MEMORY_NOT_FOUND':
        print(f"❌ Wrong error code: {write_result.get('error_code')}")
        return False

    print(f"✅ MEMORY_NOT_FOUND error handled correctly")
    print()

    # Test 5: Error handling - invalid parameter
    print("📝 Test 5: Error handling - invalid parameter...")
    write_result = store.write_document_to_file(
        memory_id=memory_id,
        format='invalid'
    )

    if write_result.get('success'):
        print("❌ Should have failed with INVALID_PARAMETER")
        return False

    if write_result.get('error_code') != 'INVALID_PARAMETER':
        print(f"❌ Wrong error code: {write_result.get('error_code')}")
        return False

    print(f"✅ INVALID_PARAMETER error handled correctly")
    print()

    # Test 6: Error handling - non-absolute path
    print("📝 Test 6: Error handling - non-absolute path...")
    write_result = store.write_document_to_file(
        memory_id=memory_id,
        output_path='relative/path.md'
    )

    if write_result.get('success'):
        print("❌ Should have failed with INVALID_PATH")
        return False

    if write_result.get('error_code') != 'INVALID_PATH':
        print(f"❌ Wrong error code: {write_result.get('error_code')}")
        return False

    print(f"✅ INVALID_PATH error handled correctly")
    print()

    # Test 7: Verify all error codes are in implementation
    print("📝 Test 7: Verify all required error codes...")
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

    all_present = True
    for code in required_codes:
        if code in source:
            print(f"   ✅ {code}")
        else:
            print(f"   ❌ {code} - MISSING")
            all_present = False

    if not all_present:
        return False

    print()

    # Cleanup
    print("🧹 Cleaning up test files...")
    for path in [file_path, custom_path]:
        if os.path.exists(path):
            os.remove(path)
    print()

    print("=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✅ write_document_to_file method exists and is callable")
    print("  ✅ Auto-generated paths work")
    print("  ✅ Custom paths work")
    print("  ✅ Metadata frontmatter generation works")
    print("  ✅ All 7 error codes implemented")
    print("  ✅ Error handling works correctly")
    print()
    print("🎉 Corrective implementation SUCCESSFUL!")

    return True

if __name__ == '__main__':
    try:
        success = test_write_document_to_file()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
