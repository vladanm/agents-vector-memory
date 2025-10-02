#!/usr/bin/env python3
"""
Test script for new vector-memory-2-mcp features
=================================================

Tests:
1. Document chunking
2. Markdown structure preservation
3. YAML frontmatter extraction
4. Document reconstruction
5. Delete memory
6. Cleanup old memories
7. knowledge_base memory type
8. Backward compatibility
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent to path for proper imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.session_memory_store import SessionMemoryStore


def test_basic_storage():
    """Test 1: Basic storage still works (backward compatibility)"""
    print("\n" + "="*60)
    print("TEST 1: Basic Storage (Backward Compatibility)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        result = store.store_memory(
            memory_type="working_memory",
            agent_id="main",
            session_id="test-session-1",
            content="This is a simple test memory without chunking",
            session_iter=1,
            title="Test Memory"
        )

        assert result["success"], f"Basic storage failed: {result}"
        assert "memory_id" in result
        print(f"✅ Basic storage works! Memory ID: {result['memory_id']}")


def test_document_chunking():
    """Test 2: Document chunking with markdown"""
    print("\n" + "="*60)
    print("TEST 2: Document Chunking")
    print("="*60)

    markdown_content = """# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1 goes here. It should be long enough to potentially span multiple chunks if needed.

## Section 2

More content for section 2. We want to test how the chunker handles hierarchical markdown structure.

### Subsection 2.1

Even more detailed content in a subsection.

## Section 3

Final section with some content."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        result = store.store_memory(
            memory_type="reports",
            agent_id="specialized-agent",
            session_id="test-session-2",
            content=markdown_content,
            session_iter=1,
            title="Test Markdown Report",
            auto_chunk=True  # Enable chunking
        )

        assert result["success"], f"Chunking storage failed: {result}"
        assert "memory_id" in result

        if "chunks_stored" in result:
            print(f"✅ Document chunked! Chunks: {result['chunks_stored']}")
        else:
            print("⚠️  No chunks stored (content may be too small)")

        # Test reconstruction
        memory_id = result["memory_id"]
        recon_result = store.reconstruct_document(memory_id)

        assert recon_result["success"], f"Reconstruction failed: {recon_result}"
        print(f"✅ Document reconstructed! Chunk count: {recon_result.get('chunk_count', 0)}")


def test_yaml_frontmatter():
    """Test 3: YAML frontmatter extraction"""
    print("\n" + "="*60)
    print("TEST 3: YAML Frontmatter Extraction")
    print("="*60)

    content_with_yaml = """---
title: Test Document
author: Test Agent
version: 1.0
---

# Actual Content

This is the main content after the YAML frontmatter."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        # Extract frontmatter
        clean_content, frontmatter = store._extract_yaml_frontmatter(content_with_yaml)

        assert "title" in frontmatter, "YAML frontmatter not extracted"
        assert "# Actual Content" in clean_content, "Clean content incorrect"
        print(f"✅ YAML extracted: {frontmatter}")
        print(f"✅ Clean content starts with: {clean_content[:30]}")


def test_delete_memory():
    """Test 4: Delete memory functionality"""
    print("\n" + "="*60)
    print("TEST 4: Delete Memory")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        # Store a memory
        result = store.store_memory(
            memory_type="working_memory",
            agent_id="main",
            session_id="test-session-3",
            content="This memory will be deleted",
            session_iter=1
        )

        memory_id = result["memory_id"]
        print(f"   Stored memory ID: {memory_id}")

        # Delete it
        delete_result = store.delete_memory(memory_id)

        assert delete_result["success"], f"Deletion failed: {delete_result}"
        print(f"✅ Memory deleted successfully!")

        # Try to delete again (should fail)
        delete_again = store.delete_memory(memory_id)
        assert not delete_again["success"], "Should fail when deleting non-existent memory"
        print(f"✅ Correctly handles deletion of non-existent memory")


def test_cleanup_old_memories():
    """Test 5: Cleanup old memories"""
    print("\n" + "="*60)
    print("TEST 5: Cleanup Old Memories")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        # Store some memories
        for i in range(3):
            store.store_memory(
                memory_type="working_memory",
                agent_id="main",
                session_id=f"test-session-{i}",
                content=f"Test memory {i}",
                session_iter=1
            )

        # Try cleanup (should find nothing since memories are new)
        cleanup_result = store.cleanup_old_memories(older_than_days=30)

        assert cleanup_result["success"], f"Cleanup failed: {cleanup_result}"
        print(f"✅ Cleanup executed! Deleted: {cleanup_result['deleted_count']}")


def test_knowledge_base_type():
    """Test 6: Knowledge base memory type"""
    print("\n" + "="*60)
    print("TEST 6: Knowledge Base Type")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        result = store.store_memory(
            memory_type="knowledge_base",
            agent_id="main",
            session_id="global-kb",
            content="This is shared knowledge accessible across sessions",
            session_iter=1,
            title="Shared Knowledge",
            auto_chunk=True
        )

        assert result["success"], f"Knowledge base storage failed: {result}"
        assert result["memory_type"] == "knowledge_base"
        print(f"✅ Knowledge base memory stored! ID: {result['memory_id']}")


def test_backward_compatibility():
    """Test 7: Existing functionality still works"""
    print("\n" + "="*60)
    print("TEST 7: Backward Compatibility")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SessionMemoryStore(db_path)

        # Test all original memory types
        types_to_test = [
            "session_context",
            "input_prompt",
            "reports",
            "working_memory",
            "system_memory",
            "report_observations"
        ]

        for mem_type in types_to_test:
            result = store.store_memory(
                memory_type=mem_type,
                agent_id="main",
                session_id="compat-test",
                content=f"Test content for {mem_type}",
                session_iter=1
            )
            assert result["success"], f"Failed for {mem_type}: {result}"

        print(f"✅ All {len(types_to_test)} original memory types work!")

        # Test generic search functionality
        search_results = store.search_memories(
            memory_type="working_memory",
            agent_id="main",
            session_id="compat-test",
            query="test",
            limit=10
        )

        assert search_results["success"], "Search failed"
        print(f"✅ Search functionality works!")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*60)
    print("VECTOR-MEMORY-2-MCP NEW FEATURES TEST SUITE")
    print("="*60)

    tests = [
        test_basic_storage,
        test_document_chunking,
        test_yaml_frontmatter,
        test_delete_memory,
        test_cleanup_old_memories,
        test_knowledge_base_type,
        test_backward_compatibility
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)