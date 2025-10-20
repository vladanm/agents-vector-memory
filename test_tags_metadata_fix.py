#!/usr/bin/env python3
"""
Test script to verify tags and metadata are returned in semantic search results.

This script tests the fix for the reported issue where tags and metadata were
not being returned in FINE and MEDIUM granularity searches.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow package imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from src.session_memory_store import SessionMemoryStore


def test_tags_metadata_fine_granularity():
    """Test that tags and metadata are returned in fine granularity search."""
    print("\n" + "=" * 80)
    print("TEST 1: Fine Granularity Search - Tags & Metadata Return")
    print("=" * 80)

    # Initialize store with test database
    db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
    store = SessionMemoryStore(db_path=db_path)

    # Store a test memory with tags and metadata
    print("\n1. Storing test memory with tags and metadata...")
    result = store.store_memory(
        memory_type="knowledge_base",
        agent_id="test-agent",
        session_id="test-session",
        content="This is test content for verifying tags and metadata return in semantic search. It contains information about Python programming and data structures.",
        title="Test Document for Tags/Metadata",
        tags=["test", "verification", "important"],
        metadata={"priority": "high", "confidence": 0.95, "source": "test_script"}
    )

    if not result["success"]:
        print(f"❌ Failed to store memory: {result.get('message')}")
        return False

    memory_id = result["memory_id"]
    print(f"✅ Memory stored successfully: ID = {memory_id}")
    print(f"   Tags: {['test', 'verification', 'important']}")
    print(f"   Metadata: {{'priority': 'high', 'confidence': 0.95, 'source': 'test_script'}}")

    # Search with fine granularity
    print("\n2. Performing fine granularity search...")
    search_result = store.search_with_granularity(
        memory_type="knowledge_base",
        granularity="fine",
        query="Python programming data structures",
        limit=5
    )

    if not search_result["success"]:
        print(f"❌ Search failed: {search_result.get('message')}")
        return False

    results = search_result.get("results", [])
    if not results:
        print("❌ No results returned")
        return False

    print(f"✅ Search returned {len(results)} result(s)")

    # Check first result for tags and metadata
    first_result = results[0]
    print(f"\n3. Checking first result for tags and metadata...")
    print(f"   Result keys: {list(first_result.keys())}")

    has_tags = "tags" in first_result
    has_metadata = "metadata" in first_result

    if not has_tags:
        print("❌ FAILED: 'tags' field is missing from result")
        return False

    if not has_metadata:
        print("❌ FAILED: 'metadata' field is missing from result")
        return False

    print(f"✅ Tags present: {first_result['tags']}")
    print(f"✅ Metadata present: {first_result['metadata']}")

    # Verify values match what we stored
    expected_tags = ["test", "verification", "important"]
    expected_metadata = {"priority": "high", "confidence": 0.95, "source": "test_script"}

    if set(first_result["tags"]) == set(expected_tags):
        print("✅ Tags match expected values")
    else:
        print(f"⚠️  Tags don't match: expected {expected_tags}, got {first_result['tags']}")

    if first_result["metadata"] == expected_metadata:
        print("✅ Metadata matches expected values")
    else:
        print(f"⚠️  Metadata doesn't match: expected {expected_metadata}, got {first_result['metadata']}")

    # Cleanup
    print("\n4. Cleaning up test data...")
    store.delete_memory(memory_id)
    print("✅ Test memory deleted")

    return True


def test_tags_metadata_medium_granularity():
    """Test that tags and metadata are returned in medium granularity search."""
    print("\n" + "=" * 80)
    print("TEST 2: Medium Granularity Search - Tags & Metadata Return")
    print("=" * 80)

    # Initialize store with test database
    db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
    store = SessionMemoryStore(db_path=db_path)

    # Store a test memory with tags and metadata
    print("\n1. Storing test memory with tags and metadata...")
    result = store.store_memory(
        memory_type="reports",
        agent_id="test-agent",
        session_id="test-session-medium",
        content="""# Analysis Report

## Introduction
This is a test report for verifying tags and metadata return in medium granularity search.

## Findings
The system performs well under various conditions. Python programming is excellent for data processing.

## Conclusion
All tests passed successfully.
""",
        title="Test Report for Medium Granularity",
        tags=["report", "analysis", "medium-test"],
        metadata={"test_type": "medium_granularity", "version": "1.0"}
    )

    if not result["success"]:
        print(f"❌ Failed to store memory: {result.get('message')}")
        return False

    memory_id = result["memory_id"]
    print(f"✅ Memory stored successfully: ID = {memory_id}")
    print(f"   Tags: {['report', 'analysis', 'medium-test']}")
    print(f"   Metadata: {{'test_type': 'medium_granularity', 'version': '1.0'}}")

    # Search with medium granularity
    print("\n2. Performing medium granularity search...")
    search_result = store.search_with_granularity(
        memory_type="reports",
        granularity="medium",
        query="Python programming data processing",
        session_id="test-session-medium",
        limit=5
    )

    if not search_result["success"]:
        print(f"❌ Search failed: {search_result.get('message')}")
        return False

    results = search_result.get("results", [])
    if not results:
        print("❌ No results returned")
        return False

    print(f"✅ Search returned {len(results)} result(s)")

    # Check first result for tags and metadata
    first_result = results[0]
    print(f"\n3. Checking first result for tags and metadata...")
    print(f"   Result keys: {list(first_result.keys())}")

    has_tags = "tags" in first_result
    has_metadata = "metadata" in first_result

    if not has_tags:
        print("❌ FAILED: 'tags' field is missing from result")
        return False

    if not has_metadata:
        print("❌ FAILED: 'metadata' field is missing from result")
        return False

    print(f"✅ Tags present: {first_result['tags']}")
    print(f"✅ Metadata present: {first_result['metadata']}")

    # Verify values match what we stored
    expected_tags = ["report", "analysis", "medium-test"]
    expected_metadata = {"test_type": "medium_granularity", "version": "1.0"}

    if set(first_result["tags"]) == set(expected_tags):
        print("✅ Tags match expected values")
    else:
        print(f"⚠️  Tags don't match: expected {expected_tags}, got {first_result['tags']}")

    if first_result["metadata"] == expected_metadata:
        print("✅ Metadata matches expected values")
    else:
        print(f"⚠️  Metadata doesn't match: expected {expected_metadata}, got {first_result['metadata']}")

    # Cleanup
    print("\n4. Cleaning up test data...")
    store.delete_memory(memory_id)
    print("✅ Test memory deleted")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TAGS & METADATA FIX VERIFICATION TEST SUITE")
    print("=" * 80)
    print("\nTesting the fix for missing tags/metadata in FINE and MEDIUM granularity searches")

    test1_passed = False
    test2_passed = False

    try:
        test1_passed = test_tags_metadata_fine_granularity()
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    try:
        test2_passed = test_tags_metadata_medium_granularity()
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Fine Granularity): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Medium Granularity): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.")
        sys.exit(1)
