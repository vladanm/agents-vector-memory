#!/usr/bin/env python3
"""
Test script for size warning implementation in search_with_granularity.

Tests:
1. Small document (<20k tokens) - should return full content
2. Large document (>20k tokens) - should return preview with warnings
3. Multiple documents with response budget tracking
4. All three search tools (knowledge_base, reports, working_memory)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from session_memory_store import SessionMemoryStore
from config import Config
import tempfile

def create_test_store():
    """Create a test database"""
    db_path = Path(tempfile.gettempdir()) / "test_size_warnings.db"
    if db_path.exists():
        db_path.unlink()

    store = SessionMemoryStore(db_path)
    return store, db_path

def create_small_document():
    """Create a document <20k tokens (~5000 tokens)"""
    return "This is a small test document. " * 300  # ~5000 tokens

def create_large_document():
    """Create a document >20k tokens (~25000 tokens)"""
    return "This is a large test document with lots of content. " * 2000  # ~25000 tokens

def create_medium_document():
    """Create a document ~15k tokens"""
    return "This is a medium-sized test document. " * 1500  # ~15000 tokens

def test_small_document_full_content(store):
    """Test 1: Small document should return full content"""
    print("\n" + "="*60)
    print("TEST 1: Small Document (<20k tokens)")
    print("="*60)

    # Store small document
    small_content = create_small_document()
    result = store.store_memory(
        memory_type="knowledge_base",
        agent_id="test-agent",
        session_id="test-session",
        content=small_content,
        title="Small Document",
        session_iter=1,
        auto_chunk=False
    )

    print(f"✓ Stored memory ID: {result['memory_id']}")

    # Search with coarse granularity
    search_result = store.search_with_granularity(
        query="test document",
        memory_type="knowledge_base",
        granularity="coarse",
        agent_id="test-agent",
        session_id="test-session"
    )

    if not search_result['success']:
        print(f"✗ Search failed: {search_result.get('message')}")
        return False

    if not search_result['results']:
        print("✗ No results returned")
        return False

    doc = search_result['results'][0]

    # Verify expectations
    print(f"\nResults:")
    print(f"  - Estimated tokens: {doc.get('estimated_tokens', 'N/A')}")
    print(f"  - Exceeds limit: {doc.get('exceeds_response_limit', 'N/A')}")
    print(f"  - Size warning: {doc.get('size_warning')}")
    print(f"  - Content length: {len(doc.get('content', ''))}")
    print(f"  - Full content included: {len(doc.get('content', '')) > 500}")

    # Assertions
    assert doc.get('estimated_tokens') is not None, "Missing estimated_tokens"
    assert doc.get('estimated_tokens') < 20000, f"Small doc should be <20k tokens, got {doc['estimated_tokens']}"
    assert doc.get('exceeds_response_limit') == False, "Small doc should not exceed limit"
    assert doc.get('size_warning') is None, "Small doc should not have size warning"
    assert len(doc.get('content', '')) == len(small_content), "Small doc should return full content"

    print("\n✅ TEST 1 PASSED: Small document returned with full content")
    return True

def test_large_document_preview(store):
    """Test 2: Large document should return preview with warnings"""
    print("\n" + "="*60)
    print("TEST 2: Large Document (>20k tokens)")
    print("="*60)

    # Store large document
    large_content = create_large_document()
    result = store.store_memory(
        memory_type="reports",
        agent_id="test-agent",
        session_id="test-session",
        content=large_content,
        title="Large Document",
        session_iter=1,
        auto_chunk=False
    )

    print(f"✓ Stored memory ID: {result['memory_id']}")

    # Search with coarse granularity
    search_result = store.search_with_granularity(
        query="test document",
        memory_type="reports",
        granularity="coarse",
        agent_id="test-agent",
        session_id="test-session"
    )

    if not search_result['success']:
        print(f"✗ Search failed: {search_result.get('message')}")
        return False

    if not search_result['results']:
        print("✗ No results returned")
        return False

    doc = search_result['results'][0]

    # Verify expectations
    print(f"\nResults:")
    print(f"  - Estimated tokens: {doc.get('estimated_tokens', 'N/A')}")
    print(f"  - Exceeds limit: {doc.get('exceeds_response_limit', 'N/A')}")
    print(f"  - Size warning: {doc.get('size_warning')}")
    print(f"  - Content length: {len(doc.get('content', ''))}")
    print(f"  - Content preview: {doc.get('content', '')[:100]}...")

    # Assertions
    assert doc.get('estimated_tokens') is not None, "Missing estimated_tokens"
    assert doc.get('estimated_tokens') > 20000, f"Large doc should be >20k tokens, got {doc['estimated_tokens']}"
    assert doc.get('exceeds_response_limit') == True, "Large doc should exceed limit"
    assert doc.get('size_warning') is not None, "Large doc should have size warning"
    assert doc['size_warning'].get('is_too_large') == True, "Size warning should indicate too large"
    assert doc['size_warning'].get('document_tokens') > 20000, "Size warning should show actual token count"
    assert 'write_document_to_file' in doc['size_warning'].get('recommended_action', ''), "Should recommend write_document_to_file"
    assert len(doc.get('content', '')) < len(large_content), "Large doc should return preview, not full content"
    assert "[Content truncated" in doc.get('content', ''), "Preview should indicate truncation"

    print("\n✅ TEST 2 PASSED: Large document returned with preview and size warning")
    return True

def test_multiple_documents_budget(store):
    """Test 3: Multiple documents should track response budget"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Documents with Budget Tracking")
    print("="*60)

    # Store multiple medium-sized documents
    for i in range(3):
        medium_content = create_medium_document()
        result = store.store_memory(
            memory_type="working_memory",
            agent_id="test-agent",
            session_id="test-session-multi",
            content=medium_content,
            title=f"Medium Document {i+1}",
            session_iter=1,
            auto_chunk=False
        )
        print(f"✓ Stored memory ID: {result['memory_id']}")

    # Search with coarse granularity
    search_result = store.search_with_granularity(
        query="test document",
        memory_type="working_memory",
        granularity="coarse",
        agent_id="test-agent",
        session_id="test-session-multi",
        limit=5
    )

    if not search_result['success']:
        print(f"✗ Search failed: {search_result.get('message')}")
        return False

    print(f"\n✓ Found {len(search_result['results'])} results")

    # Verify response budget tracking
    print(f"\nResponse Budget Info:")
    print(f"  - Response tokens: {search_result.get('response_tokens', 'N/A')}")
    budget_info = search_result.get('response_budget_info', {})
    print(f"  - Used tokens: {budget_info.get('used_tokens', 'N/A')}")
    print(f"  - Limit tokens: {budget_info.get('limit_tokens', 'N/A')}")
    print(f"  - Remaining tokens: {budget_info.get('remaining_tokens', 'N/A')}")

    # Assertions
    assert search_result.get('response_tokens') is not None, "Missing response_tokens"
    assert search_result.get('response_budget_info') is not None, "Missing response_budget_info"
    assert budget_info.get('used_tokens') is not None, "Missing used_tokens in budget_info"
    assert budget_info.get('limit_tokens') == 23000, "Limit should be 23000"
    assert budget_info.get('remaining_tokens') is not None, "Missing remaining_tokens"
    assert budget_info.get('used_tokens') < budget_info.get('limit_tokens'), "Used tokens should be less than limit"

    print("\n✅ TEST 3 PASSED: Multiple documents with budget tracking working")
    return True

def test_all_search_tools(store):
    """Test 4: All three search tools should have size warnings"""
    print("\n" + "="*60)
    print("TEST 4: All Three Search Tools (knowledge_base, reports, working_memory)")
    print("="*60)

    # Store one large document for each memory type
    memory_types = ["knowledge_base", "reports", "working_memory"]
    large_content = create_large_document()

    for mem_type in memory_types:
        result = store.store_memory(
            memory_type=mem_type,
            agent_id="test-agent",
            session_id="test-session-all-tools",
            content=large_content,
            title=f"Large {mem_type} Document",
            session_iter=1,
            auto_chunk=False
        )
        print(f"✓ Stored {mem_type} memory ID: {result['memory_id']}")

    # Test each search tool
    all_passed = True
    for mem_type in memory_types:
        print(f"\n  Testing {mem_type}_full_documents...")

        search_result = store.search_with_granularity(
            query="test document",
            memory_type=mem_type,
            granularity="coarse",
            agent_id="test-agent",
            session_id="test-session-all-tools"
        )

        if not search_result['success'] or not search_result['results']:
            print(f"    ✗ {mem_type} search failed")
            all_passed = False
            continue

        doc = search_result['results'][0]

        # Verify size warnings present
        has_estimated_tokens = doc.get('estimated_tokens') is not None
        has_exceeds_limit = doc.get('exceeds_response_limit') is not None
        has_size_warning = doc.get('size_warning') is not None
        has_budget_info = search_result.get('response_budget_info') is not None

        print(f"    - estimated_tokens: {'✓' if has_estimated_tokens else '✗'}")
        print(f"    - exceeds_response_limit: {'✓' if has_exceeds_limit else '✗'}")
        print(f"    - size_warning: {'✓' if has_size_warning else '✗'}")
        print(f"    - response_budget_info: {'✓' if has_budget_info else '✗'}")

        if not (has_estimated_tokens and has_exceeds_limit and has_size_warning and has_budget_info):
            print(f"    ✗ {mem_type} missing required fields")
            all_passed = False
        else:
            print(f"    ✅ {mem_type} search working correctly")

    if all_passed:
        print("\n✅ TEST 4 PASSED: All three search tools have size warnings")
    else:
        print("\n✗ TEST 4 FAILED: Some tools missing size warnings")

    return all_passed

def test_backward_compatibility(store):
    """Test 5: Backward compatibility - content field always exists"""
    print("\n" + "="*60)
    print("TEST 5: Backward Compatibility")
    print("="*60)

    # Store both small and large documents
    small_content = create_small_document()
    large_content = create_large_document()

    small_result = store.store_memory(
        memory_type="knowledge_base",
        agent_id="test-agent",
        session_id="test-session-compat",
        content=small_content,
        title="Small for Compatibility",
        session_iter=1,
        auto_chunk=False
    )

    large_result = store.store_memory(
        memory_type="knowledge_base",
        agent_id="test-agent",
        session_id="test-session-compat",
        content=large_content,
        title="Large for Compatibility",
        session_iter=1,
        auto_chunk=False
    )

    print(f"✓ Stored small memory ID: {small_result['memory_id']}")
    print(f"✓ Stored large memory ID: {large_result['memory_id']}")

    # Search
    search_result = store.search_with_granularity(
        query="Compatibility",
        memory_type="knowledge_base",
        granularity="coarse",
        agent_id="test-agent",
        session_id="test-session-compat"
    )

    print(f"\n✓ Found {len(search_result['results'])} results")

    all_have_content = True
    for i, doc in enumerate(search_result['results']):
        has_content = 'content' in doc
        content_is_string = isinstance(doc.get('content'), str)
        content_not_none = doc.get('content') is not None

        print(f"\nDocument {i+1}:")
        print(f"  - Has 'content' field: {'✓' if has_content else '✗'}")
        print(f"  - Content is string: {'✓' if content_is_string else '✗'}")
        print(f"  - Content not None: {'✓' if content_not_none else '✗'}")
        print(f"  - Estimated tokens: {doc.get('estimated_tokens', 'N/A')}")

        if not (has_content and content_is_string and content_not_none):
            all_have_content = False

    if all_have_content:
        print("\n✅ TEST 5 PASSED: Backward compatibility maintained - content field always exists as string")
    else:
        print("\n✗ TEST 5 FAILED: Some documents missing content field")

    return all_have_content

def main():
    print("="*60)
    print("SIZE WARNING IMPLEMENTATION TEST SUITE")
    print("="*60)

    # Create test database
    print("\n📦 Creating test database...")
    store, db_path = create_test_store()
    print(f"✓ Database created at: {db_path}")

    # Run all tests
    tests = [
        ("Small Document Full Content", test_small_document_full_content),
        ("Large Document Preview", test_large_document_preview),
        ("Multiple Documents Budget", test_multiple_documents_budget),
        ("All Search Tools", test_all_search_tools),
        ("Backward Compatibility", test_backward_compatibility)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func(store)
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n{passed_count}/{total_count} tests passed")

    # Cleanup
    print(f"\n🧹 Cleaning up test database...")
    if db_path.exists():
        db_path.unlink()
        print(f"✓ Deleted {db_path}")

    # Exit code
    sys.exit(0 if passed_count == total_count else 1)

if __name__ == "__main__":
    main()
