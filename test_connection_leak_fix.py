#!/usr/bin/env python3
"""
Test to verify the connection leak fix.

This test verifies that:
1. Connections are properly closed even when operations fail
2. Database doesn't get locked after failed operations
3. Multiple operations can succeed in sequence
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore


def test_connection_cleanup():
    """Test that connections are closed even on errors"""

    # Use a test database
    test_db = Path(__file__).parent / "memory" / "memory" / "agent_session_memory.db"
    store = SessionMemoryStore(db_path=str(test_db))

    print("Testing connection cleanup after errors...")

    # Test 1: Try to store invalid memory type (should fail but not leak)
    result1 = store.store_memory(
        memory_type="invalid_type",  # Invalid type
        agent_id="test-agent",
        session_id="test-session",
        content="Test content",
        session_iter=1
    )
    assert not result1["success"], "Expected failure for invalid memory type"
    print("✓ Test 1: Invalid memory type handled correctly")

    # Test 2: Store valid memory (should succeed if connection was closed properly)
    result2 = store.store_memory(
        memory_type="input_prompt",
        agent_id="test-agent",
        session_id="test-session-cleanup",
        content="Test content after error",
        session_iter=1,
        title="Connection Cleanup Test"
    )
    assert result2["success"], f"Expected success but got: {result2}"
    print(f"✓ Test 2: Successfully stored memory after error (ID: {result2['memory_id']})")

    # Test 3: Store another memory immediately (tests no lock)
    result3 = store.store_memory(
        memory_type="working_memory",
        agent_id="test-agent",
        session_id="test-session-cleanup",
        content="Another test content",
        session_iter=1,
        title="Second Write Test"
    )
    assert result3["success"], f"Expected success but got: {result3}"
    print(f"✓ Test 3: Immediately stored second memory (ID: {result3['memory_id']})")

    # Test 4: Delete a memory (tests delete cleanup)
    result4 = store.delete_memory(result2["memory_id"])
    assert result4["success"], f"Expected delete success but got: {result4}"
    print(f"✓ Test 4: Successfully deleted memory {result2['memory_id']}")

    # Test 5: Try to delete non-existent memory (should fail but not leak)
    result5 = store.delete_memory(999999)
    assert not result5["success"], "Expected failure for non-existent memory"
    print("✓ Test 5: Non-existent delete handled correctly")

    # Test 6: Store memory after delete operations (tests no lock)
    result6 = store.store_memory(
        memory_type="system_memory",
        agent_id="test-agent",
        session_id="test-session-cleanup",
        content="Test content after deletes",
        session_iter=1,
        title="Post-Delete Test"
    )
    assert result6["success"], f"Expected success but got: {result6}"
    print(f"✓ Test 6: Successfully stored memory after deletes (ID: {result6['memory_id']})")

    # Cleanup test data
    store.delete_memory(result3["memory_id"])
    store.delete_memory(result6["memory_id"])

    print("\n✅ All tests passed! Connection leak is fixed.")
    print("Database operations work correctly even after errors.")
    return True


if __name__ == "__main__":
    try:
        success = test_connection_cleanup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
