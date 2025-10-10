#!/usr/bin/env python3
"""
Run filter tests from Section 9 of test design document.
Uses direct database access to verify the fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.session_memory_store import SessionMemoryStore


def main():
    print("=" * 80)
    print("FILTER TESTS - Section 9 Test Execution")
    print("=" * 80)
    print()

    # Use the actual database
    db_path = "./memory/vector_memory.db"
    store = SessionMemoryStore(db_path=db_path)

    print(f"Using database: {db_path}")
    print()

    # Test 9.1: Setup baseline data
    print("Test 9.1: Baseline Multi-Session Data Setup")
    print("-" * 80)

    test_reports = [
        ("agent-alpha", "filter-session-001", "v1", "task-A", "Report A1: Database performance analysis for session 001", "Report A1"),
        ("agent-alpha", "filter-session-001", "v2", "task-A", "Report A2: Updated database performance analysis for session 001", "Report A2"),
        ("agent-beta", "filter-session-001", "v1", "task-B", "Report B1: API performance analysis for session 001", "Report B1"),
        ("agent-alpha", "filter-session-002", "v1", "task-A", "Report C1: Database performance analysis for session 002", "Report C1"),
        ("agent-beta", "filter-session-002", "v1", "task-B", "Report D1: API performance analysis for session 002", "Report D1"),
    ]

    memory_ids = []
    for agent_id, session_id, session_iter, task_code, content, title in test_reports:
        result = store.store_memory(
            memory_type="reports",
            agent_id=agent_id,
            session_id=session_id,
            session_iter=session_iter,
            content=content,
            title=title,
            task_code=task_code
        )
        memory_ids.append(result["memory_id"])
        print(f"  ✓ {title}: memory_id={result['memory_id']}, session_iter={session_iter}")

    print(f"\n✓ Test 9.1 PASSED: Created {len(memory_ids)} reports")
    print()

    # Test 9.2: session_id filtering
    print("Test 9.2: Filter by session_id='filter-session-001'")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        session_id="filter-session-001",
        limit=10
    )

    expected = 3
    actual = result["total_results"]
    status = "✓ PASSED" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Expected: {expected} results (A1, A2, B1)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    # Test 9.3: agent_id filtering
    print("Test 9.3: Filter by agent_id='agent-alpha'")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        agent_id="agent-alpha",
        limit=10
    )

    expected = 3
    actual = result["total_results"]
    status = "✓ PASSED" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Expected: {expected} results (A1, A2, C1)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    # Test 9.4: session_iter=1 (CRITICAL TEST)
    print("Test 9.4: Filter by session_iter=1 (CRITICAL TEST - INTEGER)")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        session_iter=1,  # INTEGER
        limit=10
    )

    expected = 4
    actual = result["total_results"]
    status = "✓ PASSED - FIX WORKING!" if actual == expected else f"✗ FAILED - BUG PRESENT (expected {expected}, got {actual})"
    print(f"  Filter: session_iter=1 (integer)")
    print(f"  Expected: {expected} results (A1, B1, C1, D1 - all v1 iterations)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")

    if actual == expected:
        print(f"  ✓ Integer 1 successfully matches string 'v1' in database!")
    else:
        print(f"  ✗ Type mismatch still causing filter failure")
    print()

    # Test 9.5: session_iter=2
    print("Test 9.5: Filter by session_iter=2")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        session_iter=2,  # INTEGER
        limit=10
    )

    expected = 1
    actual = result["total_results"]
    status = "✓ PASSED" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Filter: session_iter=2 (integer)")
    print(f"  Expected: {expected} result (A2 - only v2 iteration)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    # Test 9.6: task_code filtering
    print("Test 9.6: Filter by task_code='task-A'")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        task_code="task-A",
        limit=10
    )

    expected = 3
    actual = result["total_results"]
    status = "✓ PASSED" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Expected: {expected} results (A1, A2, C1)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    # Test 9.7: Combined filters (without session_iter)
    print("Test 9.7: Combined Filter (session_id + agent_id)")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        session_id="filter-session-001",
        agent_id="agent-alpha",
        limit=10
    )

    expected = 2
    actual = result["total_results"]
    status = "✓ PASSED" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Filter: session_id='filter-session-001' AND agent_id='agent-alpha'")
    print(f"  Expected: {expected} results (A1, A2)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    # Test 9.8: Combined filters WITH session_iter (CRITICAL)
    print("Test 9.8: Combined Filter WITH session_iter (CRITICAL)")
    print("-" * 80)

    result = store._search_with_granularity_impl(
        memory_type="reports",
        query="performance analysis",
        granularity="fine",
        session_id="filter-session-001",
        session_iter=1,  # INTEGER
        task_code="task-A",
        limit=10
    )

    expected = 1
    actual = result["total_results"]
    status = "✓ PASSED - Combined filters work!" if actual == expected else f"✗ FAILED (expected {expected}, got {actual})"
    print(f"  Filter: session_id='filter-session-001' AND session_iter=1 AND task_code='task-A'")
    print(f"  Expected: {expected} result (A1 only)")
    print(f"  Actual:   {actual} results")
    print(f"  {status}")
    print()

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Check logs at: ./logs/mcp_server.log")
    print("  Look for: 'Batch passed filters: N/M' entries")
    print("  Expected: N > 0 for session_iter tests (not 0/254)")
    print()


if __name__ == "__main__":
    main()
