#!/usr/bin/env python3
"""
Test script to validate session_iter filter fix.
Runs the 8 filter tests from Section 9 of test design document.
"""

import sys
import asyncio
from pathlib import Path

# Add parent to path for proper package imports
sys.path.insert(0, str(Path(__file__).parent))

from src.session_memory_store import SessionMemoryStore


async def main():
    print("=" * 80)
    print("Filter Fix Validation Test Suite")
    print("=" * 80)
    print()

    # Initialize store
    store = SessionMemoryStore(db_path="./memory/vector_memory.db")

    print("✓ SessionMemoryStore initialized")
    print()

    # Test 9.1: Setup baseline data
    print("Test 9.1: Baseline Multi-Session Data Setup")
    print("-" * 80)

    test_data = [
        ("agent-alpha", "filter-session-001", "v1", "task-A", "Report A1: Database performance analysis for session 001", "Report A1"),
        ("agent-alpha", "filter-session-001", "v2", "task-A", "Report A2: Updated database performance analysis for session 001", "Report A2"),
        ("agent-beta", "filter-session-001", "v1", "task-B", "Report B1: API performance analysis for session 001", "Report B1"),
        ("agent-alpha", "filter-session-002", "v1", "task-A", "Report C1: Database performance analysis for session 002", "Report C1"),
        ("agent-beta", "filter-session-002", "v1", "task-B", "Report D1: API performance analysis for session 002", "Report D1"),
    ]

    memory_ids = []
    for agent_id, session_id, session_iter, task_code, content, title in test_data:
        result = await store.store_report(
            agent_id=agent_id,
            session_id=session_id,
            session_iter=session_iter,
            content=content,
            title=title,
            task_code=task_code
        )
        memory_ids.append(result["memory_id"])
        print(f"  ✓ Created {title}: memory_id={result['memory_id']}")

    print(f"\n✓ Test 9.1 PASSED: Created {len(memory_ids)} reports")
    print()

    # Test 9.2: session_id filtering
    print("Test 9.2: Filter by session_id")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        session_id="filter-session-001",
        limit=10
    )
    print(f"  Filter: session_id='filter-session-001'")
    print(f"  Expected: 3 results (A1, A2, B1)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 3:
        print(f"  ✓ Test 9.2 PASSED")
    else:
        print(f"  ✗ Test 9.2 FAILED")
    print()

    # Test 9.3: agent_id filtering
    print("Test 9.3: Filter by agent_id")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        agent_id="agent-alpha",
        limit=10
    )
    print(f"  Filter: agent_id='agent-alpha'")
    print(f"  Expected: 3 results (A1, A2, C1)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 3:
        print(f"  ✓ Test 9.3 PASSED")
    else:
        print(f"  ✗ Test 9.3 FAILED")
    print()

    # Test 9.4: session_iter=1 filtering (CRITICAL TEST)
    print("Test 9.4: Filter by session_iter=1 (CRITICAL TEST)")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        session_iter=1,
        limit=10
    )
    print(f"  Filter: session_iter=1 (integer)")
    print(f"  Expected: 4 results (A1, B1, C1, D1 - all v1 iterations)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 4:
        print(f"  ✓ Test 9.4 PASSED - FIX IS WORKING!")
        print(f"  ✓ Integer 1 now matches string 'v1' in database")
    else:
        print(f"  ✗ Test 9.4 FAILED - BUG STILL PRESENT")
        print(f"  ✗ Type mismatch still causing filter failure")
    print()

    # Test 9.5: session_iter=2 filtering
    print("Test 9.5: Filter by session_iter=2")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        session_iter=2,
        limit=10
    )
    print(f"  Filter: session_iter=2 (integer)")
    print(f"  Expected: 1 result (A2 - only v2 iteration)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 1:
        print(f"  ✓ Test 9.5 PASSED")
    else:
        print(f"  ✗ Test 9.5 FAILED")
    print()

    # Test 9.6: task_code filtering
    print("Test 9.6: Filter by task_code")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        task_code="task-A",
        limit=10
    )
    print(f"  Filter: task_code='task-A'")
    print(f"  Expected: 3 results (A1, A2, C1)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 3:
        print(f"  ✓ Test 9.6 PASSED")
    else:
        print(f"  ✗ Test 9.6 FAILED")
    print()

    # Test 9.7: Combined session_id + agent_id
    print("Test 9.7: Combined Filter (session_id + agent_id)")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        session_id="filter-session-001",
        agent_id="agent-alpha",
        limit=10
    )
    print(f"  Filter: session_id='filter-session-001' AND agent_id='agent-alpha'")
    print(f"  Expected: 2 results (A1, A2)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 2:
        print(f"  ✓ Test 9.7 PASSED")
    else:
        print(f"  ✗ Test 9.7 FAILED")
    print()

    # Test 9.8: Combined with session_iter (CRITICAL)
    print("Test 9.8: Combined Filter (session_id + session_iter + task_code)")
    print("-" * 80)

    result = await store.search_reports_specific_chunks(
        query="performance analysis",
        session_id="filter-session-001",
        session_iter=1,
        task_code="task-A",
        limit=10
    )
    print(f"  Filter: session_id='filter-session-001' AND session_iter=1 AND task_code='task-A'")
    print(f"  Expected: 1 result (A1 only)")
    print(f"  Actual: {result['total_results']} results")

    if result['total_results'] == 1:
        print(f"  ✓ Test 9.8 PASSED - Combined filters with session_iter working!")
    else:
        print(f"  ✗ Test 9.8 FAILED - session_iter still breaking combined filters")
    print()

    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)
    print()
    print("Check logs at: ./logs/mcp_server.log for filter pass rates")
    print()


if __name__ == "__main__":
    asyncio.run(main())
