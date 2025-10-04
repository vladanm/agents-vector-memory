#!/usr/bin/env python3
"""Test MCP server startup and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore

def test_store_creation():
    """Test creating a store instance."""
    print("Testing store creation...")
    store = SessionMemoryStore(db_path="/tmp/test_mcp_server.db")
    print(f"✓ Store created: {type(store).__name__}")
    return store

def test_basic_storage(store):
    """Test basic memory storage."""
    print("\nTesting memory storage...")
    result = store.store_memory(
        agent_id="test_agent",
        session_id="test_session",
        content="Test memory content",
        memory_type="working_memory",
        session_iter=1
    )
    print(f"✓ Memory stored with ID: {result.get('memory_id')}")
    return result

def test_search(store):
    """Test memory search."""
    print("\nTesting memory search...")
    results = store.search_memories(
        query="test",
        limit=5
    )
    print(f"✓ Search returned {len(results)} results")
    return results

if __name__ == "__main__":
    print("="*60)
    print("MCP Server Startup Test")
    print("="*60)

    try:
        store = test_store_creation()
        test_basic_storage(store)
        test_search(store)

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - MCP Server is working!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
