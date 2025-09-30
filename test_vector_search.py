#!/usr/bin/env python3
"""
Test Vector Search Fix
=====================

Test script to verify that vector search queries work correctly
with the sqlite-vec k constraint fix.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore

def test_vector_search():
    """Test vector search functionality."""
    print("🧪 Testing Vector Search Fix...")
    
    # Use the migrated database (same as MCP server)
    db_path = Path("/Users/vladanm/projects/subagents/simple-agents/memory/memory/memory/agent_session_memory.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        return False
    
    try:
        # Initialize store
        store = SessionMemoryStore(db_path=db_path)
        print(f"✅ Connected to database: {db_path}")
        
        # Test 1: Non-semantic search (should work)
        print("\n🔍 Test 1: Non-semantic search...")
        result1 = store.search_memories(
            limit=5,
            latest_first=True
        )
        
        if result1["success"]:
            print(f"✅ Non-semantic search: Found {result1['total_results']} memories")
            for i, memory in enumerate(result1["results"][:2]):  # Show first 2
                print(f"  - {memory['memory_type']}: {memory['title'] or 'Untitled'}")
        else:
            print(f"❌ Non-semantic search failed: {result1.get('message')}")
            return False
            
        # Test 2: Semantic search (the problematic one)
        print("\n🔍 Test 2: Semantic search...")
        result2 = store.search_memories(
            query="python code",
            limit=5,
            similarity_threshold=0.5
        )
        
        if result2["success"]:
            print(f"✅ Semantic search: Found {result2['total_results']} memories")
            for i, memory in enumerate(result2["results"][:2]):  # Show first 2
                similarity = memory.get('similarity', 0)
                print(f"  - {memory['memory_type']}: {memory['title'] or 'Untitled'} (similarity: {similarity:.3f})")
        else:
            print(f"❌ Semantic search failed: {result2.get('message')}")
            return False
            
        # Test 3: Semantic search with agent scoping
        print("\n🔍 Test 3: Semantic search with agent scoping...")
        result3 = store.search_memories(
            query="database configuration",
            agent_id="migrated",
            limit=3,
            similarity_threshold=0.3
        )
        
        if result3["success"]:
            print(f"✅ Scoped semantic search: Found {result3['total_results']} memories")
            for memory in result3["results"]:
                similarity = memory.get('similarity', 0)
                print(f"  - {memory['memory_type']}: {memory['title'] or 'Untitled'} (similarity: {similarity:.3f})")
        else:
            print(f"❌ Scoped semantic search failed: {result3.get('message')}")
            return False
            
        print("\n🎉 All vector search tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_search()
    sys.exit(0 if success else 1)