#!/usr/bin/env python3
"""Test the vector search fix"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
from pathlib import Path

# Add parent to path like main.py does
sys.path.insert(0, str(Path(__file__).parent))

# Import from src package
from src.session_memory_store import SessionMemoryStore

db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
store = SessionMemoryStore(db_path)

print("Testing vector search with fix...")

result = store.search_with_granularity(
    memory_type="working_memory",
    granularity="fine",
    query="test memory",
    agent_id="main-orchestrator",
    session_id="chunking-verification-20251008",
    limit=5,
    similarity_threshold=0.3
)

print(f"\nSuccess: {result['success']}")
print(f"Total Results: {result['total_results']}")

if result['results']:
    print(f"\n✓✓✓ FIX WORKS! Found {len(result['results'])} results ✓✓✓\n")
    for i, r in enumerate(result['results'], 1):
        print(f"{i}. Chunk {r['chunk_id']} (memory {r['memory_id']})")
        print(f"   Similarity: {r['similarity']:.4f}")
        print(f"   Content: {r['chunk_content'][:80]}...")
        print()
else:
    print(f"\n✗ Still broken")
    print(f"Error: {result.get('error')}")
    print(f"Message: {result.get('message')}")
