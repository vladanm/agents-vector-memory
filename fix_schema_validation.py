#!/usr/bin/env python3
"""
Fix Schema Validation Issues
============================

This script fixes all return statements in session_memory_store.py to include
all required fields for Pydantic validation.

Issues Fixed:
1. StoreMemoryResult - Add 'error': None to successful returns
2. SearchMemoriesResult - Add 'error': None and 'message': None to successful returns
3. GranularSearchResult - Add 'error': None to successful returns when missing
4. SessionStatsResult - Add 'error': None and 'message': None to successful returns
"""

import re
from pathlib import Path

def fix_store_memory_success_return(content: str) -> str:
    """Fix _store_memory_impl successful return to include error field."""
    # Find the successful return statement in _store_memory_impl
    pattern = r'(return\s*{[\s\n]*"success":\s*True,[\s\n]*"memory_id":\s*memory_id,[\s\n]*"memory_type":\s*memory_type,[\s\n]*"agent_id":\s*agent_id,[\s\n]*"session_id":\s*session_id,[\s\n]*"content_hash":\s*content_hash,[\s\n]*"chunks_created":\s*chunks_created,[\s\n]*"created_at":\s*now,[\s\n]*"message":\s*f"Memory stored successfully with ID: {memory_id}"[\s\n]*})'

    replacement = r'''return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "content_hash": content_hash,
                    "chunks_created": chunks_created,
                    "created_at": now,
                    "message": f"Memory stored successfully with ID: {memory_id}",
                    "error": None
                }'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return content

def fix_search_memories_success_return(content: str) -> str:
    """Fix _search_memories_impl successful return to include error and message fields."""
    # Find the successful return statement in _search_memories_impl
    pattern = r'(return\s*{[\s\n]*"success":\s*True,[\s\n]*"results":\s*results,[\s\n]*"total_results":\s*len\(results\),[\s\n]*"query":\s*query,[\s\n]*"filters":\s*{[^}]+},[\s\n]*"limit":\s*limit,[\s\n]*"latest_first":\s*latest_first[\s\n]*})'

    replacement = r'''return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "filters": {
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_iter": session_iter,
                    "task_code": task_code
                },
                "limit": limit,
                "latest_first": latest_first,
                "error": None,
                "message": None
            }'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    return content

def fix_granular_search_returns(content: str) -> str:
    """Fix _search_with_granularity_impl returns to include error field."""
    # Fix fine granularity return
    pattern_fine = r'(return\s*{[\s\n]*"success":\s*True,[\s\n]*"results":\s*\[\],[\s\n]*"total_results":\s*0,[\s\n]*"granularity":\s*"fine",[\s\n]*"message":\s*"Chunk-level search requires embedding infrastructure"[\s\n]*})'

    replacement_fine = r'''return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "fine",
                "message": "Chunk-level search requires embedding infrastructure",
                "error": None
            }'''

    content = re.sub(pattern_fine, replacement_fine, content, flags=re.MULTILINE)

    # Fix medium granularity return
    pattern_medium = r'(return\s*{[\s\n]*"success":\s*True,[\s\n]*"results":\s*\[\],[\s\n]*"total_results":\s*0,[\s\n]*"granularity":\s*"medium",[\s\n]*"message":\s*"Section-level search requires embedding infrastructure"[\s\n]*})'

    replacement_medium = r'''return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "medium",
                "message": "Section-level search requires embedding infrastructure",
                "error": None
            }'''

    content = re.sub(pattern_medium, replacement_medium, content, flags=re.MULTILINE)

    return content

def fix_session_stats_success_return(content: str) -> str:
    """Fix _get_session_stats_impl successful return to include error and message fields."""
    # Find the successful return statement in _get_session_stats_impl
    pattern = r'(return\s*{[\s\n]*"success":\s*True,[\s\n]*"total_memories":\s*stats_row\[0\],[\s\n]*"memory_types":\s*stats_row\[1\],[\s\n]*"unique_agents":\s*stats_row\[2\],[\s\n]*"unique_sessions":\s*stats_row\[3\],[\s\n]*"unique_tasks":\s*stats_row\[4\],[\s\n]*"max_session_iter":\s*stats_row\[5\]\s*or\s*0,[\s\n]*"avg_content_length":\s*round\(stats_row\[6\]\s*or\s*0,\s*2\),[\s\n]*"total_access_count":\s*stats_row\[7\]\s*or\s*0,[\s\n]*"memory_type_breakdown":\s*{[^}]+},[\s\n]*"filters":\s*{[^}]+}[\s\n]*})'

    replacement = r'''return {
                "success": True,
                "total_memories": stats_row[0],
                "memory_types": stats_row[1],
                "unique_agents": stats_row[2],
                "unique_sessions": stats_row[3],
                "unique_tasks": stats_row[4],
                "max_session_iter": stats_row[5] or 0,
                "avg_content_length": round(stats_row[6] or 0, 2),
                "total_access_count": stats_row[7] or 0,
                "memory_type_breakdown": {row[0]: row[1] for row in type_rows},
                "filters": {
                    "agent_id": agent_id,
                    "session_id": session_id
                },
                "error": None,
                "message": None
            }'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    return content

def main():
    """Apply all fixes to session_memory_store.py"""
    file_path = Path(__file__).parent / "src" / "session_memory_store.py"

    print(f"Reading {file_path}...")
    content = file_path.read_text()

    print("Applying fixes...")
    print("  1. Fixing store_memory successful return...")
    content = fix_store_memory_success_return(content)

    print("  2. Fixing search_memories successful return...")
    content = fix_search_memories_success_return(content)

    print("  3. Fixing granular search returns...")
    content = fix_granular_search_returns(content)

    print("  4. Fixing session_stats successful return...")
    content = fix_session_stats_success_return(content)

    print(f"Writing fixed content to {file_path}...")
    file_path.write_text(content)

    print("✓ All fixes applied successfully!")
    print("\nFixed return statements:")
    print("  - StoreMemoryResult: Added 'error': None to successful returns")
    print("  - SearchMemoriesResult: Added 'error': None and 'message': None to successful returns")
    print("  - GranularSearchResult: Added 'error': None to successful returns")
    print("  - SessionStatsResult: Added 'error': None and 'message': None to successful returns")

if __name__ == "__main__":
    main()
