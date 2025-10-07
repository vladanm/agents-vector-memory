#!/usr/bin/env python3
"""
Apply Clean Schema Validation Fixes
====================================

Fixes all return statements in session_memory_store.py by directly replacing
the entire return statement blocks with corrected versions.
"""

from pathlib import Path

def main():
    file_path = Path(__file__).parent / "src" / "session_memory_store.py"
    content = file_path.read_text()

    print("Applying schema validation fixes...")

    # FIX 1: _store_memory_impl successful return - add 'error': None
    print("  Fix 1: StoreMemoryResult - adding 'error': None")
    old_return_1 = '''                return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "content_hash": content_hash,
                    "chunks_created": chunks_created,
                    "created_at": now,
                    "message": f"Memory stored successfully with ID: {memory_id}"
                }'''

    new_return_1 = '''                return {
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

    content = content.replace(old_return_1, new_return_1)

    # FIX 2: _search_memories_impl successful return - add 'error': None, 'message': None
    print("  Fix 2: SearchMemoriesResult - adding 'error' and 'message' fields")
    old_return_2 = '''            return {
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
                "latest_first": latest_first
            }'''

    new_return_2 = '''            return {
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

    content = content.replace(old_return_2, new_return_2)

    # FIX 3a: _search_with_granularity_impl fine granularity - add 'error': None
    print("  Fix 3a: GranularSearchResult (fine) - adding 'error': None")
    old_return_3a = '''            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "fine",
                "message": "Chunk-level search requires embedding infrastructure"
            }'''

    new_return_3a = '''            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "fine",
                "message": "Chunk-level search requires embedding infrastructure",
                "error": None
            }'''

    content = content.replace(old_return_3a, new_return_3a)

    # FIX 3b: _search_with_granularity_impl medium granularity - add 'error': None
    print("  Fix 3b: GranularSearchResult (medium) - adding 'error': None")
    old_return_3b = '''            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "medium",
                "message": "Section-level search requires embedding infrastructure"
            }'''

    new_return_3b = '''            return {
                "success": True,
                "results": [],
                "total_results": 0,
                "granularity": "medium",
                "message": "Section-level search requires embedding infrastructure",
                "error": None
            }'''

    content = content.replace(old_return_3b, new_return_3b)

    # FIX 4: _get_session_stats_impl successful return - add 'error': None, 'message': None
    print("  Fix 4: SessionStatsResult - adding 'error' and 'message' fields")
    old_return_4 = '''            return {
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
                }
            }'''

    new_return_4 = '''            return {
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

    content = content.replace(old_return_4, new_return_4)

    # Write the fixed content
    file_path.write_text(content)

    print("\n✓ All schema validation fixes applied successfully!")
    print("\nSummary of changes:")
    print("  1. StoreMemoryResult: Added 'error': None field")
    print("  2. SearchMemoriesResult: Added 'error': None and 'message': None fields")
    print("  3. GranularSearchResult (fine): Added 'error': None field")
    print("  4. GranularSearchResult (medium): Added 'error': None field")
    print("  5. SessionStatsResult: Added 'error': None and 'message': None fields")

if __name__ == "__main__":
    main()
