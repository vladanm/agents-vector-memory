#!/usr/bin/env python3
"""Fix missing commas in the patched file"""

from pathlib import Path

def main():
    file_path = Path(__file__).parent / "src" / "session_memory_store.py"
    content = file_path.read_text()

    # Fix 1: Add comma after "error": None on line 449
    content = content.replace(
        '                    "error": None\n                    "message":',
        '                    "error": None,\n                    "message":'
    )

    # Fix 2: Fix filters dict and add commas (line 574-576)
    content = content.replace(
        '                    "task_code": task_code\n                "error": None,',
        '                    "task_code": task_code\n                },\n                "error": None,'
    )

    # Fix 3: Add comma after message in fine granularity (line 644-645)
    content = content.replace(
        '                "message": "Chunk-level search requires embedding infrastructure"\n                "error": None',
        '                "message": "Chunk-level search requires embedding infrastructure",\n                "error": None'
    )

    # Fix 4: Add comma after message in medium granularity (line 655-656)
    content = content.replace(
        '                "message": "Section-level search requires embedding infrastructure"\n                "error": None',
        '                "message": "Section-level search requires embedding infrastructure",\n                "error": None'
    )

    # Fix 5: Fix session stats return (lines 918-920)
    content = content.replace(
        '                "error": None,\n                "message": None\n                "memory_type_breakdown":',
        '                "error": None,\n                "message": None,\n                "memory_type_breakdown":'
    )

    file_path.write_text(content)
    print("✓ Fixed all missing commas")

if __name__ == "__main__":
    main()
