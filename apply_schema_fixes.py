#!/usr/bin/env python3
"""
Apply Schema Validation Fixes
==============================

Directly patches the 4 problematic return statements in session_memory_store.py
to include all required fields for Pydantic validation.
"""

from pathlib import Path

def main():
    file_path = Path(__file__).parent / "src" / "session_memory_store.py"

    print(f"Reading {file_path}...")
    lines = file_path.read_text().split('\n')

    fixed_lines = []
    i = 0
    changes_made = 0

    while i < len(lines):
        line = lines[i]

        # FIX 1: _store_memory_impl successful return (around line 440-450)
        if (i < len(lines) - 10 and
            '"success": True,' in line and
            i > 430 and i < 460 and
            '"memory_id": memory_id,' in lines[i+1]):

            print(f"  Fix 1: Adding 'error' field to store_memory success return at line {i+1}")
            # Copy all lines of this return statement
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and '}' not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            # Add error field before closing brace
            fixed_lines.append('                    "error": None')
            fixed_lines.append(lines[j])  # The closing brace line
            i = j + 1
            changes_made += 1
            continue

        # FIX 2: _search_memories_impl successful return (around line 563-577)
        if (i < len(lines) - 15 and
            '"success": True,' in line and
            i > 555 and i < 580 and
            '"results": results,' in lines[i+1]):

            print(f"  Fix 2: Adding 'error' and 'message' fields to search_memories success return at line {i+1}")
            # Copy all lines of this return statement
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and '}' not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            # Add error and message fields before closing brace
            fixed_lines.append('                "error": None,')
            fixed_lines.append('                "message": None')
            fixed_lines.append(lines[j])  # The closing brace line
            i = j + 1
            changes_made += 1
            continue

        # FIX 3a: _search_with_granularity_impl fine granularity return (around line 639-645)
        if (i < len(lines) - 6 and
            '"success": True,' in line and
            i > 635 and i < 650 and
            '"granularity": "fine"' in lines[i+3]):

            print(f"  Fix 3a: Adding 'error' field to granular search (fine) return at line {i+1}")
            # Copy all lines of this return statement
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and '}' not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            # Add error field before closing brace
            fixed_lines.append('                "error": None')
            fixed_lines.append(lines[j])  # The closing brace line
            i = j + 1
            changes_made += 1
            continue

        # FIX 3b: _search_with_granularity_impl medium granularity return (around line 649-655)
        if (i < len(lines) - 6 and
            '"success": True,' in line and
            i > 645 and i < 660 and
            '"granularity": "medium"' in lines[i+3]):

            print(f"  Fix 3b: Adding 'error' field to granular search (medium) return at line {i+1}")
            # Copy all lines of this return statement
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and '}' not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            # Add error field before closing brace
            fixed_lines.append('                "error": None')
            fixed_lines.append(lines[j])  # The closing brace line
            i = j + 1
            changes_made += 1
            continue

        # FIX 4: _get_session_stats_impl successful return (around line 908-923)
        if (i < len(lines) - 15 and
            '"success": True,' in line and
            i > 900 and i < 930 and
            '"total_memories": stats_row[0],' in lines[i+1]):

            print(f"  Fix 4: Adding 'error' and 'message' fields to session_stats success return at line {i+1}")
            # Copy all lines of this return statement
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and '}' not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            # Add error and message fields before closing brace
            fixed_lines.append('                "error": None,')
            fixed_lines.append('                "message": None')
            fixed_lines.append(lines[j])  # The closing brace line
            i = j + 1
            changes_made += 1
            continue

        # No match, keep line as-is
        fixed_lines.append(line)
        i += 1

    if changes_made == 0:
        print("ERROR: No changes were made. Check line number ranges.")
        return False

    print(f"\nApplied {changes_made} fixes")
    print(f"Writing patched file to {file_path}...")
    file_path.write_text('\n'.join(fixed_lines))

    print("✓ All schema validation fixes applied successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
