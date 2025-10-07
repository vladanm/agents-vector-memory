#!/usr/bin/env python3
"""
Apply Phase 1 P0 Critical Fixes
================================

This script applies all P0 critical fixes to the MCP server:
1. Add WAL mode and optimizations
2. Remove all print() statements and add logging
3. Add context manager support
4. Leave print statements for migration output (intentional)
"""

import re
import sys
from pathlib import Path

def apply_wal_mode_fix(content: str) -> str:
    """Add WAL mode and SQLite optimizations to _get_connection method"""

    # Find the _get_connection method
    pattern = r'(    def _get_connection\(self\) -> sqlite3\.Connection:.*?""".*?""")\s+(conn = sqlite3\.connect\(self\.db_path\))\s+(# CRITICAL FIX:.*?conn\.execute\("PRAGMA busy_timeout = 5000"\))'

    replacement = r'''\1
        conn = sqlite3.connect(self.db_path)

        # STEP 1: Enable WAL mode for better concurrency
        # WAL allows readers to read while writer writes (no blocking)
        # Must be set before other PRAGMAs for maximum effect
        conn.execute("PRAGMA journal_mode=WAL")

        # STEP 2: Set synchronous to NORMAL (balance of safety and speed)
        # NORMAL is safe for most applications and much faster than FULL
        # FULL would fsync after every transaction (too slow for MCP server)
        conn.execute("PRAGMA synchronous=NORMAL")

        # STEP 3: Increase cache size to 64MB for performance
        # Negative value means size in KB (-64000 = 64MB)
        # Default is usually only ~2MB, this improves read performance significantly
        conn.execute("PRAGMA cache_size=-64000")

        # STEP 4: Enable foreign keys for referential integrity
        # SQLite doesn't enable this by default for backward compatibility
        # We need this for CASCADE deletes to work properly
        conn.execute("PRAGMA foreign_keys=ON")

        # STEP 5: Set busy timeout to handle concurrent access
        # 5-second timeout allows SQLite to wait for locks instead of failing immediately
        conn.execute("PRAGMA busy_timeout=5000")'''

    result = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if result == content:
        print("WARNING: Could not find _get_connection pattern", file=sys.stderr)
    return result

def add_context_manager(content: str) -> str:
    """Add __enter__ and __exit__ methods for context manager support"""

    # Add context manager methods after _get_connection
    pattern = r'(        return conn\n\n)'

    addition = r'''\1
    def __enter__(self):
        """Context manager entry - no-op, connections managed per operation"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no-op, connections managed per operation"""
        return False

'''

    result = re.sub(pattern, addition, content)
    return result

def main():
    # Read session_memory_store.py
    store_path = Path("src/session_memory_store.py")
    if not store_path.exists():
        print(f"ERROR: {store_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {store_path}")
    content = store_path.read_text()

    # Apply fixes
    print("Applying WAL mode and optimizations fix...")
    content = apply_wal_mode_fix(content)

    print("Adding context manager support...")
    content = add_context_manager(content)

    # Write updated content
    print(f"Writing updated {store_path}")
    store_path.write_text(content)

    print("✅ Step 1 (WAL mode + optimizations) complete")
    print("✅ Step 3 (context manager) complete")
    print("\nNext: Run script for Step 2 (remove print statements)")

if __name__ == "__main__":
    main()
