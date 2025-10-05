#!/usr/bin/env python3
"""
Test Database Initialization
=============================

Verifies that new database initialization creates complete schema.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session_memory_store import SessionMemoryStore


def test_initialization():
    """Test that database initialization creates complete schema."""

    print("🧪 Testing database initialization...\n")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        # Initialize store (should create complete schema)
        print(f"📁 Creating test database: {db_path}")
        store = SessionMemoryStore(db_path=db_path)

        print("✅ Database initialized\n")

        # Verify schema using the verification script
        import subprocess
        result = subprocess.run(
            [sys.executable, 'verify_schema.py', str(db_path)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if result.returncode == 0:
            print("\n✅ TEST PASSED: Database initialization creates complete schema")
            return True
        else:
            print("\n❌ TEST FAILED: Schema verification failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()
            print(f"\n🧹 Cleaned up test database")


if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
