#!/usr/bin/env python3
"""
Comprehensive Database Fix Verification Tests
Tests all 8 issues identified in the audit report.
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fresh_database_creation():
    """Test 1: Fresh Database Creation"""
    print("=" * 80)
    print("TEST 1: FRESH DATABASE CREATION")
    print("=" * 80)

    test_db = tempfile.mktemp(suffix=".db")
    print(f"\n1. Creating fresh database at: {test_db}")

    try:
        from src.session_memory_store import SessionMemoryStore

        print("2. Initializing SessionMemoryStore...")
        store = SessionMemoryStore(test_db)
        print("   ✅ Database created successfully")

        print("\n3. Verifying tables...")
        conn = sqlite3.connect(test_db)

        expected_tables = ['session_memories', 'memory_chunks', 'vec_chunk_search', 'vec_session_search']
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?, ?)",
            expected_tables
        )
        tables = [row[0] for row in cursor.fetchall()]

        print(f"   Expected: {expected_tables}")
        print(f"   Found:    {tables}")

        if set(tables) == set(expected_tables):
            print("   ✅ All required tables present")
        else:
            missing = set(expected_tables) - set(tables)
            print(f"   ❌ MISSING TABLES: {missing}")
            conn.close()
            os.unlink(test_db)
            return False

        print("\n4. Verifying PRAGMA settings...")
        tests_passed = True

        # Test foreign_keys
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        print(f"   foreign_keys:   {fk} (expected: 1)")
        if fk != 1:
            print("   ❌ CRITICAL: Foreign keys not enabled!")
            tests_passed = False
        else:
            print("   ✅ Foreign keys enabled")

        # Test journal_mode
        jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print(f"   journal_mode:   {jm} (expected: wal)")
        if jm.lower() != 'wal':
            print("   ⚠️  WARNING: WAL mode not enabled")
        else:
            print("   ✅ WAL mode enabled")

        # Test busy_timeout
        bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        print(f"   busy_timeout:   {bt} (expected: 30000)")
        if bt < 5000:
            print("   ❌ CRITICAL: Busy timeout too low!")
            tests_passed = False
        else:
            print("   ✅ Busy timeout configured")

        # Test cache_size
        cs = conn.execute("PRAGMA cache_size").fetchone()[0]
        print(f"   cache_size:     {cs} (expected: -64000)")
        if cs == -64000:
            print("   ✅ Cache size configured (64MB)")
        else:
            print("   ⚠️  WARNING: Cache size not optimized")
            tests_passed = False

        # Test temp_store
        ts = conn.execute("PRAGMA temp_store").fetchone()[0]
        print(f"   temp_store:     {ts} (expected: 2=MEMORY)")
        if ts == 2:
            print("   ✅ Temp store set to MEMORY")
        else:
            print("   ⚠️  WARNING: Temp store not optimized")
            tests_passed = False

        # Verify original_content column exists
        print("\n5. Verifying original_content column...")
        cursor = conn.execute("PRAGMA table_info(session_memories)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'original_content' in columns:
            print("   ✅ original_content column present")
        else:
            print("   ❌ CRITICAL: original_content column missing!")
            tests_passed = False

        conn.close()
        os.unlink(test_db)

        print("\n" + "=" * 80)
        if tests_passed:
            print("✅ TEST 1 PASSED")
        else:
            print("❌ TEST 1 FAILED")
        print("=" * 80 + "\n")
        return tests_passed

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(test_db):
            os.unlink(test_db)
        return False

def test_foreign_key_cascade():
    """Test 2: Foreign Key CASCADE Delete"""
    print("=" * 80)
    print("TEST 2: FOREIGN KEY CASCADE DELETE")
    print("=" * 80)

    test_db = tempfile.mktemp(suffix=".db")
    print(f"\n1. Creating test database at: {test_db}")

    try:
        from src.session_memory_store import SessionMemoryStore

        print("2. Initializing SessionMemoryStore...")
        store = SessionMemoryStore(test_db)

        print("\n3. Storing test memory with auto-chunking...")
        result = store.store_memory(
            memory_type="working_memory",
            agent_id="test-agent",
            session_id="test-session",
            content="""# Test Document

## Section 1
This is section 1 with some content.

## Section 2
This is section 2 with more content.

## Section 3
Final section with concluding remarks.""",
            auto_chunk=True
        )

        if not result["success"]:
            print(f"   ❌ Failed to store memory: {result.get('error')}")
            os.unlink(test_db)
            return False

        memory_id = result["memory_id"]
        chunks_created = result["chunks_created"]
        print(f"   ✅ Memory stored: ID={memory_id}, chunks={chunks_created}")

        print(f"\n4. Verifying chunks exist...")
        conn = sqlite3.connect(test_db)
        conn.execute("PRAGMA foreign_keys = ON")

        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
            (memory_id,)
        ).fetchone()[0]

        print(f"   Chunks before delete: {chunk_count}")

        print(f"\n5. Deleting parent memory (ID={memory_id})...")
        conn.execute("DELETE FROM session_memories WHERE id = ?", (memory_id,))
        conn.commit()

        print("\n6. Verifying CASCADE delete...")
        chunk_count_after = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE parent_id = ?",
            (memory_id,)
        ).fetchone()[0]

        print(f"   Chunks after delete: {chunk_count_after}")

        conn.close()
        os.unlink(test_db)

        print("\n" + "=" * 80)
        if chunk_count_after == 0:
            print("✅ TEST 2 PASSED: CASCADE delete working")
        else:
            print("❌ TEST 2 FAILED: CASCADE delete not working!")
        print("=" * 80 + "\n")

        return chunk_count_after == 0

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(test_db):
            os.unlink(test_db)
        return False

def test_store_retrieve():
    """Test 3: Store and Retrieve Operations"""
    print("=" * 80)
    print("TEST 3: STORE AND RETRIEVE OPERATIONS")
    print("=" * 80)

    test_db = tempfile.mktemp(suffix=".db")
    print(f"\n1. Creating test database at: {test_db}")

    try:
        from src.session_memory_store import SessionMemoryStore

        print("2. Initializing SessionMemoryStore...")
        store = SessionMemoryStore(test_db)

        print("\n3. Testing simple memory storage...")
        result1 = store.store_memory(
            memory_type="working_memory",
            agent_id="test-agent",
            session_id="test-session",
            content="This is a simple test memory.",
            auto_chunk=False
        )

        if not result1["success"]:
            print(f"   ❌ Failed: {result1.get('error')}")
            os.unlink(test_db)
            return False

        memory_id1 = result1["memory_id"]
        print(f"   ✅ Memory stored: ID={memory_id1}")

        print("\n4. Testing memory retrieval...")
        memory = store.get_memory(memory_id1)

        if not memory["success"]:
            print(f"   ❌ Failed: {memory.get('error')}")
            os.unlink(test_db)
            return False

        print(f"   ✅ Retrieved memory ID={memory_id1}")

        print("\n5. Testing memory deletion...")
        delete_result = store.delete_memory(memory_id1)

        if not delete_result["success"]:
            print(f"   ❌ Failed: {delete_result.get('error')}")
            os.unlink(test_db)
            return False

        print(f"   ✅ Memory deleted successfully")

        os.unlink(test_db)

        print("\n" + "=" * 80)
        print("✅ TEST 3 PASSED: All operations successful")
        print("=" * 80 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(test_db):
            os.unlink(test_db)
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VECTOR MEMORY DATABASE FIX VERIFICATION")
    print("Testing all 8 issues identified in audit report")
    print("=" * 80 + "\n")

    results = []

    # Run all tests
    results.append(("Test 1: Fresh DB Creation", test_fresh_database_creation()))
    results.append(("Test 2: CASCADE Delete", test_foreign_key_cascade()))
    results.append(("Test 3: Store/Retrieve", test_store_retrieve()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 80)
    print(f"\nFinal Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Database fixes are working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above.")
        sys.exit(1)
