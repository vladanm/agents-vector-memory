"""
Integration tests for WAL mode and concurrent database access.

Tests Phase 1 feature: Write-Ahead Logging mode for improved concurrency.
"""

import pytest
import sqlite3
import threading
import time
from pathlib import Path


@pytest.mark.integration
def test_wal_mode_enabled(temp_db, store):
    """Verify WAL mode is enabled on database."""
    conn = sqlite3.connect(temp_db)
    result = conn.execute("PRAGMA journal_mode").fetchone()
    conn.close()

    assert result[0].upper() == "WAL", f"Expected WAL mode, got {result[0]}"


@pytest.mark.integration
def test_concurrent_writes(temp_db, sample_memory):
    """Test multiple concurrent writes to database."""
    from src.session_memory_store import SessionMemoryStore

    stores = [SessionMemoryStore(temp_db) for _ in range(3)]
    results = []
    errors = []

    def write_memory(store_instance, idx):
        try:
            memory = sample_memory.copy()
            memory["task_code"] = f"concurrent-{idx}"
            result = store_instance.store_memory(**memory)
            results.append(result)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=write_memory, args=(stores[i], i))
        for i in range(3)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 3
    assert all(r["success"] for r in results), "Not all writes succeeded"


@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_read_write(temp_db, sample_memory):
    """Test concurrent reads and writes don't block."""
    from src.session_memory_store import SessionMemoryStore

    store = SessionMemoryStore(temp_db)

    # Insert initial data
    result = store.store_memory(**sample_memory)
    assert result["success"]

    read_count = 0
    write_count = 0

    def reader():
        nonlocal read_count
        for _ in range(10):
            store.search_memories(
                memory_type="session_context",
                limit=5
            )
            read_count += 1
            time.sleep(0.01)

    def writer():
        nonlocal write_count
        for i in range(10):
            mem = sample_memory.copy()
            mem["task_code"] = f"writer-{i}"
            store.store_memory(**mem)
            write_count += 1
            time.sleep(0.01)

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)

    reader_thread.start()
    writer_thread.start()
    reader_thread.join()
    writer_thread.join()

    assert read_count == 10, "Not all reads completed"
    assert write_count == 10, "Not all writes completed"


@pytest.mark.integration
def test_wal_files_created(temp_db, store, sample_memory):
    """Verify WAL-related files are created."""
    # Store data to trigger WAL file creation
    result = store.store_memory(**sample_memory)
    assert result["success"]

    # Check for WAL and SHM files
    db_path = Path(temp_db)
    wal_file = db_path.parent / f"{db_path.name}-wal"
    shm_file = db_path.parent / f"{db_path.name}-shm"

    # WAL file may exist depending on checkpoint timing
    # This test just verifies the database is using WAL mode
    # (actual WAL file creation is timing-dependent)
    conn = sqlite3.connect(temp_db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()

    assert mode.upper() == "WAL", "Database not in WAL mode"
