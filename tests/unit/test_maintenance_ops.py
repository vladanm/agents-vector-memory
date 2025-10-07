"""Tests for database maintenance operations (VACUUM, ANALYZE)."""

import tempfile
from pathlib import Path
from src.session_memory_store import SessionMemoryStore


def test_vacuum_database():
    """Test VACUUM operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SessionMemoryStore(db_path=db_path)

        # Store some data first
        store.storage.store_memory(
            memory_type="working_memory",
            agent_id="test",
            session_id="session-1",
            content="Test content for vacuum",
            session_iter=1
        )

        # Run VACUUM
        result = store.maintenance.vacuum_database()

        assert result["success"] is True
        assert result["operation"] == "VACUUM"
        assert "size_before_bytes" in result
        assert "size_after_bytes" in result
        assert "space_reclaimed_bytes" in result


def test_analyze_database():
    """Test ANALYZE operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SessionMemoryStore(db_path=db_path)

        # Store some data first
        store.storage.store_memory(
            memory_type="reports",
            agent_id="test",
            session_id="session-1",
            content="Test report for analyze",
            session_iter=1,
            task_code="test-task"
        )

        # Run ANALYZE
        result = store.maintenance.analyze_database()

        assert result["success"] is True
        assert result["operation"] == "ANALYZE"
        assert "tables_analyzed" in result
        assert result["tables_analyzed"] > 0


def test_optimize_database():
    """Test full optimization (VACUUM + ANALYZE)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SessionMemoryStore(db_path=db_path)

        # Store some data
        for i in range(5):
            store.storage.store_memory(
                memory_type="system_memory",
                agent_id="test",
                session_id=f"session-{i}",
                content=f"System memory {i}",
                session_iter=1
            )

        # Run full optimization
        result = store.maintenance.optimize_database()

        assert result["success"] is True
        assert result["operation"] == "OPTIMIZE (VACUUM + ANALYZE)"
        assert "vacuum_result" in result
        assert "analyze_result" in result
        assert result["vacuum_result"]["success"] is True
        assert result["analyze_result"]["success"] is True
