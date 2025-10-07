"""Targeted tests for exceptions.py to boost coverage from 35.56% to >70%.

Coverage target: Instantiate all exception classes and test their methods.
Target lines: All exception __init__ methods and to_dict() methods.
"""

import pytest
from src.exceptions import (
    VectorMemoryException,
    ValidationError,
    MemoryError,
    SearchError,
    ChunkingError,
    DatabaseError,
    DatabaseLockError
)


class TestVectorMemoryException:
    """Test base exception class."""

    def test_init_with_message_only(self):
        """Test exception with just a message."""
        exc = VectorMemoryException("test error")
        assert exc.message == "test error"
        assert exc.details == {}
        assert str(exc) == "test error"

    def test_init_with_message_and_details(self):
        """Test exception with message and details."""
        exc = VectorMemoryException("test error", {"key": "value"})
        assert exc.message == "test error"
        assert exc.details == {"key": "value"}

    def test_to_dict(self):
        """Test to_dict() method."""
        exc = VectorMemoryException("test error", {"field": "test"})
        result = exc.to_dict()
        assert result == {
            "success": False,
            "error": "VectorMemoryException",
            "message": "test error",
            "details": {"field": "test"}
        }


class TestValidationError:
    """Test ValidationError exception."""

    def test_init_with_field_and_message(self):
        """Test ValidationError with field and message only."""
        exc = ValidationError("agent_id", "cannot be empty")
        assert "agent_id" in exc.message
        assert "cannot be empty" in exc.message
        assert exc.details["field"] == "agent_id"
        assert "provided_value" not in exc.details

    def test_init_with_provided_value(self):
        """Test ValidationError with provided_value."""
        exc = ValidationError("memory_type", "invalid type", "INVALID_TYPE")
        assert "memory_type" in exc.message
        assert "invalid type" in exc.message
        assert exc.details["field"] == "memory_type"
        assert exc.details["provided_value"] == "INVALID_TYPE"

    def test_to_dict(self):
        """Test ValidationError.to_dict()."""
        exc = ValidationError("session_id", "must be non-empty", "")
        result = exc.to_dict()
        assert result["error"] == "ValidationError"
        assert "session_id" in result["message"]
        assert result["details"]["field"] == "session_id"


class TestMemoryError:
    """Test MemoryError exception."""

    def test_init_with_operation_and_message(self):
        """Test MemoryError with operation and message only."""
        exc = MemoryError("store", "database connection failed")
        assert "store" in exc.message
        assert "database connection failed" in exc.message
        assert exc.details["operation"] == "store"
        assert "memory_id" not in exc.details

    def test_init_with_memory_id(self):
        """Test MemoryError with memory_id."""
        exc = MemoryError("delete", "memory not found", memory_id=123)
        assert "delete" in exc.message
        assert "memory not found" in exc.message
        assert exc.details["operation"] == "delete"
        assert exc.details["memory_id"] == 123

    def test_to_dict(self):
        """Test MemoryError.to_dict()."""
        exc = MemoryError("retrieve", "corrupted data", memory_id=456)
        result = exc.to_dict()
        assert result["error"] == "MemoryError"
        assert "retrieve" in result["message"]
        assert result["details"]["memory_id"] == 456


class TestSearchError:
    """Test SearchError exception."""

    def test_init_with_message_only(self):
        """Test SearchError with message only."""
        exc = SearchError("no results found")
        assert "no results found" in exc.message
        assert exc.details == {}

    def test_init_with_query(self):
        """Test SearchError with query."""
        exc = SearchError("invalid query", query="test query")
        assert "invalid query" in exc.message
        assert exc.details["query"] == "test query"

    def test_init_with_long_query_truncation(self):
        """Test SearchError truncates long queries to 100 chars."""
        long_query = "x" * 150
        exc = SearchError("query too long", query=long_query)
        assert len(exc.details["query"]) == 100
        assert exc.details["query"] == "x" * 100

    def test_init_with_granularity(self):
        """Test SearchError with granularity."""
        exc = SearchError("invalid granularity", granularity="invalid_level")
        assert "invalid granularity" in exc.message
        assert exc.details["granularity"] == "invalid_level"

    def test_init_with_query_and_granularity(self):
        """Test SearchError with both query and granularity."""
        exc = SearchError(
            "search failed",
            query="test",
            granularity="specific_chunks"
        )
        assert exc.details["query"] == "test"
        assert exc.details["granularity"] == "specific_chunks"

    def test_to_dict(self):
        """Test SearchError.to_dict()."""
        exc = SearchError("embedding failed", query="test", granularity="full_documents")
        result = exc.to_dict()
        assert result["error"] == "SearchError"
        assert result["details"]["query"] == "test"
        assert result["details"]["granularity"] == "full_documents"


class TestChunkingError:
    """Test ChunkingError exception."""

    def test_init_with_message_only(self):
        """Test ChunkingError with message only."""
        exc = ChunkingError("chunking failed")
        assert "chunking failed" in exc.message
        assert exc.details == {}

    def test_init_with_document_length(self):
        """Test ChunkingError with document_length."""
        exc = ChunkingError("document too large", document_length=10000)
        assert "document too large" in exc.message
        assert exc.details["document_length"] == 10000

    def test_init_with_chunk_count(self):
        """Test ChunkingError with chunk_count."""
        exc = ChunkingError("too many chunks", chunk_count=500)
        assert "too many chunks" in exc.message
        assert exc.details["chunk_count"] == 500

    def test_init_with_both_metrics(self):
        """Test ChunkingError with both document_length and chunk_count."""
        exc = ChunkingError(
            "chunking overflow",
            document_length=50000,
            chunk_count=1000
        )
        assert "chunking overflow" in exc.message
        assert exc.details["document_length"] == 50000
        assert exc.details["chunk_count"] == 1000

    def test_to_dict(self):
        """Test ChunkingError.to_dict()."""
        exc = ChunkingError("invalid content", document_length=0)
        result = exc.to_dict()
        assert result["error"] == "ChunkingError"
        assert result["details"]["document_length"] == 0


class TestDatabaseError:
    """Test DatabaseError exception."""

    def test_init_with_operation_and_message(self):
        """Test DatabaseError with operation and message only."""
        exc = DatabaseError("query", "syntax error")
        assert "query" in exc.message
        assert "syntax error" in exc.message
        assert exc.details["operation"] == "query"
        assert "sql" not in exc.details

    def test_init_with_sql(self):
        """Test DatabaseError with SQL statement."""
        exc = DatabaseError("execute", "constraint violation", sql="INSERT INTO table")
        assert "execute" in exc.message
        assert "constraint violation" in exc.message
        assert exc.details["operation"] == "execute"
        assert exc.details["sql"] == "INSERT INTO table"

    def test_init_with_long_sql_truncation(self):
        """Test DatabaseError truncates SQL to 200 chars."""
        long_sql = "SELECT * FROM table WHERE " + "x=1 AND " * 50
        exc = DatabaseError("query", "timeout", sql=long_sql)
        assert len(exc.details["sql"]) == 200
        assert exc.details["sql"] == long_sql[:200]

    def test_to_dict(self):
        """Test DatabaseError.to_dict()."""
        exc = DatabaseError("migration", "version mismatch", sql="ALTER TABLE")
        result = exc.to_dict()
        assert result["error"] == "DatabaseError"
        assert result["details"]["operation"] == "migration"
        assert result["details"]["sql"] == "ALTER TABLE"


class TestDatabaseLockError:
    """Test DatabaseLockError exception (subclass of DatabaseError)."""

    def test_init_default_message(self):
        """Test DatabaseLockError with default message."""
        exc = DatabaseLockError()
        assert "Database is locked" in exc.message
        assert exc.details["operation"] == "lock"

    def test_init_custom_message(self):
        """Test DatabaseLockError with custom message."""
        exc = DatabaseLockError("Lock timeout after 5 seconds")
        assert "Lock timeout after 5 seconds" in exc.message
        assert exc.details["operation"] == "lock"

    def test_to_dict(self):
        """Test DatabaseLockError.to_dict()."""
        exc = DatabaseLockError("Database locked by another process")
        result = exc.to_dict()
        assert result["error"] == "DatabaseLockError"
        assert result["details"]["operation"] == "lock"

    def test_inheritance(self):
        """Test DatabaseLockError is subclass of DatabaseError."""
        exc = DatabaseLockError()
        assert isinstance(exc, DatabaseError)
        assert isinstance(exc, VectorMemoryException)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_validation_error(self):
        """Test raising ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("test_field", "test message")
        assert exc_info.value.details["field"] == "test_field"

    def test_raise_memory_error(self):
        """Test raising MemoryError."""
        with pytest.raises(MemoryError) as exc_info:
            raise MemoryError("store", "test failure", memory_id=999)
        assert exc_info.value.details["memory_id"] == 999

    def test_raise_search_error(self):
        """Test raising SearchError."""
        with pytest.raises(SearchError) as exc_info:
            raise SearchError("test search error", query="test")
        assert exc_info.value.details["query"] == "test"

    def test_raise_chunking_error(self):
        """Test raising ChunkingError."""
        with pytest.raises(ChunkingError) as exc_info:
            raise ChunkingError("test chunking error", chunk_count=10)
        assert exc_info.value.details["chunk_count"] == 10

    def test_raise_database_error(self):
        """Test raising DatabaseError."""
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError("test_op", "test error", sql="SELECT 1")
        assert exc_info.value.details["sql"] == "SELECT 1"

    def test_raise_database_lock_error(self):
        """Test raising DatabaseLockError."""
        with pytest.raises(DatabaseLockError) as exc_info:
            raise DatabaseLockError()
        assert exc_info.value.details["operation"] == "lock"

    def test_catch_as_base_exception(self):
        """Test catching specific exceptions as VectorMemoryException."""
        with pytest.raises(VectorMemoryException):
            raise ValidationError("field", "message")
