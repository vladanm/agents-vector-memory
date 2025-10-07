"""Custom exception classes for vector memory MCP server.

This module defines specific exception types for different error scenarios,
enabling better error handling and retry logic throughout the application.
"""

from typing import Any


class VectorMemoryException(Exception):
    """Base exception for all vector memory errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to error dict for MCP response."""
        return {
            "success": False,
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ValidationError(VectorMemoryException):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, provided_value: Any = None):
        details = {"field": field}
        if provided_value is not None:
            details["provided_value"] = str(provided_value)
        super().__init__(f"Validation failed for '{field}': {message}", details)


class MemoryError(VectorMemoryException):
    """Raised when memory storage or retrieval fails."""

    def __init__(self, operation: str, message: str, memory_id: int | None = None):
        details = {"operation": operation}
        if memory_id is not None:
            details["memory_id"] = memory_id
        super().__init__(f"Memory {operation} failed: {message}", details)


class SearchError(VectorMemoryException):
    """Raised when search operations fail."""

    def __init__(self, message: str, query: str | None = None, granularity: str | None = None):
        details = {}
        if query:
            details["query"] = query[:100]  # Truncate long queries
        if granularity:
            details["granularity"] = granularity
        super().__init__(f"Search failed: {message}", details)


class ChunkingError(VectorMemoryException):
    """Raised when document chunking fails."""

    def __init__(self, message: str, document_length: int | None = None, chunk_count: int | None = None):
        details = {}
        if document_length is not None:
            details["document_length"] = document_length
        if chunk_count is not None:
            details["chunk_count"] = chunk_count
        super().__init__(f"Chunking failed: {message}", details)


class DatabaseError(VectorMemoryException):
    """Raised when database operations fail."""

    def __init__(self, operation: str, message: str, sql: str | None = None):
        details = {"operation": operation}
        if sql:
            details["sql"] = sql[:200]  # Truncate long SQL
        super().__init__(f"Database {operation} failed: {message}", details)


class DatabaseLockError(DatabaseError):
    """Raised specifically for database lock errors (for retry logic)."""

    def __init__(self, message: str = "Database is locked"):
        super().__init__("lock", message)
