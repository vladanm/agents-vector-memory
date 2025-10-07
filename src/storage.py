"""
Core Storage Operations
======================

Basic CRUD operations for session memories.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class StorageOperations:
    """Core storage operations for session memories."""

    def __init__(self, store):
        """Initialize with reference to parent store."""
        self.store = store

    def store_memory(
        self,
        memory_type: str,
        agent_id: str,
        session_id: str,
        content: str,
        session_iter: int = 1,
        task_code: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_chunk: bool | None = None,
        embedding: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Store a memory entry with optional chunking and embedding.

        Delegates to parent store's implementation.
        """
        return self.store._store_memory_impl(
            memory_type, agent_id, session_id, content,
            session_iter, task_code, title, description,
            tags, metadata, auto_chunk, embedding
        )

    def get_memory(self, memory_id: int) -> dict[str, Any]:
        """Retrieve specific memory by ID."""
        return self.store._get_memory_impl(memory_id)

    def delete_memory(self, memory_id: int) -> dict[str, Any]:
        """Delete a memory and all associated data."""
        return self.store._delete_memory_impl(memory_id)

    def reconstruct_document(self, memory_id: int) -> dict[str, Any]:
        """Reconstruct a document from its chunks."""
        return self.store._reconstruct_document_impl(memory_id)

    def write_document_to_file(
        self,
        memory_id: int,
        output_path: str | None = None,
        include_metadata: bool = True,
        format: str = "markdown"
    ) -> dict[str, Any]:
        """Write a reconstructed document to disk."""
        return self.store._write_document_to_file_impl(
            memory_id, output_path, include_metadata, format
        )
