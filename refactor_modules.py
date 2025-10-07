#!/usr/bin/env python3
"""
Refactor session_memory_store.py into logical modules.

New structure:
- src/storage.py: Core CRUD operations
- src/search.py: Vector search operations
- src/maintenance.py: Cleanup, stats, health checks
- src/chunking_storage.py: Chunk-specific operations
- session_memory_store.py: Thin facade/orchestrator
"""

from pathlib import Path
import re

def create_storage_module():
    """Create src/storage.py with CRUD operations."""
    content = '''"""
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
'''

    Path("src/storage.py").write_text(content)
    print("✓ Created src/storage.py")


def create_search_module():
    """Create src/search.py with search operations."""
    content = '''"""
Search Operations
=================

Vector search and semantic search operations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SearchOperations:
    """Search operations for session memories."""

    def __init__(self, store):
        """Initialize with reference to parent store."""
        self.store = store

    def search_memories(
        self,
        memory_type: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        session_iter: int | None = None,
        task_code: str | None = None,
        query: str | None = None,
        limit: int = 3,
        latest_first: bool = True
    ) -> dict[str, Any]:
        """Search memories with optional filters."""
        return self.store._search_memories_impl(
            memory_type, agent_id, session_id, session_iter,
            task_code, query, limit, latest_first
        )

    def search_with_granularity(
        self,
        memory_type: str,
        granularity: str,
        query: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        session_iter: int | None = None,
        task_code: str | None = None,
        limit: int = 3,
        similarity_threshold: float = 0.7,
        auto_merge_threshold: float = 0.6
    ) -> dict[str, Any]:
        """Search with specific granularity level."""
        return self.store._search_with_granularity_impl(
            memory_type, granularity, query, agent_id, session_id,
            session_iter, task_code, limit, similarity_threshold,
            auto_merge_threshold
        )

    def load_session_context_for_task(
        self,
        agent_id: str,
        session_id: str,
        current_task_code: str
    ) -> dict[str, Any]:
        """Load session context for matching task_code."""
        return self.store._load_session_context_for_task_impl(
            agent_id, session_id, current_task_code
        )
'''

    Path("src/search.py").write_text(content)
    print("✓ Created src/search.py")


def create_maintenance_module():
    """Create src/maintenance.py with stats and maintenance."""
    content = '''"""
Maintenance Operations
=====================

Statistics, health checks, and database maintenance.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MaintenanceOperations:
    """Maintenance operations for session memories."""

    def __init__(self, store):
        """Initialize with reference to parent store."""
        self.store = store

    def get_session_stats(
        self,
        agent_id: str | None = None,
        session_id: str | None = None
    ) -> dict[str, Any]:
        """Get statistics about session memory usage."""
        return self.store._get_session_stats_impl(agent_id, session_id)

    def list_sessions(
        self,
        agent_id: str | None = None,
        limit: int = 20
    ) -> dict[str, Any]:
        """List recent sessions with basic info."""
        return self.store._list_sessions_impl(agent_id, limit)

    def cleanup_old_memories(
        self,
        days_old: int = 90,
        memory_type: str | None = None
    ) -> dict[str, Any]:
        """Cleanup old memories (placeholder for future implementation)."""
        return {
            "success": False,
            "error": "Not implemented",
            "message": "Cleanup functionality not yet implemented"
        }
'''

    Path("src/maintenance.py").write_text(content)
    print("✓ Created src/maintenance.py")


def create_chunking_storage_module():
    """Create src/chunking_storage.py with chunk operations."""
    content = '''"""
Chunk Storage Operations
========================

Chunk-specific storage and retrieval operations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChunkingStorageOperations:
    """Chunk storage operations."""

    def __init__(self, store):
        """Initialize with reference to parent store."""
        self.store = store

    def expand_chunk_context(
        self,
        memory_id: int,
        chunk_index: int,
        context_window: int = 2
    ) -> dict[str, Any]:
        """Expand chunk context by retrieving surrounding chunks."""
        return self.store._expand_chunk_context_impl(
            memory_id, chunk_index, context_window
        )

    def get_chunk(
        self,
        chunk_id: int
    ) -> dict[str, Any]:
        """Get a specific chunk (placeholder for future implementation)."""
        return {
            "success": False,
            "error": "Not implemented",
            "message": "Get chunk functionality not yet implemented"
        }

    def list_chunks(
        self,
        memory_id: int
    ) -> dict[str, Any]:
        """List all chunks for a memory (placeholder)."""
        return {
            "success": False,
            "error": "Not implemented",
            "message": "List chunks functionality not yet implemented"
        }
'''

    Path("src/chunking_storage.py").write_text(content)
    print("✓ Created src/chunking_storage.py")


def update_session_memory_store():
    """Update session_memory_store.py to be a thin facade."""

    # Read original file
    original = Path("src/session_memory_store.py").read_text()

    # Keep everything up to and including the class definition and __init__
    # Then add delegation methods

    # Find where store_memory method starts
    store_memory_start = original.find("    def store_memory(")

    # Keep everything before store_memory
    header = original[:store_memory_start]

    # Add imports for new modules
    import_section = '''
# Import modular operations
from .storage import StorageOperations
from .search import SearchOperations
from .maintenance import MaintenanceOperations
from .chunking_storage import ChunkingStorageOperations

'''

    # Insert imports after existing imports (after memory_types import)
    memory_types_import = header.rfind("from .memory_types")
    memory_types_end = header.find("\n", memory_types_import)
    header = header[:memory_types_end + 1] + import_section + header[memory_types_end + 1:]

    # Add operation modules initialization in __init__
    init_additions = '''
        # Initialize operation modules
        self.storage = StorageOperations(self)
        self.search = SearchOperations(self)
        self.maintenance = MaintenanceOperations(self)
        self.chunking = ChunkingStorageOperations(self)
'''

    # Find end of __init__ (before @property)
    init_end = header.find("    @property")
    header = header[:init_end] + init_additions + "\n" + header[init_end:]

    # Now keep all the implementation methods but rename them to _*_impl
    impl_section = original[store_memory_start:]

    # Rename public methods to _*_impl
    methods_to_rename = [
        "store_memory",
        "search_memories",
        "search_with_granularity",
        "expand_chunk_context",
        "load_session_context_for_task",
        "get_memory",
        "get_session_stats",
        "list_sessions",
        "reconstruct_document",
        "write_document_to_file",
        "delete_memory"
    ]

    for method in methods_to_rename:
        impl_section = impl_section.replace(f"def {method}(", f"def _{method}_impl(")

    # Add delegation methods after properties
    delegation_methods = '''
    # ======================
    # PUBLIC API (Delegates to modules)
    # ======================

    def store_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Store a memory entry."""
        return self.storage.store_memory(*args, **kwargs)

    def search_memories(self, *args, **kwargs) -> dict[str, Any]:
        """Search memories with filters."""
        return self.search.search_memories(*args, **kwargs)

    def search_with_granularity(self, *args, **kwargs) -> dict[str, Any]:
        """Search with specific granularity."""
        return self.search.search_with_granularity(*args, **kwargs)

    def expand_chunk_context(self, *args, **kwargs) -> dict[str, Any]:
        """Expand chunk context."""
        return self.chunking.expand_chunk_context(*args, **kwargs)

    def load_session_context_for_task(self, *args, **kwargs) -> dict[str, Any]:
        """Load session context for task."""
        return self.search.load_session_context_for_task(*args, **kwargs)

    def get_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Get memory by ID."""
        return self.storage.get_memory(*args, **kwargs)

    def get_session_stats(self, *args, **kwargs) -> dict[str, Any]:
        """Get session statistics."""
        return self.maintenance.get_session_stats(*args, **kwargs)

    def list_sessions(self, *args, **kwargs) -> dict[str, Any]:
        """List sessions."""
        return self.maintenance.list_sessions(*args, **kwargs)

    def reconstruct_document(self, *args, **kwargs) -> dict[str, Any]:
        """Reconstruct document from chunks."""
        return self.storage.reconstruct_document(*args, **kwargs)

    def write_document_to_file(self, *args, **kwargs) -> dict[str, Any]:
        """Write document to file."""
        return self.storage.write_document_to_file(*args, **kwargs)

    def delete_memory(self, *args, **kwargs) -> dict[str, Any]:
        """Delete memory."""
        return self.storage.delete_memory(*args, **kwargs)

    # ======================
    # IMPLEMENTATION METHODS
    # ======================
'''

    # Find where to insert delegation methods (after embedding_model property)
    embedding_end = header.rfind("return self._embedding_model if self._embedding_model is not False else None")
    embedding_line_end = header.find("\n", embedding_end)

    final_content = header[:embedding_line_end + 1] + "\n" + delegation_methods + "\n" + impl_section

    # Write updated file
    Path("src/session_memory_store.py").write_text(final_content)
    print("✓ Updated src/session_memory_store.py (thin facade)")


def main():
    """Execute modular refactoring."""
    print("Starting modular refactoring...")
    print()

    create_storage_module()
    create_search_module()
    create_maintenance_module()
    create_chunking_storage_module()
    update_session_memory_store()

    print()
    print("✅ Modular refactoring complete!")
    print()
    print("New structure:")
    print("  - src/storage.py (CRUD operations)")
    print("  - src/search.py (search operations)")
    print("  - src/maintenance.py (stats, maintenance)")
    print("  - src/chunking_storage.py (chunk operations)")
    print("  - src/session_memory_store.py (thin facade)")


if __name__ == "__main__":
    main()
