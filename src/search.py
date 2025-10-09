"""
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
        similarity_threshold: float = 0.5,
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
