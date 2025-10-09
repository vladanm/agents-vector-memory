"""
Configuration for Agent Session Memory MCP Server
=================================================

Centralized configuration management for the session-based memory system.
"""

import os
from pathlib import Path


class Config:
    """Configuration constants for agent session memory"""

    # Server configuration
    SERVER_NAME = "Agent Session Memory MCP Server"
    SERVER_VERSION = "1.0.0"

    # Database configuration
    DB_NAME = "agent_session_memory.db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Session memory types
    MEMORY_TYPES = [
        "knowledge_base",     # Global/session-scoped shared knowledge (NEW)
        "session_context",    # Agent session snapshots for continuity
        "input_prompt",       # Original user prompts to prevent loss
        "reports",           # Agent-generated analysis and findings
        "working_memory",    # Important info during task execution
        "system_memory",     # System configs, commands, scripts
        "report_observations" # Additional notes on existing reports
    ]

    # Agent types
    AGENT_TYPES = [
        "main",             # Main agent
        "specialized-agent" # Sub-agents
    ]

    # Security limits
    MAX_MEMORY_LENGTH = 500000       # 500k characters (~125k tokens) - supports large technical documents
    MAX_MEMORIES_PER_SEARCH = 100    # Increased for session searches
    MAX_TOTAL_MEMORIES = 100000      # Increased capacity
    MAX_TAG_LENGTH = 100
    MAX_TAGS_PER_MEMORY = 20         # Increased for better categorization
    MAX_CHUNK_SIZE = 2000
    MAX_CHUNKS_PER_DOCUMENT = 5000   # 5x increase - supports documents up to ~6M tokens when chunked

    # Chunking configuration (for document chunking feature)
    DEFAULT_CHUNK_SIZE = 450         # Was 800 - reduced to fit within 512 token embedding limit
    DEFAULT_CHUNK_OVERLAP = 50       # Was 80 - 11% overlap maintained
    MIN_CHUNK_SIZE = 150             # Was 250 - reduced to allow smaller natural boundaries

    # Session configuration
    DEFAULT_SESSION_ITER = 1
    MAX_SESSION_ITER = 1000
    MAX_SEARCH_RESULTS = 100

    # Working directory (set by initialization)
    working_dir = os.getcwd()

    # ======================
    # TASK 7: PERFORMANCE MONITORING & OPTIMIZATION
    # ======================

    # Vector search performance tuning
    VECTOR_SEARCH_BATCH_SIZE = 100          # Initial batch size for iterative search
    VECTOR_SEARCH_MAX_OFFSET = 1_000_000   # Safety limit for iterative search
    VECTOR_SEARCH_GROWTH_FACTOR = 2         # Batch size growth when selectivity low

    # Future hybrid search configuration (BM25 + vector)
    BM25_MAX_RESULTS = 50                   # Max BM25 candidates for fusion
    VECTOR_MAX_RESULTS = 50                 # Max vector candidates for fusion

    # Future reranking configuration
    RERANK_TOP_N = 20                       # Number of results to rerank
    EMBEDDING_BATCH_SIZE = 32               # Batch size for embedding generation

    # Performance logging configuration
    LOG_QUERY_TIMING = True                 # Log query execution time
    LOG_SLOW_QUERY_THRESHOLD = 2.0          # Log warning if query > 2s
    LOG_FILTER_STATS = True                 # Log metadata filtering stats
    LOG_TIMING_TO_RESPONSE = True           # Include timing in response metadata

    # Feature flags (for progressive rollout & rollback capability)
    USE_ITERATIVE_FETCHING = True           # Task 2: Iterative post-filter fetching
    USE_THRESHOLD_FILTERING = False         # Task 1: Hard threshold filtering (DISABLED)
    AUTO_BACKFILL_THRESHOLD = 1000          # Task 6: Auto-backfill if < N chunks missing
    WARM_START_EMBEDDING_MODEL = True       # Task 6: Pre-load embedding model at startup

    # Performance targets (for monitoring & alerting)
    TARGET_P50_MS = 500                     # Target P50 latency in milliseconds
    TARGET_P95_MS = 2000                    # Target P95 latency in milliseconds
    TARGET_P99_MS = 5000                    # Target P99 latency in milliseconds
    TARGET_RECALL_AT_10 = 0.90              # Target recall@10 for fine granularity

    @classmethod
    def get_db_path(cls) -> Path:
        """Get the database file path"""
        return Path(cls.working_dir) / "memory" / cls.DB_NAME

    @classmethod
    def validate_memory_type(cls, memory_type: str) -> bool:
        """Validate if memory type is supported"""
        return memory_type in cls.MEMORY_TYPES

    @classmethod
    def validate_agent_type(cls, agent_id: str) -> bool:
        """Validate if agent type is supported"""
        return agent_id in cls.AGENT_TYPES or agent_id.startswith("specialized-")

    @classmethod
    def validate_session_iter(cls, session_iter: int) -> bool:
        """Validate session iteration number"""
        return 1 <= session_iter <= cls.MAX_SESSION_ITER
