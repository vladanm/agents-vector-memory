"""
Configuration for Agent Session Memory MCP Server
=================================================

Centralized configuration management for the session-based memory system.
"""

import os
from pathlib import Path
from typing import List


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
    MAX_MEMORY_LENGTH = 100000       # Increased for reports
    MAX_MEMORIES_PER_SEARCH = 100    # Increased for session searches
    MAX_TOTAL_MEMORIES = 100000      # Increased capacity
    MAX_TAG_LENGTH = 100
    MAX_TAGS_PER_MEMORY = 20         # Increased for better categorization
    MAX_CHUNK_SIZE = 2000
    MAX_CHUNKS_PER_DOCUMENT = 1000   # Increased for large reports
    
    # Session configuration
    DEFAULT_SESSION_ITER = 1
    MAX_SESSION_ITER = 1000
    MAX_SEARCH_RESULTS = 100
    
    # Working directory (set by initialization)
    working_dir = os.getcwd()
    
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