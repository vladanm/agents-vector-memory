"""
Memory Types and Content Format Definitions
===========================================

Defines content formats, chunk types, and memory type configurations
for enhanced document processing capabilities.
"""

from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ContentFormat(Enum):
    """Content format types for specialized processing"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    CODE = "code"


@dataclass
class ChunkEntry:
    """Represents a single chunk of a larger document"""
    parent_id: int
    chunk_index: int
    content: str
    chunk_type: str = "text"
    start_char: int | None = None
    end_char: int | None = None
    token_count: int | None = None
    header_path: str | None = None
    level: int = 0
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None
    content_hash: str = ""
    id: int | None = None
    original_content: str | None = None
    is_contextually_enriched: bool = False
    granularity_level: str = "medium"

    # Additional fields for database insertion
    created_at: str | None = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_title: str | None = None
    section_hierarchy: str | None = None
    chunk_position_ratio: float | None = None
    sibling_count: int | None = None
    depth_level: int | None = None
    contains_code: bool = False
    contains_table: bool = False
    keywords: list[str] = field(default_factory=list)
    embedding: bytes | None = None


def get_memory_type_config(memory_type: str) -> dict[str, Any]:
    """
    Get chunking configuration for a specific memory type.

    Args:
        memory_type: The type of memory

    Returns:
        Configuration dictionary with chunking parameters
    """
    configs = {
        "knowledge_base": {
            "chunk_size": 512,      # Was 1200 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 64,    # Was 120 - 12.5% overlap maintained
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "reports": {
            "chunk_size": 512,      # Was 1500 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 64,    # Was 150 - 12.5% overlap maintained
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "working_memory": {
            "chunk_size": 400,      # Was 800 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 48,    # Was 80 - 12% overlap maintained
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "session_context": {
            "chunk_size": 450,      # Was 1000 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 50,    # Was 100 - 11% overlap maintained
            "preserve_structure": False,
            "default_auto_chunk": False
        },
        "input_prompt": {
            "chunk_size": 512,      # Was 2000 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 64,    # Was 200 - 12.5% overlap maintained
            "preserve_structure": False,
            "default_auto_chunk": False
        },
        "system_memory": {
            "chunk_size": 512,      # Was 1500 - reduced to fit within 512 token embedding limit
            "chunk_overlap": 64,    # Was 150 - 12.5% overlap maintained
            "preserve_structure": True,
            "default_auto_chunk": False
        }
    }

    # Default config if memory_type not found
    default_config = {
        "chunk_size": 450,          # Was 1000 - reduced to fit within 512 token embedding limit
        "chunk_overlap": 50,        # Was 100 - 11% overlap maintained
        "preserve_structure": False,
        "default_auto_chunk": False
    }

    return configs.get(memory_type, default_config)


def get_valid_memory_types() -> list[str]:
    """Get list of valid memory types"""
    return [
        "session_context",
        "input_prompt",
        "system_memory",
        "knowledge_base",
        "reports",
        "working_memory",
        "report_observation"
    ]


# Export constants
VALID_MEMORY_TYPES = get_valid_memory_types()
