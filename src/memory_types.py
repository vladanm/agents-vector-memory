"""
Memory Types and Content Format Definitions
===========================================

Defines content formats, chunk types, and memory type configurations
for enhanced document processing capabilities.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
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
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None
    header_path: Optional[str] = None
    level: int = 0
    prev_chunk_id: Optional[int] = None
    next_chunk_id: Optional[int] = None
    content_hash: str = ""
    id: Optional[int] = None
    original_content: Optional[str] = None
    is_contextually_enriched: bool = False
    granularity_level: str = "medium"

    # Additional fields for database insertion
    created_at: Optional[str] = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_title: Optional[str] = None
    section_hierarchy: Optional[str] = None
    chunk_position_ratio: Optional[float] = None
    sibling_count: Optional[int] = None
    depth_level: Optional[int] = None
    contains_code: bool = False
    contains_table: bool = False
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[bytes] = None


def get_memory_type_config(memory_type: str) -> Dict[str, Any]:
    """
    Get chunking configuration for a specific memory type.

    Args:
        memory_type: The type of memory

    Returns:
        Configuration dictionary with chunking parameters
    """
    configs = {
        "knowledge_base": {
            "chunk_size": 1200,
            "chunk_overlap": 120,
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "reports": {
            "chunk_size": 1500,
            "chunk_overlap": 150,
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "working_memory": {
            "chunk_size": 800,
            "chunk_overlap": 80,
            "preserve_structure": True,
            "default_auto_chunk": True
        },
        "session_context": {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_structure": False,
            "default_auto_chunk": False
        },
        "input_prompt": {
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "preserve_structure": False,
            "default_auto_chunk": False
        },
        "system_memory": {
            "chunk_size": 1500,
            "chunk_overlap": 150,
            "preserve_structure": True,
            "default_auto_chunk": False
        }
    }

    # Default config if memory_type not found
    default_config = {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "preserve_structure": False,
        "default_auto_chunk": False
    }

    return configs.get(memory_type, default_config)


def get_valid_memory_types() -> List[str]:
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
