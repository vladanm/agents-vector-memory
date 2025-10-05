"""
Memory Types and Content Format Definitions
===========================================

Defines content formats, chunk types, and memory type configurations
for enhanced document processing capabilities.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


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
        "system_memory": {
            "chunk_size": 600,
            "chunk_overlap": 60,
            "preserve_structure": False,
            "default_auto_chunk": True
        },
        "session_context": {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_structure": False,
            "default_auto_chunk": True
        },
        "input_prompt": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "preserve_structure": False,
            "default_auto_chunk": True
        },
        "report_observations": {
            "chunk_size": 800,
            "chunk_overlap": 80,
            "preserve_structure": False,
            "default_auto_chunk": True
        }
    }

    return configs.get(memory_type, {
        "chunk_size": 800,
        "chunk_overlap": 80,
        "preserve_structure": False,
        "default_auto_chunk": False
    })