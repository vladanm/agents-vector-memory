"""
Security Utilities for Agent Session Memory
===========================================

Provides security validation, input sanitization, and session validation
for the agent session memory MCP server.
"""

import re
import os
import hashlib
from pathlib import Path
from typing import List

from .config import Config


class SecurityError(Exception):
    """Raised when security validation fails"""
    pass


def validate_working_dir(working_dir: str) -> Path:
    """
    Validate and normalize working directory path.
    
    Args:
        working_dir: Directory path to validate
        
    Returns:
        Path: Validated memory directory path
        
    Raises:
        SecurityError: If path is invalid or unsafe
    """
    try:
        path = Path(working_dir).resolve()
        
        # Check if path exists or can be created
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise SecurityError(f"Cannot create working directory: {e}")
        
        # Check if it's a directory
        if not path.is_dir():
            raise SecurityError(f"Path is not a directory: {path}")
        
        # Check write permissions
        test_file = path / ".test_write"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise SecurityError(f"No write permissions: {e}")
        
        return path
        
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        raise SecurityError(f"Invalid working directory: {e}")


def validate_agent_id(agent_id: str) -> str:
    """
    Validate agent identifier.
    
    Args:
        agent_id: Agent identifier to validate
        
    Returns:
        str: Validated agent ID
        
    Raises:
        SecurityError: If agent ID is invalid
    """
    if not agent_id or not isinstance(agent_id, str):
        raise SecurityError("Agent ID must be a non-empty string")
    
    # Check length
    if len(agent_id) > 100:
        raise SecurityError("Agent ID too long (max 100 characters)")
    
    # Check for dangerous characters
    if re.search(r'[<>"\'\\\x00-\x1f]', agent_id):
        raise SecurityError("Agent ID contains invalid characters")
    
    # Validate against known types
    if agent_id not in ["main"] and not agent_id.startswith("specialized-"):
        # Allow custom agent IDs but warn
        pass
    
    return agent_id.strip()


def validate_session_id(session_id: str) -> str:
    """
    Validate session identifier.
    
    Args:
        session_id: Session identifier to validate
        
    Returns:
        str: Validated session ID
        
    Raises:
        SecurityError: If session ID is invalid
    """
    if not session_id or not isinstance(session_id, str):
        raise SecurityError("Session ID must be a non-empty string")
    
    # Check length
    if len(session_id) > 200:
        raise SecurityError("Session ID too long (max 200 characters)")
    
    # Check for dangerous characters
    if re.search(r'[<>"\'\\\x00-\x1f]', session_id):
        raise SecurityError("Session ID contains invalid characters")
    
    return session_id.strip()


def validate_task_code(task_code: str) -> str:
    """
    Validate task code identifier.
    
    Args:
        task_code: Task code to validate
        
    Returns:
        str: Validated task code
        
    Raises:
        SecurityError: If task code is invalid
    """
    if not task_code:
        return task_code
    
    if not isinstance(task_code, str):
        raise SecurityError("Task code must be a string")
    
    # Check length
    if len(task_code) > 100:
        raise SecurityError("Task code too long (max 100 characters)")
    
    # Check for dangerous characters
    if re.search(r'[<>"\'\\\x00-\x1f]', task_code):
        raise SecurityError("Task code contains invalid characters")
    
    return task_code.strip()


def validate_memory_type(memory_type: str) -> str:
    """
    Validate memory type.
    
    Args:
        memory_type: Memory type to validate
        
    Returns:
        str: Validated memory type
        
    Raises:
        SecurityError: If memory type is invalid
    """
    if not memory_type or not isinstance(memory_type, str):
        raise SecurityError("Memory type must be a non-empty string")
    
    if not Config.validate_memory_type(memory_type):
        raise SecurityError(f"Invalid memory type: {memory_type}. Valid types: {Config.MEMORY_TYPES}")
    
    return memory_type


def validate_content(content: str) -> str:
    """
    Validate memory content.
    
    Args:
        content: Content to validate
        
    Returns:
        str: Validated content
        
    Raises:
        SecurityError: If content is invalid
    """
    if not content or not isinstance(content, str):
        raise SecurityError("Content must be a non-empty string")
    
    # Check length
    if len(content) > Config.MAX_MEMORY_LENGTH:
        raise SecurityError(f"Content too long (max {Config.MAX_MEMORY_LENGTH} characters)")
    
    # Check for null bytes
    if '\x00' in content:
        raise SecurityError("Content contains null bytes")
    
    return content


def validate_session_iter(session_iter: int) -> int:
    """
    Validate session iteration number.
    
    Args:
        session_iter: Session iteration to validate
        
    Returns:
        int: Validated session iteration
        
    Raises:
        SecurityError: If session iteration is invalid
    """
    if session_iter is None:
        return Config.DEFAULT_SESSION_ITER
    
    if not isinstance(session_iter, int):
        raise SecurityError("Session iteration must be an integer")
    
    if not Config.validate_session_iter(session_iter):
        raise SecurityError(f"Session iteration out of range (1-{Config.MAX_SESSION_ITER})")
    
    return session_iter


def validate_tags(tags: List[str]) -> List[str]:
    """
    Validate list of tags.
    
    Args:
        tags: List of tags to validate
        
    Returns:
        List[str]: Validated tags
        
    Raises:
        SecurityError: If tags are invalid
    """
    if not tags:
        return []
    
    if not isinstance(tags, list):
        raise SecurityError("Tags must be a list")
    
    if len(tags) > Config.MAX_TAGS_PER_MEMORY:
        raise SecurityError(f"Too many tags (max {Config.MAX_TAGS_PER_MEMORY})")
    
    validated_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            raise SecurityError("All tags must be strings")
        
        if len(tag) > Config.MAX_TAG_LENGTH:
            raise SecurityError(f"Tag too long (max {Config.MAX_TAG_LENGTH} characters): {tag}")
        
        # Check for dangerous characters
        if re.search(r'[<>"\'\\\x00-\x1f]', tag):
            raise SecurityError(f"Tag contains invalid characters: {tag}")
        
        clean_tag = tag.strip()
        if clean_tag and clean_tag not in validated_tags:
            validated_tags.append(clean_tag)
    
    return validated_tags


def generate_content_hash(content: str) -> str:
    """
    Generate a hash of the content for deduplication.
    
    Args:
        content: Content to hash
        
    Returns:
        str: SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def sanitize_input(text: str, max_length: int = None) -> str:
    """
    Sanitize text input by removing dangerous characters.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Apply length limit if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()
    
    return sanitized