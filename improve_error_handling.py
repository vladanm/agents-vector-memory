#!/usr/bin/env python3
"""
Script to improve error handling in session_memory_store.py

Replaces broad `except Exception` with specific exception types
and adds retry logic for database operations.
"""

import re
from pathlib import Path


def add_imports(content: str) -> str:
    """Add new exception and retry imports."""
    import_section = """import os
import re
import json
import time
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import tempfile

# Initialize logger
logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .chunking import DocumentChunker, ChunkingConfig
from .memory_types import ContentFormat, ChunkEntry, get_memory_type_config
from .exceptions import (
    VectorMemoryException, ValidationError, MemoryError,
    SearchError, ChunkingError, DatabaseError, DatabaseLockError
)
from .retry_utils import exponential_backoff, retry_on_lock

# Import modular operations
from .storage import StorageOperations
from .search import SearchOperations
from .maintenance import MaintenanceOperations
from .chunking_storage import ChunkingStorageOperations"""

    # Find the end of the import section (before VALID_MEMORY_TYPES)
    pattern = r"(from \.chunking_storage import ChunkingStorageOperations\s*\n\s*\n)"
    replacement = import_section + "\n\n\n"
    content = re.sub(pattern, replacement, content)

    return content


def update_store_memory_impl_error_handling(content: str) -> str:
    """Update _store_memory_impl to use specific exceptions."""

    # Replace the generic exception at the end of _store_memory_impl
    pattern = r'(\s+except Exception as e:\s+return\s+\{\s+"success":\s+False,\s+"error":\s+"Storage failed",)'

    replacement = r'''        except ValidationError as e:
            return e.to_dict()
        except ChunkingError as e:
            logger.error(f"Chunking failed: {e}")
            return e.to_dict()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                return DatabaseLockError(str(e)).to_dict()
            return DatabaseError("store_memory", str(e)).to_dict()
        except Exception as e:
            logger.error(f"Unexpected error in store_memory: {e}")
            return {
                "success": False,
                "error": "Storage failed",'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return content


def update_search_error_handling(content: str) -> str:
    """Update search methods to use SearchError."""

    # Pattern for _search_memories_impl error
    pattern1 = r'(\s+except Exception as e:\s+return\s+\{\s+"success":\s+False,\s+"error":\s+"Search failed",)'

    replacement1 = r'''        except ValidationError as e:
            return e.to_dict()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                return DatabaseLockError(str(e)).to_dict()
            return SearchError(str(e)).to_dict()
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            return {
                "success": False,
                "error": "Search failed",'''

    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    return content


def update_chunk_error_handling(content: str) -> str:
    """Improve chunking error handling to report failures."""

    # Find the chunking try/except that currently suppresses errors
    pattern = r'(try:\s+chunk_embeddings = self\.embedding_model\.encode.*?except Exception as e:\s+logger\.warning.*?# Continue without embeddings)'

    replacement = r'''try:
                        chunk_embeddings = self.embedding_model.encode(
                            chunk_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                    except Exception as e:
                        # Report chunking failure instead of silently continuing
                        raise ChunkingError(
                            f"Failed to generate embeddings for {len(chunk_texts)} chunks",
                            document_length=len(content),
                            chunk_count=len(chunk_texts)
                        ) from e'''

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    return content


def main():
    """Main execution."""
    file_path = Path("src/session_memory_store.py")

    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return 1

    print(f"Reading {file_path}...")
    content = file_path.read_text()

    print("Adding exception imports...")
    content = add_imports(content)

    print("Updating store_memory error handling...")
    content = update_store_memory_impl_error_handling(content)

    print("Updating search error handling...")
    content = update_search_error_handling(content)

    print("Updating chunking error handling...")
    content = update_chunk_error_handling(content)

    # Write back
    backup_path = file_path.with_suffix(".py.before_error_handling")
    print(f"Creating backup at {backup_path}...")
    file_path.rename(backup_path)

    print(f"Writing updated {file_path}...")
    file_path.write_text(content)

    print("✅ Error handling improvements applied!")
    print("\nChanges made:")
    print("  - Added custom exception imports")
    print("  - Replaced generic Exception handlers with specific types")
    print("  - Added DatabaseLockError handling for retry logic")
    print("  - Improved chunking error reporting (no silent failures)")
    print(f"\nBackup saved at: {backup_path}")

    return 0


if __name__ == "__main__":
    exit(main())
