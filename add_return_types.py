#!/usr/bin/env python3
"""
Add missing return type annotations to functions.
"""

import re
from pathlib import Path

def add_return_types_to_main(file_path: Path) -> None:
    """Add return types to main.py MCP tool functions."""
    content = file_path.read_text()

    # All MCP tool functions return dict[str, Any] (they wrap store results)
    mcp_tools = [
        "store_session_context",
        "store_input_prompt",
        "store_system_memory",
        "store_report",
        "store_knowledge_base",
        "store_report_observation",
        "store_working_memory",
        "search_session_context",
        "search_system_memory",
        "search_input_prompts",
        "search_knowledge_base",
        "search_reports_specific_chunks",
        "search_reports_section_context",
        "search_reports_full_documents",
        "search_working_memory_specific_chunks",
        "search_working_memory_section_context",
        "search_working_memory_full_documents",
        "load_session_context_for_task",
        "get_memory_by_id",
        "get_session_stats",
        "list_sessions",
        "reconstruct_document",
        "write_document_to_file",
        "delete_memory",
        "expand_chunk_context",
        "cleanup_old_memories"
    ]

    for tool in mcp_tools:
        # Find function definition without return type
        pattern = rf"(def {tool}\([^)]*\)):"
        replacement = r"\1 -> dict[str, Any]:"
        content = re.sub(pattern, replacement, content)

    # initialize_store returns None
    content = re.sub(
        r"(def initialize_store\([^)]*\)):",
        r"\1 -> None:",
        content
    )

    file_path.write_text(content)
    print(f"✓ Updated {file_path.name}")

def add_return_types_to_store(file_path: Path) -> None:
    """Add return types to session_memory_store.py methods."""
    content = file_path.read_text()

    # Properties return appropriate types
    replacements = [
        (r"(def chunker\(self\)):", r"\1 -> 'DocumentChunker':"),
        (r"(def token_encoder\(self\)):", r"\1 -> Any:"),
        (r"(def embedding_model\(self\)):", r"\1 -> Any:"),
        (r"(def __enter__\(self\)):", r"\1 -> 'SessionMemoryStore':"),
        (r"(def __exit__\(self, [^)]+\)):", r"\1 -> bool:"),
        (r"(def _init_schema\(self\)):", r"\1 -> None:"),
        (r"(def _migrate_schema\(self, [^)]+\)):", r"\1 -> int:"),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    file_path.write_text(content)
    print(f"✓ Updated {file_path.name}")

def main():
    base_dir = Path(__file__).parent

    # Update main.py
    main_py = base_dir / "main.py"
    if main_py.exists():
        add_return_types_to_main(main_py)

    # Update session_memory_store.py
    store_py = base_dir / "src" / "session_memory_store.py"
    if store_py.exists():
        add_return_types_to_store(store_py)

    print("\n✅ Return type annotations added!")

if __name__ == "__main__":
    main()
