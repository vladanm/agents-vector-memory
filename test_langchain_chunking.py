#!/usr/bin/env python3
"""
Test LangChain chunking implementation
"""

import sys
from pathlib import Path
import sqlite3

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.session_memory_store import SessionMemoryStore

# Test document with various header levels
TEST_CONTENT = """# LangChain Text Splitters

## Overview
LangChain provides powerful text splitting utilities for chunking documents while preserving semantic structure and maintaining reasonable chunk sizes.

## MarkdownHeaderTextSplitter
This splitter intelligently splits markdown documents based on header hierarchy, maintaining context and structure throughout the chunking process.

### Key Features
The MarkdownHeaderTextSplitter preserves header metadata in each chunk, allowing for better context preservation and semantic search capabilities.

### Configuration Options
You can customize which header levels to split on, whether to strip headers from content, and how to handle metadata propagation.

## RecursiveCharacterTextSplitter
Works in conjunction with the markdown splitter to enforce size constraints on chunks.

### Size Management
The recursive splitter ensures chunks don't exceed specified token limits while respecting natural text boundaries like paragraphs and sentences.

### Overlap Strategy
Configurable overlap between chunks helps maintain context across chunk boundaries, improving semantic coherence.

## Integration Benefits
Using LangChain splitters eliminates custom chunking logic and provides battle-tested, well-maintained code for text processing.

### Performance
The LangChain implementation is optimized for speed and memory efficiency, handling large documents with ease.

### Maintenance
By leveraging a widely-used library, we benefit from community contributions, bug fixes, and ongoing improvements.
"""

def test_langchain_chunking():
    """Test that LangChain chunking produces properly sized chunks"""

    # Use test database
    db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
    store = SessionMemoryStore(db_path)

    print("=" * 80)
    print("TESTING LANGCHAIN CHUNKING")
    print("=" * 80)

    # Store document with auto_chunk
    result = store.store_memory(
        memory_type="reports",
        agent_id="langchain-test",
        session_id="test-lc",
        content=TEST_CONTENT,
        title="LangChain Splitters Test",
        auto_chunk=True
    )

    if not result.get("success"):
        print(f"\n❌ Failed to store memory: {result.get('error', 'Unknown error')}")
        print(f"   Message: {result.get('message', 'No message')}")
        return False

    memory_id = result["memory_id"]
    print(f"\n✓ Stored memory ID: {memory_id}")
    print(f"✓ Auto-chunk enabled: {result.get('chunks_stored', 0) > 0}")

    # Query chunks directly from database
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT
            id,
            chunk_index,
            token_count,
            header_path,
            level,
            substr(content, 1, 80) as content_preview
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (memory_id,))

    chunks = cursor.fetchall()
    conn.close()

    print(f"\n✓ Total chunks created: {len(chunks)}")
    print("\n" + "=" * 80)
    print("CHUNK ANALYSIS")
    print("=" * 80)

    min_tokens = float('inf')
    max_tokens = 0
    total_tokens = 0

    for chunk_id, idx, token_count, header_path, level, preview in chunks:
        min_tokens = min(min_tokens, token_count)
        max_tokens = max(max_tokens, token_count)
        total_tokens += token_count

        print(f"\nChunk {idx}:")
        print(f"  ID: {chunk_id}")
        print(f"  Tokens: {token_count}")
        print(f"  Level: {level}")
        print(f"  Header Path: {header_path}")
        print(f"  Preview: {preview}...")

    avg_tokens = total_tokens / len(chunks) if chunks else 0

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")
    print(f"Min tokens: {min_tokens}")
    print(f"Max tokens: {max_tokens}")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Total tokens: {total_tokens}")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    success = True

    # Check: No tiny chunks (should be at least 100 tokens minimum in practice)
    if min_tokens < 50:
        print(f"❌ FAILED: Found chunk with only {min_tokens} tokens (too small)")
        success = False
    else:
        print(f"✓ PASSED: Minimum chunk size ({min_tokens} tokens) is reasonable")

    # Check: Chunks within reasonable size (reports have 1500 token limit)
    if max_tokens > 1600:
        print(f"❌ FAILED: Found chunk with {max_tokens} tokens (exceeds limit)")
        success = False
    else:
        print(f"✓ PASSED: Maximum chunk size ({max_tokens} tokens) is within limits")

    # Check: At least some chunks created
    if len(chunks) < 2:
        print(f"❌ FAILED: Only {len(chunks)} chunk(s) created")
        success = False
    else:
        print(f"✓ PASSED: {len(chunks)} chunks created")

    # Check: Header paths preserved
    has_headers = any(h for _, _, _, h, _, _ in chunks if h)
    if has_headers:
        print("✓ PASSED: Header paths preserved in chunks")
    else:
        print("⚠ WARNING: No header paths found in chunks")

    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)

    return success

if __name__ == "__main__":
    try:
        success = test_langchain_chunking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
