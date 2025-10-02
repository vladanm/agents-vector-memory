#!/usr/bin/env python3
"""
Test YAML frontmatter document storage
"""

import sqlite3

# Test document with YAML frontmatter
TEST_CONTENT = """---
title: LangChain Text Splitters Reference
category: documentation
tags: [langchain, chunking, markdown]
---

# LangChain Text Splitters

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

def check_latest_memory():
    """Check the latest memory and its chunks"""
    db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"

    conn = sqlite3.connect(db_path)

    # Get latest memory
    cursor = conn.execute("""
        SELECT id, memory_type, created_at
        FROM session_memories
        ORDER BY id DESC
        LIMIT 1
    """)

    memory = cursor.fetchone()
    if not memory:
        print("No memories found")
        return

    memory_id, memory_type, created_at = memory

    print("="*80)
    print(f"Latest Memory: ID {memory_id}")
    print("="*80)
    print(f"Type: {memory_type}")
    print(f"Created: {created_at}")

    # Get chunks
    cursor = conn.execute("""
        SELECT chunk_index, token_count, level, header_path, substr(content, 1, 80)
        FROM memory_chunks
        WHERE parent_id = ?
        ORDER BY chunk_index
    """, (memory_id,))

    chunks = cursor.fetchall()
    conn.close()

    if not chunks:
        print("\n❌ No chunks found!")
        return

    print(f"\n✓ Total chunks: {len(chunks)}")
    print("\n" + "="*80)
    print("CHUNK ANALYSIS")
    print("="*80)

    min_tokens = min(c[1] for c in chunks)
    max_tokens = max(c[1] for c in chunks)
    avg_tokens = sum(c[1] for c in chunks) / len(chunks)

    for idx, tokens, level, header_path, preview in chunks:
        print(f"\nChunk {idx}:")
        print(f"  Tokens: {tokens}")
        print(f"  Level: {level}")
        print(f"  Header: {header_path}")
        print(f"  Preview: {preview}...")

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Min tokens: {min_tokens}")
    print(f"Max tokens: {max_tokens}")
    print(f"Avg tokens: {avg_tokens:.1f}")

    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    if min_tokens >= 250:
        print(f"✅ PASSED: Minimum chunk size ({min_tokens} tokens) meets requirement")
    else:
        print(f"❌ FAILED: Minimum chunk size ({min_tokens} tokens) below 250 token minimum")

    if max_tokens <= 1600:
        print(f"✅ PASSED: Maximum chunk size ({max_tokens} tokens) within limits")
    else:
        print(f"❌ FAILED: Maximum chunk size ({max_tokens} tokens) exceeds limit")

    if len(chunks) <= 5:
        print(f"✅ PASSED: Reasonable chunk count ({len(chunks)} chunks)")
    else:
        print(f"⚠ WARNING: High chunk count ({len(chunks)} chunks) - expected ~2-3 for this document")

if __name__ == "__main__":
    check_latest_memory()
