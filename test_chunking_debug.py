#!/usr/bin/env python3
"""Debug script to test chunking functionality"""

import sys
sys.path.insert(0, '.')

from src.chunking import DocumentChunker
from src.memory_types import get_memory_type_config

# Test content
test_content = """---
title: Test Document
---

# Main Header

Some intro content.

## Section 1

Content for section 1.

### Subsection 1.1

Detailed content here.

## Section 2

More content.
"""

print("=" * 60)
print("Testing Document Chunking")
print("=" * 60)

# Get config for reports
config = get_memory_type_config("reports")
print(f"\nConfig for 'reports' memory type:")
print(f"  chunk_size: {config.get('chunk_size')}")
print(f"  chunk_overlap: {config.get('chunk_overlap')}")
print(f"  preserve_structure: {config.get('preserve_structure')}")

# Create chunker
chunker = DocumentChunker()
print("\nChunker created")

# Chunk the document
try:
    chunks = chunker.chunk_document(
        content=test_content,
        parent_id=999,
        metadata={"memory_type": "reports"}
    )

    print(f"\n✅ Chunking succeeded!")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  chunk_index: {chunk.chunk_index}")
        print(f"  chunk_type: {chunk.chunk_type}")
        print(f"  header_path: {chunk.header_path}")
        print(f"  level: {chunk.level}")
        print(f"  token_count: {chunk.token_count}")
        print(f"  content preview: {chunk.content[:100]}...")

except Exception as e:
    print(f"\n❌ Chunking failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)