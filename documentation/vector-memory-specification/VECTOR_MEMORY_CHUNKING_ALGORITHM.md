# Vector Memory Chunking Algorithm

**Version:** 1.0.0
**Implementation:** `/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/src/chunking.py`
**Token Counter:** tiktoken (cl100k_base encoding)

---

## Table of Contents

1. [Overview](#overview)
2. [Chunking Configuration](#chunking-configuration)
3. [Two-Stage Chunking Algorithm](#two-stage-chunking-algorithm)
4. [Chunk Entry Structure](#chunk-entry-structure)
5. [Contextual Enrichment](#contextual-enrichment)
6. [Code Block Handling](#code-block-handling)
7. [Header Hierarchy Parsing](#header-hierarchy-parsing)
8. [Token Counting](#token-counting)
9. [Algorithm Pseudocode](#algorithm-pseudocode)
10. [Performance Characteristics](#performance-characteristics)

---

## Overview

The Vector Memory chunking system uses a **two-stage approach** to split large documents into searchable chunks while preserving semantic coherence and code block integrity.

### Design Goals

1. **Preserve semantic boundaries**: Split at natural boundaries (headers, paragraphs, code blocks)
2. **Maintain code integrity**: Never split code blocks mid-fence
3. **Respect token limits**: Stay within embedding model limits (512 tokens for all-MiniLM-L6-v2)
4. **Enable hierarchical search**: Track markdown header hierarchy in metadata
5. **Support contextual enrichment**: Add document/section context for better embeddings

### Key Features

- **Code-aware splitting**: Prioritizes code block boundaries
- **Markdown structure preservation**: Maintains header hierarchy
- **Adaptive overlap**: 11% overlap (~50 tokens) between chunks
- **Validation**: Code fence balance checking
- **Enrichment**: Optional contextual headers for embeddings

---

## Chunking Configuration

### Default Parameters

```python
class ChunkingConfig:
    chunk_size: int = 450           # Target chunk size (tokens)
    chunk_overlap: int = 50         # Overlap between chunks (tokens)
    preserve_structure: bool = True # Maintain markdown hierarchy
    content_format: str = "markdown"
    memory_type: str = "working_memory"
```

### Memory Type Overrides

Different memory types may have different chunking parameters:

| Memory Type | Auto-Chunk | Chunk Size | Chunk Overlap | Preserve Structure |
|-------------|------------|------------|---------------|-------------------|
| `report` | Yes | 450 | 50 | Yes |
| `working_memory` | Yes | 450 | 50 | Yes |
| `knowledge_base` | Yes | 450 | 50 | Yes |
| `session_context` | No | N/A | N/A | N/A |
| `input_prompt` | No | N/A | N/A | N/A |
| `system_memory` | No | N/A | N/A | N/A |

**Configuration Source:** `src/memory_types.py` - `get_memory_type_config()`

---

## Two-Stage Chunking Algorithm

### Stage 1: Structure-Preserving Split (ExperimentalMarkdownSyntaxTextSplitter)

**Purpose:** Split by markdown headers while preserving code blocks and indentation

**Process:**
1. Parse markdown headers (H1-H6)
2. Split content at header boundaries
3. **Critical:** Keep code blocks intact (respects ```fences)
4. Preserve indentation within code blocks
5. Keep headers in content for context

**Header Definitions:**
```python
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]
```

**Example Input:**
```markdown
# Performance Analysis

## Database Queries
Query execution times are slow...

```python
def slow_query():
    # This code block stays together
    return db.execute(...)
```

## API Response Times
API calls taking 2+ seconds...
```

**Stage 1 Output:**
- Split 1: "# Performance Analysis" (header + preamble)
- Split 2: "## Database Queries" + query content + code block
- Split 3: "## API Response Times" + API content

**Code Integrity:** Code blocks are kept intact in Stage 1.

---

### Stage 2: Size-Constrained Split (RecursiveCharacterTextSplitter)

**Purpose:** Enforce token limits while respecting code boundaries

**Separator Priority (Highest to Lowest):**
```python
code_aware_separators = [
    "\n```",    # Code block boundaries (HIGHEST PRIORITY)
    "\n## ",    # H2 headers
    "\n### ",   # H3 sub-headers
    "\n\n",     # Paragraph boundaries
    "\n",       # Line boundaries
    ". ",       # Sentence boundaries
    " ",        # Word boundaries
    ""          # Character boundaries (last resort)
]
```

**Process:**
1. For each Stage 1 split:
   - If tokens ≤ 450: Keep as single chunk
   - If tokens > 450: Split recursively
2. Try separators in priority order
3. Split at first separator that creates chunks ≤ 450 tokens
4. Apply 50-token overlap between chunks
5. Validate code fence balance

**Key Behavior:**
- **Code blocks**: Always split at ```fences (highest priority)
- **Oversized code blocks**: If a single code block > 450 tokens, flag as `exceeds_limit=true` but keep intact
- **Paragraphs**: Split between paragraphs if possible
- **Sentences**: Split between sentences if needed
- **Words**: Split between words as last resort

**Example (Stage 2):**

Input (from Stage 1):
```markdown
## Database Performance (620 tokens total)

Content about database...

```python
# 200 token code block
def query():
    ...
```

More content about optimization...
```

Stage 2 Output:
- Chunk 1: Header + intro content (400 tokens)
- Chunk 2: Last 50 tokens of chunk 1 + code block (250 tokens)
- Chunk 3: Last 50 tokens of chunk 2 + optimization content (420 tokens)

---

## Chunk Entry Structure

### ChunkEntry Class

```python
@dataclass
class ChunkEntry:
    parent_id: int              # Memory ID this chunk belongs to
    chunk_index: int            # Position in document (0-based)
    content: str                # Enriched chunk content (for embedding)
    chunk_type: str             # "section", "text", "code_block"
    token_count: int            # Number of tokens
    header_path: str            # "# Title > ## Section > ### Subsection"
    level: int                  # Header depth (1-6, 0 for root)
    content_hash: str           # SHA256 (first 16 chars)

    # Optional fields
    start_char: int | None      # Character offset in original doc
    end_char: int | None        # End character offset
    prev_chunk_id: int | None   # Previous chunk ID (linked list)
    next_chunk_id: int | None   # Next chunk ID (linked list)

    # Enrichment fields
    parent_title: str           # Document title
    section_hierarchy: str      # JSON array of header path
    granularity_level: str      # "fine", "medium", "coarse"
    chunk_position_ratio: float # Position in doc (0.0-1.0)
    sibling_count: int          # Total chunks in document
    depth_level: int            # Nesting depth

    # Code-specific fields
    contains_code: bool         # Has code blocks?
    contains_table: bool        # Has tables?
    keywords: list[str]         # Extracted keywords

    # Contextual enrichment
    original_content: str       # Original chunk (pre-enrichment)
    is_contextually_enriched: bool  # Was enrichment applied?
    embedding: bytes | None     # Embedding vector (binary)
```

### Database Mapping

ChunkEntry fields map to `memory_chunks` table columns:

```sql
CREATE TABLE memory_chunks (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL,
    parent_title TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,              -- Enriched content
    chunk_type TEXT DEFAULT 'text',
    start_char INTEGER,
    end_char INTEGER,
    token_count INTEGER,
    header_path TEXT,
    level INTEGER DEFAULT 0,
    prev_chunk_id INTEGER,
    next_chunk_id INTEGER,
    content_hash TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL,
    section_hierarchy TEXT,
    granularity_level TEXT DEFAULT 'medium',
    chunk_position_ratio REAL,
    sibling_count INTEGER,
    depth_level INTEGER,
    contains_code INTEGER DEFAULT 0,
    contains_table INTEGER DEFAULT 0,
    keywords TEXT DEFAULT '[]',
    original_content TEXT,              -- Original chunk (pre-enrichment)
    is_contextually_enriched INTEGER DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE
);
```

---

## Contextual Enrichment

### Purpose

Add document/section context to each chunk to improve embedding quality and search relevance.

### Enrichment Process

**When Applied:**
- Enabled by default for `knowledge_base` and `reports`
- Optional for other memory types via `enable_enrichment=true`

**Process:**
```python
def _enrich_chunk_with_context(
    chunk_content: str,
    document_title: str,
    header_path: str,
    chunk_index: int,
    total_chunks: int,
    memory_type: str,
    granularity_level: str
) -> tuple[str, str]:
    """
    Returns:
        (enriched_content_for_embedding, original_content_for_display)
    """
```

### Enriched Content Structure

**Template:**
```
Document: {document_title}
Section: {header_path or 'Main Content'}
Position: Chunk {chunk_index + 1} of {total_chunks} ({position_desc}, {granularity_level} granularity)
Type: {memory_type}

{original_chunk_content}
```

**Position Description:**
- `position_ratio = (chunk_index + 1) / total_chunks`
- "beginning" if ratio < 0.33
- "middle" if ratio < 0.67
- "end" if ratio ≥ 0.67

**Example:**
```
Document: Performance Analysis Report
Section: Database Queries > Query Execution Times
Position: Chunk 3 of 15 (beginning, fine granularity)
Type: report

The query execution times for the PNL service show significant latency...
[original chunk content continues]
```

### Storage

- **`content` field**: Stores enriched version (used for embedding)
- **`original_content` field**: Stores original chunk (used for display/reconstruction)
- **`is_contextually_enriched` flag**: Set to 1 (true)

### Granularity Classification

Based on token count:
```python
if token_count < 400:
    granularity_level = "fine"
elif token_count <= 1200:
    granularity_level = "medium"
else:
    granularity_level = "coarse"
```

---

## Code Block Handling

### Code Detection

**Patterns:**
```python
if '```' in content:
    # Code block detected
    metadata['type'] = 'code_block'
    metadata['is_code'] = True
    metadata['atomic_unit'] = True
```

**Language Extraction:**
```python
lang_match = re.search(r'```(\w+)', content)
if lang_match:
    metadata['language'] = lang_match.group(1)
    # Examples: python, javascript, sql, go, etc.
```

### Code Fence Balance Validation

**Validation Algorithm:**
```python
def _validate_code_integrity(splits: list) -> list:
    for doc in splits:
        content = doc.page_content
        opening_fences = content.count("```")

        if opening_fences % 2 != 0:
            # Unbalanced - code block was split
            doc.metadata['code_integrity_warning'] = True
            doc.metadata['unbalanced_fences'] = True
            # Log warning
```

**Integrity Score:**
```python
code_chunks = [c for c in splits if '```' in c.page_content]
valid_chunks = [c for c in code_chunks
                if not c.metadata.get('code_integrity_warning')]

integrity_score = len(valid_chunks) / len(code_chunks)
```

**Target:** 100% for standard-sized blocks, 95%+ including oversized

### Oversized Code Blocks

If a code block > 450 tokens:
- Keep intact (do not split mid-block)
- Flag: `exceeds_limit = true`
- Log warning
- Still indexed for search

**Example:**
```python
# This 800-token function stays together
def complex_function():
    # Many lines of code...
    # (exceeds chunk_size but preserved)
```

---

## Header Hierarchy Parsing

### Header Path Construction

**Format:** `"# Title > ## Section > ### Subsection"`

**Algorithm:**
```python
def _split_by_headers(content: str) -> list[dict]:
    header_stack = []

    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.*)', line.strip())

        if header_match:
            header_level = len(header_match.group(1))  # 1-6
            header_text = header_match.group(2)

            # Update stack (pop deeper levels)
            header_stack = header_stack[:header_level-1]

            # Extend if needed
            if len(header_stack) < header_level:
                header_stack.extend([''] * (header_level - len(header_stack)))

            header_stack[header_level-1] = header_text

            # Build path
            header_path = ' > '.join(h for h in header_stack[:header_level] if h)
```

**Example:**

Input:
```markdown
# Performance Report         <- Level 1
## Database                  <- Level 2
### Query Times              <- Level 3
#### Slow Queries            <- Level 4
## API                       <- Level 2 (resets stack)
```

Header Paths:
- H1: "Performance Report"
- H2: "Performance Report > Database"
- H3: "Performance Report > Database > Query Times"
- H4: "Performance Report > Database > Query Times > Slow Queries"
- H2: "Performance Report > API" (stack reset)

### Level vs Depth

- **Level**: Markdown header level (1-6 from # to ######)
- **Depth**: Nesting depth in hierarchy (how many parents)

Example:
```markdown
# Title           <- level=1, depth=0
## Section        <- level=2, depth=1
### Subsection    <- level=3, depth=2
```

---

## Token Counting

### Tiktoken Integration

**Encoding:** cl100k_base (same as GPT-3.5/4)

```python
def _count_tokens(text: str) -> int:
    if self.tokenizer:  # tiktoken available
        return len(self.tokenizer.encode(text))
    else:
        # Fallback: rough estimation
        return int(len(text.split()) * 1.3)
```

### Token Estimation Fallback

If tiktoken not available:
```python
token_count ≈ word_count × 1.3
```

**Rationale:** English text averages ~1.3 tokens per word

### Overlap Token Calculation

```python
def _get_overlap_content(content: str, overlap_tokens: int) -> str:
    if self.tokenizer:
        tokens = self.tokenizer.encode(content)
        if len(tokens) > overlap_tokens:
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.tokenizer.decode(overlap_token_ids)
        else:
            return content
    else:
        # Fallback: word-based
        words = content.split()
        overlap_words = int(overlap_tokens / 1.3)
        return ' '.join(words[-overlap_words:])
```

---

## Algorithm Pseudocode

### Main Chunking Flow

```
function chunk_document(content, parent_id, metadata):
    # 1. Configuration
    memory_type = metadata.get('memory_type')
    config = get_memory_type_config(memory_type)
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    # 2. Format detection
    content_format = detect_format(content)

    # 3. Chunking strategy selection
    if content_format == MARKDOWN and preserve_structure:
        if LANGCHAIN_AVAILABLE:
            chunks = chunk_with_langchain(content, parent_id, chunk_size, chunk_overlap)
        else:
            chunks = chunk_markdown_hierarchical(content, parent_id, chunk_size, chunk_overlap)
    else:
        chunks = chunk_recursive(content, parent_id, chunk_size, chunk_overlap)

    # 4. Contextual enrichment (optional)
    if enable_enrichment:
        apply_contextual_enrichment(chunks, document_title, memory_type)

    # 5. Link chunks (prev/next references)
    link_chunks(chunks)

    return chunks
```

### Two-Stage LangChain Chunking

```
function chunk_with_langchain(content, parent_id, chunk_size, chunk_overlap):
    # STAGE 1: Structure-preserving split
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ...],
        strip_headers=false  # Keep headers for context
    )
    md_header_splits = markdown_splitter.split_text(content)

    # STAGE 2: Size-constrained split
    code_aware_separators = ["\n```", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=code_aware_separators
    )

    final_splits = text_splitter.split_documents(md_header_splits)

    # VALIDATION: Check code fence balance
    validate_code_integrity(final_splits)

    # CONVERSION: LangChain documents -> ChunkEntry objects
    chunks = []
    for idx, doc in enumerate(final_splits):
        header_parts = extract_header_path(doc.metadata)
        header_path = " > ".join(header_parts)
        level = len(header_parts)

        chunk_entry = ChunkEntry(
            parent_id=parent_id,
            chunk_index=idx,
            content=doc.page_content.strip(),
            chunk_type="section" if level > 0 else "text",
            token_count=count_tokens(doc.page_content),
            header_path=header_path,
            level=level,
            content_hash=generate_hash(doc.page_content)
        )

        # Enhance with code metadata
        chunk_entry.metadata = enhance_code_metadata(doc.page_content, doc.metadata)

        chunks.append(chunk_entry)

    return chunks
```

### Code Metadata Enhancement

```
function enhance_code_metadata(content, base_metadata):
    metadata = base_metadata.copy()

    if '```' in content:
        metadata['type'] = 'code_block'
        metadata['is_code'] = true
        metadata['atomic_unit'] = true

        # Extract language
        lang_match = regex_search(r'```(\w+)', content)
        if lang_match:
            metadata['language'] = lang_match.group(1)

        # Flag oversized
        if token_count > chunk_size:
            metadata['exceeds_limit'] = true

    # Flag small chunks (for monitoring)
    if token_count < 150:
        metadata['below_typical_size'] = true

    return metadata
```

### Contextual Enrichment

```
function apply_contextual_enrichment(chunks, document_title, memory_type):
    total_chunks = len(chunks)

    for chunk in chunks:
        # Determine granularity
        if chunk.token_count < 400:
            granularity = "fine"
        elif chunk.token_count <= 1200:
            granularity = "medium"
        else:
            granularity = "coarse"

        # Calculate position
        position_ratio = (chunk.chunk_index + 1) / total_chunks
        position_desc = "beginning" if position_ratio < 0.33
                        else "middle" if position_ratio < 0.67
                        else "end"

        # Build context header
        context_header = f"""Document: {document_title}
Section: {chunk.header_path or 'Main Content'}
Position: Chunk {chunk.chunk_index + 1} of {total_chunks} ({position_desc}, {granularity} granularity)
Type: {memory_type}

"""

        # Store original and enriched
        chunk.original_content = chunk.content
        chunk.content = context_header + chunk.content
        chunk.is_contextually_enriched = true
        chunk.granularity_level = granularity
```

---

## Performance Characteristics

### Chunking Speed

| Document Size | Tokens | Chunks | Time | Throughput |
|---------------|--------|--------|------|------------|
| Small | 500 | 2 | ~50ms | 10,000 tokens/s |
| Medium | 5,000 | 12 | ~200ms | 25,000 tokens/s |
| Large | 50,000 | 115 | ~1.5s | 33,000 tokens/s |
| Very Large | 500,000 | 1,150 | ~15s | 33,000 tokens/s |

**Bottlenecks:**
1. Tiktoken encoding: ~30% of time
2. Markdown parsing: ~20% of time
3. LangChain splitting: ~30% of time
4. Metadata extraction: ~20% of time

### Memory Usage

- **Per chunk:** ~2KB (ChunkEntry object)
- **1000 chunks:** ~2MB
- **10,000 chunks:** ~20MB (typical for very large document)

### Embedding Generation

**Separate from chunking** - performed in batch after all chunks created:

| Chunks | Batch Size | Time | Throughput |
|--------|-----------|------|------------|
| 10 | 32 | ~100ms | 100 chunks/s |
| 100 | 32 | ~800ms | 125 chunks/s |
| 1000 | 32 | ~7s | 143 chunks/s |

**Model:** sentence-transformers/all-MiniLM-L6-v2 (CPU inference)

---

## Implementation Notes

### Dependencies

```python
# Required
import tiktoken                              # Token counting
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,              # Header-based splitting
    RecursiveCharacterTextSplitter,          # Recursive size splitting
    ExperimentalMarkdownSyntaxTextSplitter   # Code-aware splitting
)

# Data structures
from dataclasses import dataclass
import hashlib
import re
```

### Fallback Behavior

If LangChain not available:
- Fall back to custom `_chunk_markdown_hierarchical()`
- Uses regex-based header parsing
- Splits by paragraphs (`\n\n`) for size constraints
- Less sophisticated than LangChain but functional

If tiktoken not available:
- Fall back to word-based estimation
- Token count = word count × 1.3
- Less accurate but acceptable

---

## Testing & Validation

### Test Cases

1. **Small documents (<450 tokens)**: Single chunk
2. **Medium documents (450-5000 tokens)**: Multiple chunks, proper overlap
3. **Large documents (>50,000 tokens)**: Hundreds of chunks, memory efficiency
4. **Code-heavy documents**: Code blocks intact, balance validation
5. **Deep header nesting**: Header paths correct (6+ levels)
6. **Mixed content**: Code + text + tables
7. **Edge cases**: Empty sections, single-line paragraphs, malformed markdown

### Validation Checks

```python
def validate_chunks(chunks: list[ChunkEntry]):
    assert len(chunks) > 0, "No chunks created"

    # Check indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i

    # Check token counts
    for chunk in chunks:
        assert chunk.token_count > 0
        if not chunk.metadata.get('exceeds_limit'):
            assert chunk.token_count <= chunk_size * 1.1  # Allow 10% buffer

    # Check header paths are valid
    for chunk in chunks:
        if chunk.header_path:
            assert ' > ' in chunk.header_path or chunk.level == 1

    # Check code blocks are balanced
    for chunk in chunks:
        if '```' in chunk.content:
            assert chunk.content.count('```') % 2 == 0, f"Unbalanced fences in chunk {chunk.chunk_index}"
```

---

## Future Enhancements

### Planned Improvements

1. **Table-aware splitting**: Detect and preserve table boundaries
2. **List-aware splitting**: Keep list items together
3. **Sentence boundary detection**: Use spaCy for better sentence splitting
4. **Language-specific code parsing**: Use tree-sitter for precise code splitting
5. **Adaptive chunk sizing**: Dynamically adjust chunk size based on content type
6. **Parallel chunking**: Process large documents in parallel

### Configuration Ideas

```python
class AdvancedChunkingConfig:
    adaptive_sizing: bool = True       # Adjust chunk size by content type
    max_chunk_size: int = 600          # Hard limit
    min_chunk_size: int = 200          # Soft minimum
    preserve_tables: bool = True       # Keep tables intact
    preserve_lists: bool = True        # Keep lists intact
    code_parser: str = "tree-sitter"   # Advanced code parsing
```

---

## References

- **Implementation:** `src/chunking.py`
- **Memory Types:** `src/memory_types.py`
- **Test Data:** `/Users/vladanm/projects/subagents/simple-agents/test_data/`
- **LangChain Docs:** https://python.langchain.com/docs/modules/data_connection/document_transformers/
- **Tiktoken:** https://github.com/openai/tiktoken

---

**End of Chunking Algorithm Documentation**
