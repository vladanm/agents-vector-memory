# Vector Memory MCP Server - Complete Protocol Specification

**Version:** 1.0.0
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Database:** SQLite with sqlite-vec extension
**Protocol:** Model Context Protocol (MCP) over STDIO

---

## Table of Contents

1. [Overview](#overview)
2. [Storage Tools](#storage-tools)
3. [Search Tools](#search-tools)
4. [Utility Tools](#utility-tools)
5. [Return Types](#return-types)
6. [Error Handling](#error-handling)
7. [Usage Examples](#usage-examples)

---

## Overview

The Vector Memory MCP Server provides hierarchical, session-scoped memory storage with semantic vector search capabilities. It supports multiple memory types, automatic chunking of large documents, and three granularity levels for search (fine, medium, coarse).

### Key Features

- **Session-scoped memory**: All memories are organized by `session_id` and `session_iter`
- **Agent-scoped memory**: Sub-agents have isolated memory spaces via `agent_id`
- **Automatic chunking**: Large documents (>450 tokens) are automatically split into searchable chunks
- **Vector embeddings**: All chunks are embedded using sentence-transformers for semantic search
- **Multi-granularity search**: Fine (chunks), medium (sections), coarse (full documents)
- **Hierarchical structure**: Markdown headers are preserved in chunk metadata

### Memory Types

| Type | Description | Default Chunking | Typical Use |
|------|-------------|------------------|-------------|
| `session_context` | Session snapshots for continuity | No | Main agent session state |
| `input_prompt` | Original user prompts | No | Preserve user input verbatim |
| `system_memory` | System configs, commands, scripts | No | Technical system info |
| `report` | Agent analysis and findings | Yes | Sub-agent reports |
| `report_observation` | Additional notes on reports | No | Follow-up observations |
| `working_memory` | Intermediate task findings | Yes | Sub-agent work in progress |
| `knowledge_base` | Persistent cross-session knowledge | Yes | Global documentation |

---

## Storage Tools

### 1. store_session_context

**Purpose:** Store session context memory (main orchestrator only)

**Parameters:**
```python
{
  "session_id": str,          # Required: Session identifier
  "session_iter": str,        # Required: Iteration number (e.g., "v1", "v2")
  "content": str,             # Required: Context content (compressed user input)
  "task_code": str | None     # Optional: Task code for filtering
}
```

**Returns:**
```python
{
  "success": bool,
  "memory_id": int,           # Database ID of stored memory
  "memory_type": "session_context",
  "agent_id": "main-orchestrator",  # Fixed for session context
  "session_id": str,
  "content_hash": str,        # SHA256 hash (first 16 chars)
  "chunks_created": int,      # 0 (session context not chunked)
  "created_at": str,          # ISO 8601 timestamp
  "message": str,
  "error": str | None
}
```

**Validation Rules:**
- `session_id` must not be empty
- `content` must not exceed 500,000 characters
- Content hash must be unique (duplicate detection)
- Auto-chunking is disabled for this memory type

**Example:**
```python
result = store_session_context(
    session_id="analyze-code-1234",
    session_iter="v1",
    content="User requested code analysis of PNL service...",
    task_code="pnl-analysis"
)
```

---

### 2. store_input_prompt

**Purpose:** Store original user input prompt to prevent loss

**Parameters:**
```python
{
  "session_id": str,          # Required: Session identifier
  "session_iter": str,        # Required: Iteration number
  "content": str,             # Required: Verbatim user prompt
  "task_code": str | None     # Optional: Task code
}
```

**Returns:** Same structure as `store_session_context`

**Special Behavior:**
- Stored verbatim (no modification)
- No chunking applied
- `agent_id` fixed to "main-orchestrator"
- Used for audit trail and context reconstruction

---

### 3. store_system_memory

**Purpose:** Store system-level memory (configs, commands, scripts)

**Parameters:**
```python
{
  "agent_id": str,            # Required: Agent identifier
  "session_id": str,          # Required: Session identifier
  "content": str,             # Required: System information
  "session_iter": str | None, # Optional: Iteration filter
  "task_code": str | None     # Optional: Task code
}
```

**Returns:** Standard `StoreMemoryResult` structure

**Use Cases:**
- Shell commands used in tasks
- Configuration snippets
- API endpoint documentation
- System state information

---

### 4. store_report

**Purpose:** Store agent report memory (analysis findings)

**Parameters:**
```python
{
  "agent_id": str,            # Required: Sub-agent identifier
  "session_id": str,          # Required: Session identifier
  "content": str,             # Required: Report content (markdown)
  "session_iter": str | None, # Optional: Iteration
  "task_code": str | None     # Optional: Task code
}
```

**Returns:**
```python
{
  "success": bool,
  "memory_id": int,
  "memory_type": "report",
  "agent_id": str,
  "session_id": str,
  "content_hash": str,
  "chunks_created": int,      # > 0 if auto-chunked
  "created_at": str,
  "message": str,
  "error": str | None
}
```

**Auto-Chunking Behavior:**
- Reports are automatically chunked if > 450 tokens
- Chunk size: 450 tokens (target)
- Chunk overlap: 50 tokens (~11%)
- Markdown headers preserved in `header_path`
- Embeddings generated for each chunk

**Example:**
```python
result = store_report(
    agent_id="code-explorer-agent",
    session_id="analyze-code-1234",
    content=large_markdown_report,  # Will be chunked
    task_code="pnl-analysis"
)
# Result: chunks_created = 15 (if report was ~6750 tokens)
```

---

### 5. store_report_observation

**Purpose:** Store additional observations about existing reports

**Parameters:** Same as `store_report`

**Returns:** Same as `store_report`

**Typical Use:**
- Follow-up notes on previous reports
- Corrections or updates
- Additional context discovered later

---

### 6. store_working_memory

**Purpose:** Store working memory during task execution

**Parameters:** Same as `store_report`

**Returns:** Same as `store_report`

**Auto-Chunking:** Yes (same as reports)

**Typical Use:**
- Intermediate findings during analysis
- Partial results before final report
- Sub-agent "thinking out loud"

---

### 7. store_knowledge_base

**Purpose:** Store knowledge base entry (not session-scoped)

**Parameters:**
```python
{
  "agent_id": str,            # Required: Agent identifier
  "title": str,               # Required: Entry title
  "content": str,             # Required: Knowledge content
  "category": str | None      # Optional: Category (default: "general")
}
```

**Returns:** Standard `StoreMemoryResult` structure

**Special Behavior:**
- `session_id` is NULL (not session-scoped)
- Persists across sessions
- Auto-chunked if content > 450 tokens
- Title stored in metadata for retrieval

**Example:**
```python
result = store_knowledge_base(
    agent_id="code-explorer-agent",
    title="Go Language Best Practices",
    content=documentation_text,
    category="programming-languages"
)
```

---

## Search Tools

### Search Architecture Overview

The Vector Memory MCP Server provides **three granularity levels** for search:

1. **FINE (specific_chunks)**: Individual chunks (~400 tokens each)
   - Use when you need specific details or exact information
   - Returns chunk-level matches with similarity scores
   - Fastest search, most precise results

2. **MEDIUM (section_context)**: Section-level with auto-merging
   - Use for understanding context around findings
   - Returns chunks with expanded context (3 chunks before/after)
   - Auto-merges sections if ≥60% of chunks match

3. **COARSE (full_documents)**: Complete documents
   - Use for comprehensive overview or document structure
   - Returns entire memory entries (scoped search, no vector search)
   - Similarity score fixed at 2.0 (scoped match indicator)

---

### 8. search_session_context

**Purpose:** Search session context memories (main orchestrator)

**Parameters:**
```python
{
  "session_id": str,          # Required: Session identifier
  "session_iter": str | None, # Optional: Filter by iteration
  "limit": int                # Default: 5, max: 100
}
```

**Returns:**
```python
{
  "success": bool,
  "results": [
    {
      "id": int,
      "memory_type": "session_context",
      "agent_id": "main-orchestrator",
      "session_id": str,
      "session_iter": int,
      "task_code": str | None,
      "content": str,
      "title": str | None,
      "description": str | None,
      "tags": list[str],
      "metadata": dict,
      "content_hash": str,
      "created_at": str,
      "updated_at": str,
      "accessed_at": str,
      "access_count": int,
      "similarity": 2.0,        # Scoped match (not vector search)
      "source_type": "scoped"
    }
  ],
  "total_results": int,
  "query": None,
  "filters": {...},
  "limit": int,
  "latest_first": bool,
  "error": str | None,
  "message": str | None
}
```

**Ordering:**
- `session_iter DESC, created_at DESC` (newest first)
- Use `latest_first=false` to reverse order

---

### 9. search_input_prompts

**Purpose:** Search input prompt memories

**Parameters:** Same as `search_session_context`

**Returns:** Same structure as `search_session_context`

**Use Cases:**
- Retrieve original user prompts
- Audit trail reconstruction
- Context for continuation tasks

---

### 10. search_system_memory

**Purpose:** Semantic search across system memory (configs, commands)

**Parameters:**
```python
{
  "query": str,               # Required: Search query
  "agent_id": str | None,     # Optional: Filter by agent
  "session_id": str | None,   # Optional: Filter by session
  "session_iter": str | None, # Optional: Filter by iteration
  "task_code": str | None,    # Optional: Filter by task
  "limit": int                # Default: 10
}
```

**Returns:** Same structure as `search_session_context` (with similarity scores)

**Search Behavior:**
- If `query` is None: scoped search (no vector search)
- If `query` provided: vector similarity search
- Filters applied after vector search (post-filtering)

---

### 11-13. search_reports_* (Three Granularities)

#### search_reports_specific_chunks (FINE)

**Purpose:** Search reports at chunk level for specific details

**Parameters:**
```python
{
  "query": str,               # Required: Search query
  "agent_id": str | None,     # Optional: Filter by agent
  "session_id": str | None,   # Optional: Filter by session
  "session_iter": str | None, # Optional: Filter by iteration
  "task_code": str | None,    # Optional: Filter by task
  "limit": int                # Default: 10
}
```

**Returns:**
```python
{
  "success": bool,
  "results": [
    {
      "chunk_id": int,
      "memory_id": int,
      "chunk_index": int,
      "chunk_content": str,   # ~400 tokens
      "chunk_type": str,      # "section", "text", "code_block"
      "header_path": str,     # "# Title > ## Section"
      "level": int,           # Header depth (1-6)
      "similarity": float,    # 0.0-1.0 (cosine similarity)
      "source": "chunk",
      "granularity": "fine"
    }
  ],
  "total_results": int,
  "granularity": "fine",
  "message": str | None,
  "error": str | None
}
```

**Similarity Score Calculation:**
```python
similarity = 1.0 - (l2_distance² / 2.0)
```
Where `l2_distance` is the L2 distance from sqlite-vec.

**Search Algorithm (FINE):**
1. Generate query embedding (384 dimensions)
2. Convert to bytes for sqlite-vec
3. Iterative vector search:
   - Start with batch_size = 100
   - Fetch candidates from `vec_chunk_search`
   - Apply metadata filters (agent_id, session_id, etc.)
   - Continue fetching until `limit` results found or max_offset reached
   - Adaptive batch growth if selectivity < 5%
4. Sort by L2 distance
5. Convert distance to similarity score
6. Return top `limit` results

---

#### search_reports_section_context (MEDIUM)

**Purpose:** Search reports at section level with expanded context

**Parameters:** Same as `search_reports_specific_chunks` (limit default: 5)

**Returns:**
```python
{
  "success": bool,
  "results": [
    {
      "memory_id": int,
      "section_header": str,  # "# Title > ## Section"
      "section_content": str, # Full section content (all chunks)
      "header_path": str,
      "chunks_in_section": int,
      "matched_chunks": int,
      "match_ratio": float,   # matched / total
      "auto_merged": bool,    # True if match_ratio ≥ 0.6
      "similarity": float,    # Average of matching chunks
      "source": "expanded_section",
      "granularity": "medium"
    }
  ],
  "total_results": int,
  "granularity": "medium",
  "message": str | None,
  "error": str | None
}
```

**Search Algorithm (MEDIUM):**
1. Perform FINE search with `limit * 5` to get matching chunks
2. Group chunks by section (using `header_path` parent)
3. For each section:
   - **Section Key Extraction (BUG FIX):**
     - Extract first TWO parts of `header_path` (H1 > H2)
     - Old behavior: Only H1 (root level)
     - New behavior: "# Title > ## Section" (H2 level)
   - Fetch all chunks in that section
   - Calculate match_ratio = matched_chunks / total_chunks
   - Build `section_content` by joining all chunks
   - Calculate average similarity of matching chunks
   - Flag `auto_merged = true` if match_ratio ≥ 0.6
4. Sort sections by average similarity
5. Return top `limit` sections

**Example:**
```markdown
# Performance Analysis Report          <- H1
## Database Query Performance          <- H2 (section boundary)
### Query Execution Times               <- H3 (subsection)
Content about query times...
### Index Usage                         <- H3 (subsection)
Content about indexes...
## API Response Times                   <- H2 (new section)
Content about API...
```

In MEDIUM granularity:
- Section 1: "Performance Analysis Report > Database Query Performance"
- Section 2: "Performance Analysis Report > API Response Times"

---

#### search_reports_full_documents (COARSE)

**Purpose:** Search reports returning complete documents

**Parameters:** Same as `search_reports_specific_chunks` (limit default: 3)

**Returns:** Same structure as `search_session_context` (full memory entries)

**Search Behavior:**
- No vector search (scoped lookup)
- Filters: memory_type, agent_id, session_id, session_iter, task_code
- Similarity score = 2.0 (indicates scoped match)
- Returns complete `content` field (not chunks)
- Ordering: `session_iter DESC, created_at DESC`

**Use Cases:**
- Get full report for comprehensive reading
- Export reports to files
- When document structure matters (headers, formatting)

---

### 14-16. search_working_memory_* (Three Granularities)

Same structure and behavior as `search_reports_*`, but filters for `memory_type = "working_memory"`.

**Tools:**
- `search_working_memory_specific_chunks` (FINE)
- `search_working_memory_section_context` (MEDIUM)
- `search_working_memory_full_documents` (COARSE)

---

### 17-19. search_knowledge_base_* (Three Granularities)

Same structure and behavior as `search_reports_*`, but:
- Filters for `memory_type = "knowledge_base"`
- No `session_id` or `session_iter` filters (not session-scoped)
- Can filter by `category` (optional parameter)

**Tools:**
- `search_knowledge_base_specific_chunks` (FINE)
- `search_knowledge_base_section_context` (MEDIUM)
- `search_knowledge_base_full_documents` (COARSE)

---

## Utility Tools

### 20. load_session_context_for_task

**Purpose:** Load all relevant session context for task continuation

**Parameters:**
```python
{
  "session_id": str,          # Required: Session identifier
  "session_iter": str         # Required: Iteration number
}
```

**Returns:**
```python
{
  "success": bool,
  "session_context": {...},   # Latest session_context memory
  "input_prompts": [...],     # All input prompts for session
  "recent_reports": [...],    # Recent reports (last 5)
  "recent_working_memory": [...],  # Recent working memory (last 5)
  "message": str,
  "error": str | None
}
```

**Use Cases:**
- Sub-agent continuation after restart
- Loading context for parallel tasks
- Debugging session state

---

### 21. expand_chunk_context

**Purpose:** Expand context around a specific chunk

**Parameters:**
```python
{
  "chunk_id": str,            # Required: Chunk identifier
  "surrounding_chunks": int   # Default: 2 (chunks before/after)
}
```

**Returns:**
```python
{
  "success": bool,
  "memory_id": int,
  "target_chunk_index": int,
  "context_window": int,
  "chunks_returned": int,
  "expanded_content": str,    # Joined content of all chunks
  "chunks": [
    {
      "chunk_id": int,
      "chunk_index": int,
      "content": str,
      "chunk_type": str,
      "header_path": str,
      "level": int
    }
  ],
  "error": str | None,
  "message": str | None
}
```

**Example:**
```python
# Get chunk 10 with 2 chunks before and after
result = expand_chunk_context(
    chunk_id="12345",
    surrounding_chunks=2
)
# Returns chunks [8, 9, 10, 11, 12] joined into expanded_content
```

---

### 22. reconstruct_document

**Purpose:** Reconstruct full document from memory_id

**Parameters:**
```python
{
  "memory_id": str            # Required: Memory identifier
}
```

**Returns:**
```python
{
  "success": bool,
  "memory_id": int,
  "content": str,             # Full reconstructed content
  "title": str,
  "memory_type": str,
  "chunk_count": int,
  "message": str,
  "error": str | None
}
```

**Reconstruction Algorithm:**
1. Fetch all chunks for `memory_id` ordered by `chunk_index`
2. For each chunk:
   - Use `original_content` if available (pre-enrichment content)
   - Fallback to `content` for backward compatibility
3. Join chunks with `\n\n` separator
4. Return reconstructed document

**Use Cases:**
- Export full report to file
- Display complete document in UI
- Verify chunking integrity

---

### 23. get_memory_by_id

**Purpose:** Get specific memory by ID

**Parameters:**
```python
{
  "memory_id": str            # Required: Memory identifier
}
```

**Returns:**
```python
{
  "success": bool,
  "memory": {
    "id": int,
    "memory_type": str,
    "agent_id": str,
    "session_id": str,
    "session_iter": int,
    "task_code": str | None,
    "content": str,
    "title": str | None,
    "description": str | None,
    "tags": list[str],
    "metadata": dict,
    "content_hash": str,
    "created_at": str,
    "updated_at": str,
    "accessed_at": str,
    "access_count": int
  },
  "error": str | None,
  "message": str | None
}
```

---

### 24. get_session_stats

**Purpose:** Get statistics for a session

**Parameters:**
```python
{
  "session_id": str           # Required: Session identifier
}
```

**Returns:**
```python
{
  "success": bool,
  "session_id": str,
  "memory_counts": {
    "session_context": int,
    "input_prompt": int,
    "report": int,
    "working_memory": int,
    "system_memory": int,
    ...
  },
  "agent_counts": {
    "main-orchestrator": int,
    "code-explorer-agent": int,
    ...
  },
  "total_memories": int,
  "total_chunks": int,
  "earliest_created": str,
  "latest_created": str,
  "error": str | None,
  "message": str | None
}
```

---

### 25. list_sessions

**Purpose:** List recent sessions with activity counts

**Parameters:**
```python
{
  "limit": int,               # Default: 20
  "agent_id": str | None      # Optional: Filter by agent
}
```

**Returns:**
```python
{
  "success": bool,
  "sessions": [
    {
      "session_id": str,
      "agent_id": str,
      "memory_count": int,
      "latest_activity": str,
      "memory_types": {...}   # Breakdown by type
    }
  ],
  "total_sessions": int,
  "error": str | None,
  "message": str | None
}
```

**Ordering:** Most recent activity first

---

### 26. write_document_to_file

**Purpose:** Write reconstructed document to file

**Parameters:**
```python
{
  "memory_id": str,           # Required: Memory identifier
  "output_path": str          # Required: File path
}
```

**Returns:**
```python
{
  "success": bool,
  "file_path": str,           # Absolute path to written file
  "memory_id": int,
  "bytes_written": int,
  "message": str,
  "error": str | None
}
```

**File Format:**
- Markdown with YAML frontmatter (if PyYAML available)
- Includes metadata: memory_id, title, memory_type, chunk_count
- UTF-8 encoding

---

### 27. delete_memory

**Purpose:** Delete memory and all chunks

**Parameters:**
```python
{
  "memory_id": str            # Required: Memory identifier
}
```

**Returns:**
```python
{
  "success": bool,
  "memory_id": int,
  "error": str | None,
  "message": str
}
```

**Cascade Behavior:**
- Deletes from `session_memories`
- Deletes all chunks from `memory_chunks` (foreign key cascade)
- Deletes embeddings from `vec_chunk_search` (foreign key cascade)

**Warning:** This cannot be undone!

---

### 28. cleanup_old_memories

**Purpose:** Clean up memories older than specified days

**Parameters:**
```python
{
  "days_old": int,            # Default: 90
  "dry_run": bool             # Default: true
}
```

**Returns:**
```python
{
  "success": bool,
  "memories_deleted": int,
  "chunks_deleted": int,
  "dry_run": bool,
  "message": str,
  "error": str | None
}
```

**Safety:**
- Always defaults to `dry_run=true`
- Must explicitly set `dry_run=false` to delete
- Deletes memories where `created_at < now() - days_old`

---

## Return Types

### StoreMemoryResult

```python
{
  "success": bool,
  "memory_id": int | None,
  "memory_type": str | None,
  "agent_id": str | None,
  "session_id": str | None,
  "content_hash": str | None,
  "chunks_created": int | None,
  "created_at": str | None,  # ISO 8601
  "message": str,
  "error": str | None
}
```

### SearchMemoriesResult

```python
{
  "success": bool,
  "results": list[dict],      # See specific search tool for structure
  "total_results": int,
  "query": str | None,
  "filters": dict,
  "limit": int,
  "latest_first": bool,
  "error": str | None,
  "message": str | None
}
```

### GranularSearchResult

```python
{
  "success": bool,
  "results": list[dict],      # Structure depends on granularity
  "total_results": int,
  "granularity": "fine" | "medium" | "coarse",
  "message": str | None,
  "error": str | None
}
```

---

## Error Handling

### Error Codes

| Error Type | HTTP Equivalent | Description |
|------------|----------------|-------------|
| ValidationError | 400 | Invalid parameters |
| MemoryError | 500 | Storage failure |
| SearchError | 500 | Search failure |
| ChunkingError | 500 | Chunking failure |
| DatabaseError | 500 | Database operation failure |
| DatabaseLockError | 503 | Database locked (retry) |

### Error Response Structure

```python
{
  "success": false,
  "error": "ERROR_TYPE",
  "message": "Human-readable error description",
  ... # Other fields set to None or default values
}
```

### Retry Logic

- **Database locks:** Exponential backoff (30s timeout)
- **Embedding model load:** One retry (first call may take 5-7s)
- **Vector search:** No retry (fail fast)

---

## Usage Examples

### Example 1: Store and Search Reports

```python
# Store a report from sub-agent
result = store_report(
    agent_id="code-explorer-agent",
    session_id="analyze-code-1234",
    content=report_markdown,
    task_code="pnl-analysis"
)

print(f"Stored memory_id: {result['memory_id']}")
print(f"Chunks created: {result['chunks_created']}")

# Search for specific information (FINE granularity)
search_result = search_reports_specific_chunks(
    query="database connection issues",
    agent_id="code-explorer-agent",
    session_id="analyze-code-1234",
    limit=5
)

for chunk in search_result['results']:
    print(f"Similarity: {chunk['similarity']:.3f}")
    print(f"Section: {chunk['header_path']}")
    print(f"Content: {chunk['chunk_content'][:200]}...")
```

### Example 2: Multi-Granularity Search

```python
# Fine: Find specific mentions
fine_results = search_reports_specific_chunks(
    query="performance bottleneck",
    limit=10
)

# Medium: Get context around findings
medium_results = search_reports_section_context(
    query="performance bottleneck",
    limit=5
)

# Coarse: Get full reports
coarse_results = search_reports_full_documents(
    query="performance bottleneck",
    limit=3
)

# Compare results
print(f"Fine: {len(fine_results['results'])} chunks")
print(f"Medium: {len(medium_results['results'])} sections")
print(f"Coarse: {len(coarse_results['results'])} documents")
```

### Example 3: Session Context Management

```python
# Store session context
store_session_context(
    session_id="session-123",
    session_iter="v1",
    content="User requested analysis of PNL service..."
)

# Later: Load full context for continuation
context = load_session_context_for_task(
    session_id="session-123",
    session_iter="v1"
)

print(f"Session context: {context['session_context']['content']}")
print(f"Input prompts: {len(context['input_prompts'])}")
print(f"Recent reports: {len(context['recent_reports'])}")
```

### Example 4: Document Reconstruction

```python
# Search for a report
results = search_reports_full_documents(
    query="performance analysis",
    limit=1
)

memory_id = results['results'][0]['id']

# Reconstruct full document
doc = reconstruct_document(memory_id=memory_id)

print(f"Title: {doc['title']}")
print(f"Chunks: {doc['chunk_count']}")
print(f"Content length: {len(doc['content'])} chars")

# Write to file
write_result = write_document_to_file(
    memory_id=memory_id,
    output_path="/path/to/report.md"
)

print(f"Written to: {write_result['file_path']}")
```

---

## Performance Characteristics

### Storage Performance

- **Without chunking:** <100ms (single DB insert)
- **With chunking:** 1-3s (includes embedding generation)
  - Chunking: ~100ms
  - Embedding generation: ~1-2s (batch processing)
  - DB inserts: ~100-500ms (depends on chunk count)

### Search Performance

| Granularity | Typical Latency | Description |
|-------------|----------------|-------------|
| FINE | 200-500ms | Vector search + metadata filtering |
| MEDIUM | 500-1500ms | FINE search + section grouping |
| COARSE | 50-200ms | Scoped search (no vector ops) |

**Slow Query Threshold:** 2.0 seconds (logged as warning)

### Embedding Model Warmup

- **First call:** 5-7 seconds (model loading)
- **Subsequent calls:** <1 second
- **Solution:** Pre-warm at server startup (WARM_START_EMBEDDING_MODEL=true)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `EMBEDDING_DIM` | 384 | Embedding dimensions |
| `DEFAULT_CHUNK_SIZE` | 450 | Target chunk size (tokens) |
| `DEFAULT_CHUNK_OVERLAP` | 50 | Chunk overlap (tokens) |
| `VECTOR_SEARCH_BATCH_SIZE` | 100 | Initial batch for iterative search |
| `LOG_SLOW_QUERY_THRESHOLD` | 2.0 | Slow query threshold (seconds) |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `USE_ITERATIVE_FETCHING` | true | Task 2: Iterative post-filter fetching |
| `USE_THRESHOLD_FILTERING` | false | Task 1: Hard threshold filtering (DISABLED) |
| `AUTO_BACKFILL_THRESHOLD` | 1000 | Auto-backfill embeddings if < N missing |
| `WARM_START_EMBEDDING_MODEL` | true | Pre-load model at startup |

---

## Version History

- **1.0.0** (Current)
  - Initial release
  - Three granularity search
  - Medium granularity H2 section fix
  - Iterative post-filter fetching
  - Auto-backfill embeddings
  - Performance monitoring

---

## References

- **Codebase:** `/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/`
- **Main Server:** `main.py`
- **Core Logic:** `src/session_memory_store.py`
- **Chunking:** `src/chunking.py`
- **Database Schema:** `src/db_migrations.py`
- **Configuration:** `src/config.py`

---

**End of Specification**
