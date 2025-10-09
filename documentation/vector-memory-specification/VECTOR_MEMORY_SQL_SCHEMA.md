# Vector Memory SQL Schema

**Version:** 1.0.0
**Database:** SQLite 3.x with WAL mode
**Extensions:** sqlite-vec (vec0 virtual tables)
**Implementation:** `src/db_migrations.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Core Tables](#core-tables)
3. [Vector Search Tables](#vector-search-tables)
4. [Indexes](#indexes)
5. [Foreign Key Relationships](#foreign-key-relationships)
6. [Pragmas and Configuration](#pragmas-and-configuration)
7. [Sample Queries](#sample-queries)
8. [Schema Migration](#schema-migration)

---

## Overview

### Database Architecture

```
┌─────────────────────┐
│ session_memories    │ (Main memory table)
│ - id                │
│ - content           │
│ - memory_type       │
│ - agent_id          │
│ - session_id        │
└──────┬──────────────┘
       │
       │ 1:N
       ▼
┌─────────────────────┐
│ memory_chunks       │ (Chunked content)
│ - id                │
│ - parent_id (FK)    │
│ - content           │
│ - header_path       │
└──────┬──────────────┘
       │
       │ 1:1
       ▼
┌─────────────────────┐
│ vec_chunk_search    │ (Vector index)
│ - chunk_id (FK)     │
│ - embedding[384]    │
└─────────────────────┘
```

### Design Principles

1. **Session-scoped storage**: All memories belong to sessions
2. **Hierarchical chunking**: Large documents split into searchable chunks
3. **Vector embeddings**: All chunks have 384-d embeddings for semantic search
4. **Referential integrity**: Foreign keys with CASCADE delete
5. **WAL mode**: Better concurrency for read-heavy workloads

---

## Core Tables

### session_memories

**Purpose:** Main storage for memory entries (documents, reports, context)

```sql
CREATE TABLE session_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT NOT NULL,           -- Type: report, working_memory, etc.
    agent_id TEXT NOT NULL,              -- Agent identifier
    session_id TEXT NOT NULL,            -- Session identifier
    session_iter INTEGER DEFAULT 1,      -- Iteration number (v1, v2, etc.)
    task_code TEXT,                      -- Optional task code for filtering
    content TEXT NOT NULL,               -- Full document content
    original_content TEXT,               -- Original before enrichment
    title TEXT,                          -- Document title
    description TEXT,                    -- Document description
    tags TEXT NOT NULL DEFAULT '[]',     -- JSON array of tags
    metadata TEXT DEFAULT '{}',          -- JSON object for arbitrary metadata
    content_hash TEXT UNIQUE NOT NULL,   -- SHA256 hash (first 16 chars)
    embedding BLOB,                      -- Document-level embedding (optional)
    created_at TEXT NOT NULL,            -- ISO 8601 timestamp
    updated_at TEXT NOT NULL,            -- ISO 8601 timestamp
    accessed_at TEXT,                    -- Last access timestamp
    access_count INTEGER DEFAULT 0,      -- Access counter
    auto_chunk INTEGER DEFAULT 0,        -- Flag: was auto-chunked?
    chunk_count INTEGER DEFAULT 0,       -- Number of chunks created
    auto_chunked INTEGER DEFAULT 0       -- Deprecated: use auto_chunk
);
```

**Key Columns:**
- `content_hash`: Prevents duplicate storage (UNIQUE constraint)
- `session_iter`: Allows multiple iterations per session (v1, v2, v3)
- `tags` and `metadata`: JSON fields for flexible schema extension
- `auto_chunk`: Indicates if chunking was applied (1=yes, 0=no)

**Data Types:**
- `TEXT`: UTF-8 strings (no length limit in SQLite)
- `INTEGER`: 64-bit signed integers
- `BLOB`: Binary data (embeddings)

---

### memory_chunks

**Purpose:** Chunked content for large documents with hierarchical metadata

```sql
CREATE TABLE memory_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL,          -- FK to session_memories.id
    parent_title TEXT,                   -- Parent document title
    chunk_index INTEGER NOT NULL,        -- Position in document (0-based)
    content TEXT NOT NULL,               -- Chunk content (enriched)
    chunk_type TEXT DEFAULT 'text',      -- Type: section, text, code_block
    start_char INTEGER,                  -- Character offset in original
    end_char INTEGER,                    -- End character offset
    token_count INTEGER,                 -- Number of tokens (tiktoken)
    header_path TEXT,                    -- "# Title > ## Section > ### Subsection"
    level INTEGER DEFAULT 0,             -- Header depth (0-6)
    prev_chunk_id INTEGER,               -- Previous chunk (linked list)
    next_chunk_id INTEGER,               -- Next chunk (linked list)
    content_hash TEXT NOT NULL,          -- SHA256 hash (first 16 chars)
    embedding BLOB,                      -- Embedding bytes (deprecated)
    created_at TEXT NOT NULL,            -- ISO 8601 timestamp

    -- Enrichment metadata
    section_hierarchy TEXT,              -- JSON array of header path
    granularity_level TEXT DEFAULT 'medium',  -- fine, medium, coarse
    chunk_position_ratio REAL,           -- Position in doc (0.0-1.0)
    sibling_count INTEGER,               -- Total chunks in document
    depth_level INTEGER,                 -- Nesting depth

    -- Content-specific flags
    contains_code INTEGER DEFAULT 0,     -- Has code blocks? (boolean)
    contains_table INTEGER DEFAULT 0,    -- Has tables? (boolean)
    keywords TEXT DEFAULT '[]',          -- JSON array of extracted keywords

    -- Contextual enrichment
    original_content TEXT,               -- Original chunk (pre-enrichment)
    is_contextually_enriched INTEGER DEFAULT 0,  -- Was enrichment applied?

    FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE,
    UNIQUE(parent_id, chunk_index)       -- Prevents duplicate chunk indices
);
```

**Key Columns:**
- `parent_id`: Links to parent document (CASCADE delete)
- `chunk_index`: Sequential position (0, 1, 2, ...)
- `header_path`: Full header hierarchy for section search
- `original_content`: Stores pre-enrichment content for reconstruction
- `content`: Stores enriched content (with context header) for embedding

**Linked List:**
- `prev_chunk_id` and `next_chunk_id` form doubly-linked list
- Allows efficient navigation between adjacent chunks

---

## Vector Search Tables

### vec_chunk_search (vec0 virtual table)

**Purpose:** Vector index for chunk embeddings (sqlite-vec extension)

```sql
CREATE VIRTUAL TABLE vec_chunk_search
USING vec0(
    chunk_id INTEGER PRIMARY KEY,        -- FK to memory_chunks.id
    embedding float[384]                 -- 384-dimensional vector (all-MiniLM-L6-v2)
);
```

**Key Concepts:**
- **Virtual table:** Not a real table (managed by sqlite-vec extension)
- **vec0:** sqlite-vec vector search interface
- **float[384]:** Fixed-size vector (384 dimensions)
- **MATCH operator:** Performs k-NN search

**Insert Example:**
```sql
INSERT INTO vec_chunk_search (chunk_id, embedding)
VALUES (12345, ?);  -- Binding: embedding.tobytes()
```

**Search Example:**
```sql
SELECT chunk_id, distance
FROM vec_chunk_search
WHERE embedding MATCH ?      -- Binding: query_embedding.tobytes()
    AND k = 100              -- Return top 100 nearest neighbors
ORDER BY distance
LIMIT 10;
```

**Distance Metric:** L2 (Euclidean) distance

**Performance:**
- Approximate nearest neighbor search (not exact)
- Typical latency: 50-200ms for k=100
- Memory usage: ~1.5KB per embedding (384 × 4 bytes)

---

### vec_session_search (vec0 virtual table)

**Purpose:** Vector index for document-level embeddings (future use)

```sql
CREATE VIRTUAL TABLE vec_session_search
USING vec0(
    memory_id INTEGER PRIMARY KEY,       -- FK to session_memories.id
    embedding float[384]                 -- Document-level embedding
);
```

**Current Usage:** Limited (chunk-level search preferred)

**Future Use Cases:**
- Document similarity search
- Duplicate detection
- Document clustering

---

## Indexes

### Primary Indexes

**Session lookup:**
```sql
CREATE INDEX idx_agent_session
ON session_memories(agent_id, session_id);
```

**Session with iteration:**
```sql
CREATE INDEX idx_agent_session_iter
ON session_memories(agent_id, session_id, session_iter);
```

**Session with task code:**
```sql
CREATE INDEX idx_agent_session_task
ON session_memories(agent_id, session_id, task_code);
```

**Memory type:**
```sql
CREATE INDEX idx_memory_type
ON session_memories(memory_type);
```

**Temporal ordering:**
```sql
CREATE INDEX idx_created_at
ON session_memories(created_at);
```

**Session iteration:**
```sql
CREATE INDEX idx_session_iter
ON session_memories(session_iter);
```

**Chunk parent lookup:**
```sql
CREATE INDEX idx_chunk_parent
ON memory_chunks(parent_id);
```

### Index Usage

**Query:** Search reports for specific agent + session
```sql
-- Uses: idx_agent_session
SELECT * FROM session_memories
WHERE agent_id = 'code-explorer-agent'
    AND session_id = 'session-123'
    AND memory_type = 'report'
ORDER BY session_iter DESC, created_at DESC;
```

**Query:** Get all chunks for document
```sql
-- Uses: idx_chunk_parent
SELECT * FROM memory_chunks
WHERE parent_id = 567
ORDER BY chunk_index;
```

---

## Foreign Key Relationships

### Cascading Deletes

```
session_memories (id)
    ↓ ON DELETE CASCADE
memory_chunks (parent_id)
    ↓ ON DELETE CASCADE
vec_chunk_search (chunk_id)
```

**Example:**
```sql
-- Delete memory entry
DELETE FROM session_memories WHERE id = 567;

-- Automatically deletes:
-- - All chunks in memory_chunks where parent_id = 567
-- - All embeddings in vec_chunk_search for those chunks
```

### Referential Integrity

**Enable foreign keys:**
```sql
PRAGMA foreign_keys = ON;
```

**Verify constraints:**
```sql
PRAGMA foreign_key_check;
```

---

## Pragmas and Configuration

### Database Configuration

```sql
-- Enable WAL mode (write-ahead logging) for better concurrency
PRAGMA journal_mode=WAL;

-- Set synchronous mode to NORMAL (faster writes, safe with WAL)
PRAGMA synchronous=NORMAL;

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Set busy timeout (30 seconds) for lock contention
PRAGMA busy_timeout=30000;
```

### WAL Mode Benefits

- **Concurrent reads:** Readers don't block writers
- **Better performance:** Writes are faster
- **Atomic commits:** All-or-nothing transactions
- **Checkpoint control:** Manual or automatic

### File System

With WAL mode, three files are created:
- `database.db` - Main database
- `database.db-wal` - Write-ahead log
- `database.db-shm` - Shared memory file

---

## Sample Queries

### Store Memory with Chunks

```sql
-- 1. Insert parent memory
INSERT INTO session_memories
    (memory_type, agent_id, session_id, session_iter, content,
     title, tags, metadata, content_hash, created_at, updated_at, accessed_at)
VALUES
    ('report', 'code-explorer-agent', 'session-123', 1, 'Full report content...',
     'Performance Analysis', '["performance", "database"]', '{"confidence": 0.9}',
     'a1b2c3d4e5f6...', '2025-01-10T15:00:00Z', '2025-01-10T15:00:00Z', '2025-01-10T15:00:00Z');

-- Get memory_id
SELECT last_insert_rowid();  -- Returns: 567

-- 2. Insert chunks
INSERT INTO memory_chunks
    (parent_id, chunk_index, content, chunk_type, token_count,
     header_path, level, content_hash, created_at, original_content, is_contextually_enriched)
VALUES
    (567, 0, 'Enriched chunk 0...', 'section', 380, '# Performance > ## Database', 2,
     'abc123...', '2025-01-10T15:00:00Z', 'Original chunk 0...', 1),
    (567, 1, 'Enriched chunk 1...', 'section', 420, '# Performance > ## API', 2,
     'def456...', '2025-01-10T15:00:00Z', 'Original chunk 1...', 1);

-- 3. Insert embeddings
INSERT INTO vec_chunk_search (chunk_id, embedding)
VALUES
    (10001, ?),  -- Binding: chunk_0_embedding.tobytes()
    (10002, ?);  -- Binding: chunk_1_embedding.tobytes()
```

### Search Reports (COARSE)

```sql
SELECT *
FROM session_memories
WHERE memory_type = 'report'
    AND agent_id = 'code-explorer-agent'
    AND session_id = 'session-123'
ORDER BY session_iter DESC, created_at DESC
LIMIT 10;
```

### Search Chunks (FINE)

```sql
SELECT
    vc.chunk_id,
    mc.parent_id,
    mc.chunk_index,
    mc.content,
    mc.header_path,
    mc.level,
    vc.distance
FROM vec_chunk_search vc
JOIN memory_chunks mc ON vc.chunk_id = mc.id
JOIN session_memories m ON mc.parent_id = m.id
WHERE vc.embedding MATCH ?  -- Query embedding
    AND k = 100
    AND m.memory_type = 'report'
    AND m.agent_id = 'code-explorer-agent'
ORDER BY vc.distance
LIMIT 10;
```

### Reconstruct Document

```sql
SELECT
    mc.chunk_index,
    mc.original_content,
    mc.content
FROM memory_chunks mc
WHERE mc.parent_id = 567
ORDER BY mc.chunk_index;
```

### Get Section Chunks (MEDIUM)

```sql
SELECT
    id,
    chunk_index,
    content,
    header_path
FROM memory_chunks
WHERE parent_id = 567
    AND (header_path = '# Performance > ## Database'
         OR header_path LIKE '# Performance > ## Database >%')
ORDER BY chunk_index;
```

### Session Statistics

```sql
SELECT
    m.memory_type,
    COUNT(*) as count,
    SUM(m.chunk_count) as total_chunks,
    AVG(m.chunk_count) as avg_chunks_per_memory
FROM session_memories m
WHERE m.session_id = 'session-123'
GROUP BY m.memory_type;
```

### Find Memories Without Chunks

```sql
SELECT
    m.id,
    m.title,
    m.memory_type,
    m.auto_chunk
FROM session_memories m
LEFT JOIN memory_chunks mc ON m.id = mc.parent_id
WHERE m.auto_chunk = 1
    AND mc.id IS NULL;
```

### Check Chunk Embedding Coverage

```sql
SELECT
    COUNT(DISTINCT mc.id) as total_chunks,
    COUNT(DISTINCT vc.chunk_id) as chunks_with_embeddings,
    (COUNT(DISTINCT mc.id) - COUNT(DISTINCT vc.chunk_id)) as missing_embeddings
FROM memory_chunks mc
LEFT JOIN vec_chunk_search vc ON mc.id = vc.chunk_id;
```

---

## Schema Migration

### Migration Process

**Implementation:** `src/db_migrations.py::run_migrations()`

**Steps:**
1. Check existing schema (PRAGMA table_info)
2. Add missing columns (ALTER TABLE ADD COLUMN)
3. Create missing tables
4. Create missing indexes
5. Load sqlite-vec extension
6. Create vec0 virtual tables

### Example: Add Missing Column

```python
# Check if column exists
cursor = conn.execute("PRAGMA table_info(session_memories)")
existing_columns = {row[1] for row in cursor.fetchall()}

# Add if missing
if 'original_content' not in existing_columns:
    conn.execute("ALTER TABLE session_memories ADD COLUMN original_content TEXT")
```

### Version History

**v1.0.0 (Current):**
- Initial schema
- Chunking support
- Vector search tables
- Contextual enrichment fields

**Future Migrations:**
- Table-aware chunking metadata
- List detection fields
- Code parsing metadata (tree-sitter)

---

## Database Statistics

### Typical Sizes

| Component | Small Session | Medium Session | Large Session |
|-----------|---------------|----------------|---------------|
| Memories | 10 | 100 | 1000 |
| Chunks | 50 | 1000 | 10,000 |
| Embeddings | 50 | 1000 | 10,000 |
| DB Size | ~1 MB | ~10 MB | ~100 MB |
| Embedding Size | ~75 KB | ~1.5 MB | ~15 MB |

**Embedding Calculation:**
- Per embedding: 384 × 4 bytes = 1,536 bytes
- 10,000 embeddings: 15,360,000 bytes ≈ 15 MB

### Growth Projections

**Assumptions:**
- Average document: 5,000 tokens
- Average chunks per document: 12
- Sessions per day: 100
- Retention: 90 days

**Estimated Growth:**
- Documents: 100 × 90 = 9,000
- Chunks: 9,000 × 12 = 108,000
- Database size: ~900 MB
- Embeddings: ~162 MB

---

## Optimization Tips

### Query Performance

1. **Use EXPLAIN QUERY PLAN:**
```sql
EXPLAIN QUERY PLAN
SELECT * FROM session_memories
WHERE agent_id = ? AND session_id = ?;
```

2. **Covering indexes (future):**
```sql
-- Include frequently accessed columns in index
CREATE INDEX idx_agent_session_covering
ON session_memories(agent_id, session_id, memory_type, created_at);
```

3. **Partition by date (future):**
```sql
-- Separate tables for old data
CREATE TABLE session_memories_archive AS
SELECT * FROM session_memories
WHERE created_at < '2024-01-01';
```

### Storage Optimization

1. **Vacuum regularly:**
```sql
VACUUM;  -- Reclaim space after deletes
```

2. **Checkpoint WAL:**
```sql
PRAGMA wal_checkpoint(TRUNCATE);
```

3. **Analyze for statistics:**
```sql
ANALYZE;  -- Update query planner statistics
```

---

## References

- **Implementation:** `src/db_migrations.py`
- **SQLite Documentation:** https://sqlite.org/docs.html
- **sqlite-vec:** https://github.com/asg017/sqlite-vec
- **WAL Mode:** https://sqlite.org/wal.html

---

**End of SQL Schema Documentation**
