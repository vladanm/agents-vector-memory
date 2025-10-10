# Vector Memory MCP Server - Comprehensive Test Suite Design

**Version:** 1.0
**Server:** /Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/main.py
**Database:** /Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db
**Logs:** /Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/logs
**Test Approach:** MCP Tool Call Sequences (not Python code)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Test Environment Setup](#test-environment-setup)
3. [MCP Functions Inventory](#mcp-functions-inventory)
4. [Test Data Samples](#test-data-samples)
5. [Storage Function Tests](#storage-function-tests)
6. [Search Function Tests](#search-function-tests)
7. [Chunking and Document Tests](#chunking-and-document-tests)
8. [Utility Function Tests](#utility-function-tests)
9. [Edge Case and Error Tests](#edge-case-and-error-tests)
10. [Integration Workflow Tests](#integration-workflow-tests)
11. [Validation Criteria](#validation-criteria)
12. [Test Execution Guide](#test-execution-guide)

---

## 1. Introduction

### Purpose

This document provides a systematic test plan for the Vector Memory MCP Server. It is designed to be executed by a test agent using only MCP tool callsno Python scripts or direct database access. Each test specifies:

- **Setup Phase:** What test data to prepare
- **Execution Phase:** Exact MCP tool calls with parameters
- **Validation Phase:** How to verify correct behavior
- **Cleanup Phase:** How to clean up test data

### Scope

All 28 MCP tools will be tested across:
- 7 memory types (session_context, input_prompt, system_memory, report, report_observation, working_memory, knowledge_base)
- 3 granularity levels (FINE, MEDIUM, COARSE)
- Chunking, embedding, and semantic search
- Document reconstruction and utility functions
- Edge cases and error conditions

### Test Principles

1. **Systematic Coverage:** Every function tested with valid and invalid inputs
2. **Semantic Search Validation:** Verify relevance ranking and similarity scores
3. **Chunking Verification:** Test markdown structure preservation and chunk boundaries
4. **Granularity Testing:** Confirm FINE returns chunks, MEDIUM returns sections, COARSE returns full documents
5. **Memory Isolation:** Verify agent-scoped and session-scoped filtering works correctly

---

## 2. Test Environment Setup

### Prerequisites

1. **MCP Server Running:** Ensure main.py is running with correct database path
2. **Database Access:** Database at /Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db
3. **Log Monitoring:** Tail logs at /Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/logs/mcp_server.log
4. **Clean State:** Recommend using a test database or accept that test data will persist

### Test Session Identifiers

Use consistent identifiers for all tests:
- **Test Session ID:** `test-session-001`
- **Test Session Iteration:** `v1`
- **Test Agent IDs:** `test-agent-alpha`, `test-agent-beta`, `main-orchestrator`
- **Test Task Codes:** `task-001`, `task-002`, `task-003`

### Environment Variables

Verify these config settings in logs:
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2 (384 dimensions)
- `DEFAULT_CHUNK_SIZE`: 450 tokens
- `DEFAULT_CHUNK_OVERLAP`: 50 tokens
- `WARM_START_EMBEDDING_MODEL`: true

---

## 3. MCP Functions Inventory

### Discovery Step (MUST BE FIRST)

Before testing, enumerate all available MCP functions to confirm the server is running correctly:

**Expected MCP Resources/Tools:**
- Use MCP client to call `list_tools()` or equivalent discovery mechanism
- Verify all 28 expected tools are registered
- Check tool signatures match specifications

### Storage Functions (7 tools)

| Function | Memory Type | Auto-Chunk | Fixed Agent |
|----------|-------------|------------|-------------|
| `store_session_context` | session_context | No | main-orchestrator |
| `store_input_prompt` | input_prompt | No | main-orchestrator |
| `store_system_memory` | system_memory | No | Any agent |
| `store_report` | reports | Yes | Any agent |
| `store_report_observation` | report_observations | No | Any agent |
| `store_working_memory` | working_memory | Yes | Any agent |
| `store_knowledge_base` | knowledge_base | Yes | Any agent |

### Search Functions (16 tools)

**Non-Vector (2):**
- `search_session_context` (scoped, no query)
- `search_input_prompts` (scoped, no query)

**Vector Search (1):**
- `search_system_memory` (semantic)

**Granular Search (13 = 3 memory types × 3 granularities + knowledge_base variants):**

| Base Function | Granularity | Returns |
|---------------|-------------|---------|
| `search_reports_specific_chunks` | FINE | Individual chunks |
| `search_reports_section_context` | MEDIUM | Sections with context |
| `search_reports_full_documents` | COARSE | Complete documents |
| `search_working_memory_specific_chunks` | FINE | Individual chunks |
| `search_working_memory_section_context` | MEDIUM | Sections with context |
| `search_working_memory_full_documents` | COARSE | Complete documents |
| `search_knowledge_base_specific_chunks` | FINE | Individual chunks |
| `search_knowledge_base_section_context` | MEDIUM | Sections with context |
| `search_knowledge_base_full_documents` | COARSE | Complete documents |

### Utility Functions (9 tools)

| Function | Purpose |
|----------|---------|
| `load_session_context_for_task` | Load session context for continuation |
| `expand_chunk_context` | Expand surrounding chunks around a specific chunk |
| `reconstruct_document` | Rebuild full document from chunks |
| `get_memory_by_id` | Get specific memory by ID |
| `get_session_stats` | Get session statistics |
| `list_sessions` | List recent sessions |
| `write_document_to_file` | Export document to file |
| `delete_memory` | Delete memory and chunks |
| `cleanup_old_memories` | Bulk cleanup by age |

---

## 4. Test Data Samples

### Sample 1: Small Content (No Chunking)

```markdown
This is a small test document. It contains minimal content that will not trigger automatic chunking because it is well under 450 tokens.
```

**Expected:** 0 chunks created, content stored as-is

---

### Sample 2: Large Multi-Section Document (With Chunking)

```markdown
# Performance Analysis Report

This is the introduction to the performance analysis report. It provides an overview of the system under test.

## Database Query Performance

### Query Execution Times

The database query execution times were measured over a 24-hour period. The average query time was 150ms with a P95 of 450ms and P99 of 800ms. This indicates that most queries complete quickly but there are occasional slow queries that need investigation.

We identified three main categories of slow queries:
1. Full table scans on large tables
2. Missing indexes on frequently joined columns
3. Complex aggregations without proper optimization

### Index Usage Analysis

The index usage analysis revealed that several key tables are missing optimal indexes. The users table, which contains 10 million rows, has only a primary key index. Adding composite indexes on (email, created_at) and (status, last_login) would significantly improve query performance.

Index hit ratio: 85% (target: 95%)
Missing indexes: 12 identified
Unused indexes: 3 candidates for removal

## API Response Times

### Endpoint Performance

The API endpoint performance analysis shows that most endpoints respond within acceptable limits. However, the /search endpoint has a P95 response time of 2.5 seconds, which exceeds our SLA of 1 second.

Top 5 slowest endpoints:
1. /search - 2.5s P95
2. /reports/generate - 1.8s P95
3. /export/data - 1.5s P95
4. /analytics/dashboard - 1.2s P95
5. /user/preferences - 0.9s P95

### Caching Strategy

Implementing a caching layer for frequently accessed data reduced response times by 60% on average. Redis was chosen as the caching solution due to its low latency and high throughput capabilities.

Cache hit rate: 75%
Average latency reduction: 60%
Memory usage: 2.5GB
Eviction policy: LRU with 1-hour TTL

## Memory and Resource Usage

### Memory Profile

The application memory usage shows a steady increase over time, suggesting a potential memory leak. Heap dumps were analyzed and revealed that object retention is primarily in the session management layer.

Current memory usage: 8.5GB
Peak memory usage: 12.1GB
Suspected leak location: SessionCache class
Garbage collection frequency: Every 15 minutes

### CPU Utilization

CPU utilization remains within acceptable ranges during normal operation but spikes to 95% during batch processing jobs. Consider implementing job queuing and rate limiting to smooth out resource consumption.

Average CPU: 45%
Peak CPU: 95%
Bottleneck: Batch report generation

## Recommendations

Based on the analysis, we recommend the following actions:

1. Add missing database indexes (Priority: HIGH)
2. Implement Redis caching layer (Priority: HIGH)
3. Investigate and fix memory leak in SessionCache (Priority: CRITICAL)
4. Optimize /search endpoint with pagination (Priority: MEDIUM)
5. Implement job queue for batch processing (Priority: MEDIUM)
```

**Expected:** ~15-20 chunks created, markdown headers preserved in `header_path`

**Header Hierarchy:**
- "# Performance Analysis Report" (H1, level 1)
- "# Performance Analysis Report > ## Database Query Performance" (H2, level 2)
- "# Performance Analysis Report > ## Database Query Performance > ### Query Execution Times" (H3, level 3)
- etc.

---

### Sample 3: Code-Heavy Content

```markdown
# API Implementation Guide

## Authentication Middleware

The authentication middleware validates JWT tokens:

```python
def authenticate_request(request):
    token = request.headers.get('Authorization')
    if not token:
        raise AuthenticationError("Missing token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        request.user = User.get(payload['user_id'])
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
```

## Database Connection

Connection pooling configuration:

```python
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'production',
    'pool_size': 20,
    'max_overflow': 10,
    'pool_timeout': 30
}
```
```

**Expected:** 2-4 chunks, code blocks preserved, `contains_code` flag set to true

---

### Sample 4: YAML Frontmatter Content

```markdown
---
title: System Configuration
version: 1.2.3
author: DevOps Team
date: 2025-01-15
tags:
  - configuration
  - production
  - deployment
---

# System Configuration

This document describes the production system configuration.

## Environment Variables

Required environment variables for production deployment.
```

**Expected:** Frontmatter extracted into metadata, content chunked normally

---

### Sample 5: Minimal Content

```
Just a single line.
```

**Expected:** 0 chunks, stored as-is, no errors

---

### Sample 6: Empty Content

```
(empty string)
```

**Expected:** Storage should handle gracefully or return validation error

---

## 5. Storage Function Tests

### Test 5.1: store_session_context (Basic)

**Category:** Storage - Session Context
**Memory Type:** session_context
**Auto-Chunk:** No

#### Setup Phase
- Test session: `test-session-001`
- Session iteration: `v1`
- Test content: Sample 1 (small content)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_session_context",
  "parameters": {
    "session_id": "test-session-001",
    "session_iter": "v1",
    "content": "User requested performance analysis of the PNL service. Focus on database queries and API response times.",
    "task_code": "task-001",
    "title": "Session Context for PNL Analysis",
    "description": "Main orchestrator session state",
    "tags": ["session", "pnl", "analysis"]
  }
}
```

#### Validation Phase

**Expected Response Structure:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "memory_type": "session_context",
  "agent_id": "main-orchestrator",
  "session_id": "test-session-001",
  "content_hash": "<16-char-hash>",
  "chunks_created": 0,
  "created_at": "<ISO8601-timestamp>",
  "message": "Memory stored successfully with ID: <id>",
  "error": null
}
```

**Validation Checks:**
1.  `success` is true
2.  `memory_id` is a positive integer
3.  `agent_id` is "main-orchestrator" (fixed for session_context)
4.  `chunks_created` is 0 (no chunking for session_context)
5.  `content_hash` is 16 characters (SHA256 truncated)
6.  `created_at` is valid ISO 8601 timestamp
7.  No error message

**Log Verification:**
- Check logs for: "Memory stored successfully"
- Verify: No "SLOW QUERY" warnings (should be < 100ms)
- Note the memory_id for subsequent tests

#### Cleanup Phase
- Record memory_id: `SESSION_CONTEXT_001 = <memory_id>`
- Keep for integration tests (do not delete yet)

---

### Test 5.2: store_input_prompt

**Category:** Storage - Input Prompt
**Memory Type:** input_prompt
**Auto-Chunk:** No

#### Setup Phase
- Test session: `test-session-001`
- Session iteration: `v1`
- Test content: Original user prompt

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_input_prompt",
  "parameters": {
    "session_id": "test-session-001",
    "session_iter": "v1",
    "content": "Analyze the performance of the PNL service, focusing on database query times and API response latency. Provide recommendations.",
    "task_code": "task-001"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "memory_type": "input_prompt",
  "agent_id": "main-orchestrator",
  "chunks_created": 0,
  "error": null
}
```

**Validation Checks:**
1.  Stored with `agent_id = "main-orchestrator"`
2.  No chunking applied
3.  Content stored verbatim (exact match)

#### Cleanup Phase
- Record memory_id: `INPUT_PROMPT_001 = <memory_id>`

---

### Test 5.3: store_system_memory

**Category:** Storage - System Memory
**Memory Type:** system_memory
**Auto-Chunk:** No

#### Setup Phase
- Agent: `test-agent-alpha`
- Session: `test-session-001`
- Content: System configuration snippet

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_system_memory",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "session_iter": "v1",
    "task_code": "task-001",
    "content": "PostgreSQL connection: host=db.prod.internal port=5432 database=pnl_service user=service_account pool_size=20",
    "title": "Database Configuration"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "agent_id": "test-agent-alpha",
  "memory_type": "system_memory",
  "chunks_created": 0
}
```

**Validation Checks:**
1.  Stored with custom agent_id
2.  No chunking applied
3.  Can be retrieved by agent_id filter

#### Cleanup Phase
- Record memory_id: `SYSTEM_MEMORY_001 = <memory_id>`

---

### Test 5.4: store_report (With Auto-Chunking)

**Category:** Storage - Report with Chunking
**Memory Type:** reports
**Auto-Chunk:** Yes

#### Setup Phase
- Agent: `test-agent-alpha`
- Session: `test-session-001`
- Content: Sample 2 (large multi-section document)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "session_iter": "v1",
    "task_code": "task-001",
    "content": "<Sample 2 content here>",
    "title": "Performance Analysis Report",
    "tags": ["performance", "database", "api"]
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "memory_type": "reports",
  "agent_id": "test-agent-alpha",
  "chunks_created": 15-20,
  "error": null
}
```

**Validation Checks:**
1.  `chunks_created` is > 0 (expect 15-20 chunks for Sample 2)
2.  Storage time < 3 seconds (includes chunking and embedding)
3.  Check logs for: "Generated {N} chunk embeddings"
4.  No chunking errors in logs

**Log Analysis:**
- Verify: "Chunking document..." message
- Verify: "Generated {N} chunk embeddings"
- Verify: Total time logged (should be 1-3 seconds)

#### Cleanup Phase
- Record memory_id: `REPORT_001 = <memory_id>`
- Record chunk count: `REPORT_001_CHUNKS = <chunks_created>`

---

### Test 5.5: store_report (Code-Heavy Content)

**Category:** Storage - Code Blocks
**Memory Type:** reports
**Auto-Chunk:** Yes

#### Setup Phase
- Agent: `test-agent-alpha`
- Content: Sample 3 (code-heavy content)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "session_iter": "v1",
    "task_code": "task-002",
    "content": "<Sample 3 content>",
    "title": "API Implementation Guide"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "chunks_created": 2-4
}
```

**Validation Checks:**
1.  Code blocks preserved in chunks
2.  `contains_code` flag set to true in chunk metadata (verify in logs)
3.  No code fence corruption (verify by reconstruction later)

#### Cleanup Phase
- Record memory_id: `REPORT_CODE_001 = <memory_id>`

---

### Test 5.6: store_report_observation

**Category:** Storage - Report Observation
**Memory Type:** report_observations
**Auto-Chunk:** No

#### Setup Phase
- Link to: REPORT_001 (from Test 5.4)
- Content: Follow-up note

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report_observation",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "session_iter": "v1",
    "task_code": "task-001",
    "parent_report_id": "<REPORT_001>",
    "content": "Additional finding: The SessionCache memory leak was confirmed by heap dump analysis. Recommendation priority upgraded to CRITICAL.",
    "title": "Critical Memory Leak Update"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "memory_type": "report_observations",
  "chunks_created": 0
}
```

**Validation Checks:**
1.  No chunking applied
2.  `parent_report_id` stored in metadata
3.  Can be retrieved by filtering

#### Cleanup Phase
- Record memory_id: `OBSERVATION_001 = <memory_id>`

---

### Test 5.7: store_working_memory

**Category:** Storage - Working Memory
**Memory Type:** working_memory
**Auto-Chunk:** Yes

#### Setup Phase
- Agent: `test-agent-alpha`
- Content: Medium-length intermediate findings

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_working_memory",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "session_iter": "v1",
    "task_code": "task-001",
    "content": "Initial findings from database profiling: Identified 12 missing indexes. Query patterns show heavy use of email lookups without index. Next steps: Analyze index creation impact on write performance.",
    "title": "Database Profiling Notes"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_type": "working_memory",
  "chunks_created": 0-2
}
```

**Validation Checks:**
1.  Auto-chunking behavior depends on content length
2.  Small content (< 450 tokens) ’ 0 chunks
3.  Large content ’ chunks created

#### Cleanup Phase
- Record memory_id: `WORKING_MEMORY_001 = <memory_id>`

---

### Test 5.8: store_knowledge_base (No Session Scope)

**Category:** Storage - Knowledge Base
**Memory Type:** knowledge_base
**Auto-Chunk:** Yes

#### Setup Phase
- Agent: `test-agent-alpha`
- Content: Sample 2 (reuse large document)
- Note: No session_id required (global knowledge)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_knowledge_base",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "title": "PostgreSQL Performance Optimization",
    "content": "<Sample 2 or similar large content>",
    "category": "database",
    "description": "Best practices for PostgreSQL query optimization",
    "tags": ["postgresql", "performance", "indexes"]
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <integer>,
  "memory_type": "knowledge_base",
  "chunks_created": 15-20
}
```

**Validation Checks:**
1.  No session_id required
2.  Auto-chunking applied for large content
3.  Can be searched across all sessions
4.  Title and category stored in metadata

#### Cleanup Phase
- Record memory_id: `KNOWLEDGE_BASE_001 = <memory_id>`

---

### Test 5.9: Duplicate Content Detection

**Category:** Storage - Deduplication
**Expected:** Duplicate prevention via content_hash

#### Setup Phase
- Reuse content from Test 5.1

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_session_context",
  "parameters": {
    "session_id": "test-session-001",
    "session_iter": "v1",
    "content": "<exact same content as Test 5.1>",
    "task_code": "task-001"
  }
}
```

#### Validation Phase

**Expected Behavior:**
- Check if `content_hash` collision is detected
- System may either:
  - A) Accept duplicate (creates new memory_id, same hash)
  - B) Reject duplicate (returns error)

**Validation Checks:**
1.  Verify behavior documented in logs
2.  If duplicate accepted: Two memory entries with same content_hash
3.  If duplicate rejected: Error message clear and helpful

#### Cleanup Phase
- Document observed behavior for deduplication

---

## 6. Search Function Tests

### Test 6.1: search_session_context (Scoped Search)

**Category:** Search - Session Context
**Type:** Scoped (no vector search)

#### Setup Phase
- Prerequisites: Test 5.1 completed (SESSION_CONTEXT_001 exists)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_session_context",
  "parameters": {
    "session_id": "test-session-001",
    "session_iter": "v1",
    "limit": 5
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": <SESSION_CONTEXT_001>,
      "memory_type": "session_context",
      "agent_id": "main-orchestrator",
      "session_id": "test-session-001",
      "session_iter": 1,
      "content": "User requested performance analysis...",
      "similarity": 2.0,
      "source_type": "scoped"
    }
  ],
  "total_results": 1,
  "query": null,
  "limit": 5
}
```

**Validation Checks:**
1.  Returns stored session_context
2.  `similarity` is 2.0 (indicates scoped match, not vector search)
3.  `source_type` is "scoped"
4.  Results ordered by `session_iter DESC, created_at DESC`
5.  No vector search performed (check logs: no embedding generation)

#### Cleanup Phase
- None (read-only operation)

---

### Test 6.2: search_input_prompts

**Category:** Search - Input Prompts
**Type:** Scoped

#### Setup Phase
- Prerequisites: Test 5.2 completed (INPUT_PROMPT_001 exists)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_input_prompts",
  "parameters": {
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": <INPUT_PROMPT_001>,
      "memory_type": "input_prompt",
      "content": "Analyze the performance of the PNL service...",
      "similarity": 2.0
    }
  ],
  "total_results": 1
}
```

**Validation Checks:**
1.  Returns stored input_prompt verbatim
2.  No vector search
3.  Scoped to session

---

### Test 6.3: search_system_memory (Semantic Search)

**Category:** Search - System Memory
**Type:** Semantic Vector Search

#### Setup Phase
- Prerequisites: Test 5.3 completed (SYSTEM_MEMORY_001 exists)
- Query: "database connection configuration"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_system_memory",
  "parameters": {
    "query": "database connection configuration",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": <SYSTEM_MEMORY_001>,
      "memory_type": "system_memory",
      "content": "PostgreSQL connection: host=db.prod.internal...",
      "similarity": 0.65-0.95,
      "source_type": "semantic"
    }
  ],
  "total_results": 1
}
```

**Validation Checks:**
1.  Similarity score between 0.0 and 1.0
2.  Higher similarity for better semantic match
3.  Results contain SYSTEM_MEMORY_001
4.  Check logs: "Generated query embedding" message
5.  Search time < 500ms

**Semantic Relevance Test:**
- The query "database connection configuration" should match "PostgreSQL connection..."
- Verify similarity > 0.5 (indicates semantic relevance)

---

### Test 6.4: search_reports_specific_chunks (FINE Granularity)

**Category:** Search - Reports (FINE)
**Granularity:** FINE (chunk-level)

#### Setup Phase
- Prerequisites: Test 5.4 completed (REPORT_001 with 15-20 chunks)
- Query: "database query performance"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "database query performance",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "chunk_id": <integer>,
      "memory_id": <REPORT_001>,
      "chunk_index": <integer>,
      "chunk_content": "The database query execution times were measured...",
      "chunk_type": "section",
      "header_path": "# Performance Analysis Report > ## Database Query Performance > ### Query Execution Times",
      "level": 3,
      "similarity": 0.75-0.95,
      "source": "chunk",
      "granularity": "fine"
    }
  ],
  "total_results": 5-10,
  "granularity": "fine"
}
```

**Validation Checks:**
1.  Returns individual chunks (~400-500 tokens each)
2.  `granularity` is "fine"
3.  `header_path` shows markdown hierarchy
4.  Top result has highest similarity
5.  Results sorted by similarity DESC
6.  `level` indicates header depth (1-6)
7.  Search focused on "Database Query Performance" section

**Semantic Relevance:**
- Top chunk should contain text about query execution times, indexes, or performance
- Verify similarity scores are reasonable (> 0.5 for good matches)

#### Cleanup Phase
- Record top chunk_id for Test 8.2 (expand_chunk_context)

---

### Test 6.5: search_reports_section_context (MEDIUM Granularity)

**Category:** Search - Reports (MEDIUM)
**Granularity:** MEDIUM (section-level with auto-merge)

#### Setup Phase
- Prerequisites: Test 5.4 completed
- Query: "database query performance"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_reports_section_context",
  "parameters": {
    "query": "database query performance",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 5
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "memory_id": <REPORT_001>,
      "section_header": "# Performance Analysis Report > ## Database Query Performance",
      "section_content": "<full section content with all chunks>",
      "header_path": "# Performance Analysis Report > ## Database Query Performance",
      "chunks_in_section": 6,
      "matched_chunks": 4,
      "match_ratio": 0.67,
      "auto_merged": true,
      "similarity": 0.80,
      "source": "expanded_section",
      "granularity": "medium"
    }
  ],
  "total_results": 2-3,
  "granularity": "medium"
}
```

**Validation Checks:**
1.  Returns full sections (not individual chunks)
2.  `section_header` is H1 > H2 format (not just H1)
3.  `section_content` is multiple paragraphs (all chunks joined)
4.  `match_ratio` = matched_chunks / chunks_in_section
5.  `auto_merged` is true if match_ratio >= 0.6
6.  `similarity` is average of matching chunks
7.  Section boundaries correct (H2 level in this case)

**Section Boundary Validation:**
- "## Database Query Performance" section should include:
  - "### Query Execution Times"
  - "### Index Usage Analysis"
- Should NOT include:
  - "## API Response Times" (different H2 section)

#### Cleanup Phase
- None

---

### Test 6.6: search_reports_full_documents (COARSE Granularity)

**Category:** Search - Reports (COARSE)
**Granularity:** COARSE (full documents)

#### Setup Phase
- Prerequisites: Test 5.4 completed
- Query: "performance analysis"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_reports_full_documents",
  "parameters": {
    "query": "performance analysis",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 3
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": <REPORT_001>,
      "memory_type": "reports",
      "content": "<complete report content, not chunks>",
      "title": "Performance Analysis Report",
      "similarity": 2.0,
      "source_type": "scoped"
    }
  ],
  "total_results": 1,
  "granularity": "coarse"
}
```

**Validation Checks:**
1.  Returns complete document (full content field)
2.  `similarity` is 2.0 (scoped match, no vector search for coarse)
3.  Content is NOT chunked (returned as single string)
4.  No chunk metadata in response
5.  Fast response (< 200ms, no vector ops)

---

### Test 6.7: search_working_memory_specific_chunks

**Category:** Search - Working Memory (FINE)
**Granularity:** FINE

#### Setup Phase
- Prerequisites: Test 5.7 completed
- Query: "missing indexes database"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_working_memory_specific_chunks",
  "parameters": {
    "query": "missing indexes database",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "chunk_content": "...Identified 12 missing indexes...",
      "similarity": 0.70-0.90,
      "granularity": "fine"
    }
  ]
}
```

**Validation Checks:**
1.  Returns working_memory chunks
2.  Semantic match for "missing indexes"
3.  FINE granularity behavior

---

### Test 6.8: search_knowledge_base_section_context

**Category:** Search - Knowledge Base (MEDIUM)
**Granularity:** MEDIUM

#### Setup Phase
- Prerequisites: Test 5.8 completed
- Query: "database index optimization"

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_knowledge_base_section_context",
  "parameters": {
    "query": "database index optimization",
    "agent_id": "test-agent-alpha",
    "limit": 5
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "section_header": "# PostgreSQL Performance Optimization > ## Index Usage Analysis",
      "section_content": "<full section>",
      "granularity": "medium"
    }
  ]
}
```

**Validation Checks:**
1.  Returns knowledge_base sections
2.  No session_id filter (global search)
3.  Semantic relevance to query

---

### Test 6.9: Cross-Agent Memory Isolation

**Category:** Search - Memory Isolation
**Purpose:** Verify agent-scoped memory doesn't leak

#### Setup Phase
- Store two reports:
  1. Agent A: test-agent-alpha
  2. Agent B: test-agent-beta

#### Execution Phase

**Step 1: Store report for Agent A**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "content": "Agent A's confidential findings about security vulnerabilities.",
    "task_code": "task-security"
  }
}
```

**Step 2: Store report for Agent B**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-beta",
    "session_id": "test-session-001",
    "content": "Agent B's public findings about performance improvements.",
    "task_code": "task-performance"
  }
}
```

**Step 3: Search with Agent A filter**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "findings",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
- Results should only contain Agent A's report
- Agent B's report should NOT appear

**Validation Checks:**
1.  Only Agent A's report returned
2.  No leakage of Agent B's data
3.  Filter applied correctly in post-filtering

**Step 4: Search with Agent B filter**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "findings",
    "agent_id": "test-agent-beta",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

**Validation:**
-  Only Agent B's report returned

---

### Test 6.10: Search Performance Under Load

**Category:** Search - Performance
**Purpose:** Verify iterative fetching handles low selectivity

#### Setup Phase
- Prerequisites: Multiple reports stored across different agents/sessions
- Simulate low selectivity by using very specific filters

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "performance",
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "task_code": "task-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Log Analysis:**
1.  Check for "Iterative search performance" log entry
2.  Verify: `fetched` count > `kept` count (post-filtering working)
3.  Verify: `batches` > 1 if selectivity < 10%
4.  Verify: No "Could not find enough results" warning
5.  Total elapsed < 2 seconds (slow query threshold)

**Expected Log:**
```
Iterative search performance: elapsed=0.456s, fetched=500, kept=10, target=10, batches=5
```

---

## 7. Chunking and Document Tests

### Test 7.1: Chunk Boundary Verification

**Category:** Chunking - Markdown Structure
**Purpose:** Verify headers preserved in chunks

#### Setup Phase
- Prerequisites: Test 5.4 completed (REPORT_001)

#### Execution Phase

**Step 1: Search for specific section**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "index usage analysis",
    "session_id": "test-session-001",
    "limit": 5
  }
}
```

**Step 2: Verify header_path**

#### Validation Phase

**Expected:**
- At least one chunk with `header_path` containing:
  - "### Index Usage Analysis"

**Validation Checks:**
1.  `header_path` shows full hierarchy: "# Title > ## Section > ### Subsection"
2.  `level` is 3 (for ### header)
3.  Chunk content starts within that subsection
4.  No header corruption (e.g., missing #, wrong nesting)

---

### Test 7.2: Document Reconstruction

**Category:** Chunking - Document Integrity
**Purpose:** Verify chunks can be reconstructed into original document

#### Setup Phase
- Prerequisites: Test 5.4 completed (REPORT_001 with chunks)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "reconstruct_document",
  "parameters": {
    "memory_id": "<REPORT_001>"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <REPORT_001>,
  "content": "<full reconstructed document>",
  "title": "Performance Analysis Report",
  "memory_type": "reports",
  "chunk_count": 15-20,
  "message": "Document reconstructed from {N} chunks"
}
```

**Validation Checks:**
1.  `content` matches original Sample 2 content (exact or nearly exact)
2.  All markdown headers present
3.  No chunk boundaries visible (seamless joins)
4.  No duplicate content at chunk boundaries
5.  `chunk_count` matches storage result from Test 5.4

**Content Integrity:**
- Manually compare reconstructed content with Sample 2
- Verify: All 5 H2 sections present
- Verify: All H3 subsections present
- Verify: No text loss at chunk boundaries

---

### Test 7.3: YAML Frontmatter Extraction

**Category:** Chunking - Frontmatter
**Purpose:** Verify YAML frontmatter extracted to metadata

#### Setup Phase
- Content: Sample 4 (YAML frontmatter)

#### Execution Phase

**Step 1: Store document with frontmatter**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "content": "<Sample 4 with YAML frontmatter>",
    "task_code": "task-003"
  }
}
```

**Step 2: Retrieve and check metadata**
```json
{
  "tool": "get_memory_by_id",
  "parameters": {
    "memory_id": "<result from step 1>"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory": {
    "metadata": {
      "frontmatter": {
        "title": "System Configuration",
        "version": "1.2.3",
        "author": "DevOps Team",
        "tags": ["configuration", "production", "deployment"]
      }
    }
  }
}
```

**Validation Checks:**
1.  Frontmatter extracted to `metadata.frontmatter`
2.  Content field does NOT include frontmatter (stripped)
3.  All frontmatter fields preserved

---

### Test 7.4: Code Block Preservation

**Category:** Chunking - Code Blocks
**Purpose:** Verify code fences not corrupted

#### Setup Phase
- Prerequisites: Test 5.5 completed (REPORT_CODE_001)

#### Execution Phase

**Step 1: Reconstruct document**
```json
{
  "tool": "reconstruct_document",
  "parameters": {
    "memory_id": "<REPORT_CODE_001>"
  }
}
```

#### Validation Phase

**Validation Checks:**
1.  Code fences (```) intact
2.  Syntax highlighting language preserved (```python)
3.  Indentation preserved
4.  No extra whitespace in code blocks
5.  Code not split mid-function

**Manual Inspection:**
- Compare reconstructed code blocks with Sample 3
- Verify: `def authenticate_request(request):` is complete
- Verify: DATABASE_CONFIG dictionary is intact

---

### Test 7.5: Chunk Overlap Verification

**Category:** Chunking - Overlap
**Purpose:** Verify 50-token overlap works correctly

#### Setup Phase
- Prerequisites: Test 5.4 completed

#### Execution Phase

**Step 1: Get two consecutive chunks**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "database query",
    "limit": 10
  }
}
```

**Step 2: Find chunks with consecutive chunk_index**
- Identify chunks with `chunk_index` N and N+1 from same memory_id

#### Validation Phase

**Manual Analysis:**
1. Compare end of chunk N with start of chunk N+1
2.  Last ~50 tokens of chunk N should appear at start of chunk N+1
3.  Overlap preserves context at boundaries
4.  No mid-sentence cuts (overlap should end at sentence boundary if possible)

**Log Verification:**
- Check logs for chunking parameters:
  - `chunk_size: 450`
  - `chunk_overlap: 50`

---

## 8. Utility Function Tests

### Test 8.1: load_session_context_for_task

**Category:** Utility - Session Loading
**Purpose:** Load complete session context for continuation

#### Setup Phase
- Prerequisites: Tests 5.1, 5.2, 5.4, 5.7 completed

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "load_session_context_for_task",
  "parameters": {
    "session_id": "test-session-001",
    "session_iter": "v1"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "found_previous_context": true,
  "context": {
    "id": <SESSION_CONTEXT_001>,
    "content": "User requested performance analysis...",
    "session_id": "test-session-001",
    "session_iter": 1
  },
  "message": "Found session context for test-session-001:v1"
}
```

**Validation Checks:**
1.  Returns latest session_context for the session
2.  `found_previous_context` is true
3.  Context contains complete session state

---

### Test 8.2: expand_chunk_context

**Category:** Utility - Chunk Context
**Purpose:** Expand surrounding chunks around a target chunk

#### Setup Phase
- Prerequisites: Test 6.4 completed (recorded chunk_id)
- Use chunk_id from top search result

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "expand_chunk_context",
  "parameters": {
    "chunk_id": "<chunk_id from Test 6.4>",
    "surrounding_chunks": 2
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <REPORT_001>,
  "target_chunk_index": 5,
  "context_window": 2,
  "chunks_returned": 5,
  "expanded_content": "<joined content of chunks 3,4,5,6,7>",
  "chunks": [
    {"chunk_index": 3, "content": "..."},
    {"chunk_index": 4, "content": "..."},
    {"chunk_index": 5, "content": "..."},
    {"chunk_index": 6, "content": "..."},
    {"chunk_index": 7, "content": "..."}
  ]
}
```

**Validation Checks:**
1.  Returns target chunk plus 2 before and 2 after (5 total if available)
2.  `chunks_returned` is 5 (or less at document boundaries)
3.  `expanded_content` is chunks joined with "\n\n"
4.  Chunks ordered by `chunk_index`
5.  No gaps in chunk_index sequence

---

### Test 8.3: get_memory_by_id

**Category:** Utility - Memory Retrieval
**Purpose:** Get specific memory by ID

#### Setup Phase
- Prerequisites: Test 5.1 completed

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "get_memory_by_id",
  "parameters": {
    "memory_id": "<SESSION_CONTEXT_001>"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory": {
    "id": <SESSION_CONTEXT_001>,
    "memory_type": "session_context",
    "agent_id": "main-orchestrator",
    "session_id": "test-session-001",
    "content": "User requested performance analysis...",
    "title": "Session Context for PNL Analysis",
    "tags": ["session", "pnl", "analysis"],
    "metadata": {"scope": "session"},
    "created_at": "<ISO8601>",
    "access_count": 1-3
  }
}
```

**Validation Checks:**
1.  Returns complete memory record
2.  All fields populated correctly
3.  `access_count` incremented (may be > 1 due to previous reads)

---

### Test 8.4: get_session_stats

**Category:** Utility - Session Statistics
**Purpose:** Get aggregated session statistics

#### Setup Phase
- Prerequisites: Multiple tests completed (session has data)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "get_session_stats",
  "parameters": {
    "session_id": "test-session-001"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "total_memories": 5-10,
  "memory_types": 5-7,
  "unique_agents": 2-3,
  "unique_sessions": 1,
  "max_session_iter": 1,
  "memory_type_breakdown": {
    "session_context": 1,
    "input_prompt": 1,
    "reports": 2-3,
    "working_memory": 1,
    "system_memory": 1
  }
}
```

**Validation Checks:**
1.  `total_memories` matches count of stored items
2.  `memory_type_breakdown` shows counts per type
3.  `unique_agents` includes test agents and main-orchestrator
4.  All counts are non-negative integers

---

### Test 8.5: list_sessions

**Category:** Utility - Session Listing
**Purpose:** List recent sessions with activity

#### Setup Phase
- Prerequisites: Tests completed (session active)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "list_sessions",
  "parameters": {
    "limit": 20
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "agent_id": "test-agent-alpha",
      "session_id": "test-session-001",
      "memory_count": 5-10,
      "latest_iter": 1,
      "latest_activity": "<ISO8601>",
      "memory_types": ["session_context", "reports", "working_memory"]
    }
  ],
  "total_sessions": 1-N
}
```

**Validation Checks:**
1.  Returns test-session-001
2.  `memory_count` > 0
3.  Ordered by `latest_activity DESC` (most recent first)

---

### Test 8.6: write_document_to_file

**Category:** Utility - File Export
**Purpose:** Export reconstructed document to file

#### Setup Phase
- Prerequisites: Test 5.4 completed
- Target path: `/tmp/test_export_report.md`

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "write_document_to_file",
  "parameters": {
    "memory_id": "<REPORT_001>",
    "output_path": "/tmp/test_export_report.md"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "file_path": "/tmp/test_export_report.md",
  "memory_id": <REPORT_001>,
  "bytes_written": 5000-10000,
  "message": "Document written to /tmp/test_export_report.md"
}
```

**Validation Checks:**
1.  File created at specified path
2.  File size matches `bytes_written`
3.  File contains YAML frontmatter (if PyYAML available)
4.  File content matches reconstructed document

**Manual Verification:**
```bash
cat /tmp/test_export_report.md | head -20
```

Expected output:
```markdown
---
memory_id: <REPORT_001>
title: "Performance Analysis Report"
memory_type: "reports"
chunk_count: 15
---

# Performance Analysis Report
...
```

#### Cleanup Phase
```bash
rm /tmp/test_export_report.md
```

---

### Test 8.7: delete_memory

**Category:** Utility - Memory Deletion
**Purpose:** Delete memory and verify cascade deletion

#### Setup Phase
- Create a new test report specifically for deletion

**Step 1: Store test report**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-deletion",
    "content": "This report will be deleted for testing purposes.",
    "task_code": "task-delete-test"
  }
}
```
Record: `DELETE_TEST_ID = <memory_id>`

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "delete_memory",
  "parameters": {
    "memory_id": "<DELETE_TEST_ID>"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "memory_id": <DELETE_TEST_ID>,
  "message": "Memory {id} deleted successfully"
}
```

**Validation Checks:**
1.  Deletion succeeds
2.  Memory no longer retrievable via get_memory_by_id
3.  Chunks cascade deleted (verify in logs: "DELETE FROM memory_chunks")

**Verification Step:**
```json
{
  "tool": "get_memory_by_id",
  "parameters": {
    "memory_id": "<DELETE_TEST_ID>"
  }
}
```

**Expected:**
```json
{
  "success": false,
  "error": "Memory not found"
}
```

---

### Test 8.8: cleanup_old_memories (Dry Run)

**Category:** Utility - Bulk Cleanup
**Purpose:** Test cleanup function in dry-run mode

#### Setup Phase
- Existing test data is fine (won't be deleted)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "cleanup_old_memories",
  "parameters": {
    "days_old": 90,
    "dry_run": true
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "deleted_count": 0-N,
  "oldest_deleted": null,
  "message": "Would delete {N} memories older than 90 days"
}
```

**Validation Checks:**
1.  `dry_run: true` by default
2.  No actual deletions (count is projection)
3.  Returns count of what would be deleted
4.  Message clearly indicates dry-run mode

**Safety Check:**
- Verify test data still exists after dry-run

---

## 9. Edge Case and Error Tests

### Test 9.1: Empty Content

**Category:** Edge Case - Empty Input
**Purpose:** Handle empty string gracefully

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "content": "",
    "task_code": "task-empty"
  }
}
```

#### Validation Phase

**Expected Behavior:**
- Option A: Accept empty content (success: true, chunks_created: 0)
- Option B: Reject with validation error

**Validation Checks:**
1.  No crash or exception
2.  Error message clear if rejected
3.  If accepted: `chunks_created` is 0

---

### Test 9.2: Very Large Content

**Category:** Edge Case - Large Document
**Purpose:** Test with content > 500,000 characters

#### Setup Phase
- Generate large content (repeat Sample 2 multiple times)

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "content": "<very large content, 500K+ chars>",
    "task_code": "task-large"
  }
}
```

#### Validation Phase

**Expected:**
- May hit validation limit (500,000 char max)
- If accepted: Many chunks created (100+)

**Validation Checks:**
1.  Storage time < 10 seconds (even for large content)
2.  No timeout errors
3.  Check logs for batch embedding generation
4.  If rejected: Clear error about size limit

---

### Test 9.3: Invalid Memory ID

**Category:** Error Handling - Invalid Input
**Purpose:** Test with non-existent memory_id

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "get_memory_by_id",
  "parameters": {
    "memory_id": "999999999"
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": false,
  "memory": null,
  "error": "Memory not found",
  "message": "No memory with ID: 999999999"
}
```

**Validation Checks:**
1.  `success` is false
2.  Error message helpful
3.  No stack trace exposed to user

---

### Test 9.4: Invalid Chunk ID Format

**Category:** Error Handling - Type Validation
**Purpose:** Test with invalid chunk_id type

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "expand_chunk_context",
  "parameters": {
    "chunk_id": "not-a-number",
    "surrounding_chunks": 2
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": false,
  "error": "Invalid chunk_id",
  "message": "chunk_id must be a valid integer, got: not-a-number"
}
```

**Validation Checks:**
1.  Type validation catches error
2.  Clear error message
3.  No crash

---

### Test 9.5: Missing Required Parameters

**Category:** Error Handling - Missing Parameters
**Purpose:** Test with missing required fields

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_session_context",
  "parameters": {
    "session_id": "test-session-001"
  }
}
```
(Missing `session_iter` and `content`)

#### Validation Phase

**Expected:**
- MCP framework should reject before reaching tool code
- Error indicates which parameters are missing

**Validation Checks:**
1.  Parameter validation works
2.  Error message lists missing fields

---

### Test 9.6: Search with No Results

**Category:** Edge Case - Empty Results
**Purpose:** Query that matches nothing

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "xyzabc123nonexistentterm",
    "session_id": "test-session-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Expected Response:**
```json
{
  "success": true,
  "results": [],
  "total_results": 0,
  "message": null
}
```

**Validation Checks:**
1.  Empty results array (not null)
2.  `success` is still true (not an error)
3.  No exceptions in logs

---

### Test 9.7: Unicode and Special Characters

**Category:** Edge Case - Character Encoding
**Purpose:** Handle Unicode correctly

#### Execution Phase

**MCP Tool Call:**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-alpha",
    "session_id": "test-session-001",
    "content": "Test with émojis = <‰ and spëcial çhars: -‡, 'D91(J), å,ž",
    "task_code": "task-unicode"
  }
}
```

#### Validation Phase

**Validation Checks:**
1.  Content stored without corruption
2.  Retrieval returns exact Unicode
3.  Search works with Unicode queries
4.  No encoding errors in logs

**Verification:**
```json
{
  "tool": "search_reports_full_documents",
  "parameters": {
    "query": "émojis = ",
    "session_id": "test-session-001",
    "limit": 1
  }
}
```

Expected: Finds the Unicode content

---

## 10. Integration Workflow Tests

### Test 10.1: Complete Store-Search-Retrieve Workflow

**Category:** Integration - Full Lifecycle
**Purpose:** Test complete workflow from storage to retrieval

#### Workflow Steps

**Step 1: Store session context**
```json
{
  "tool": "store_session_context",
  "parameters": {
    "session_id": "workflow-test-001",
    "session_iter": "v1",
    "content": "User wants to analyze API performance focusing on the /search endpoint."
  }
}
```

**Step 2: Store input prompt**
```json
{
  "tool": "store_input_prompt",
  "parameters": {
    "session_id": "workflow-test-001",
    "session_iter": "v1",
    "content": "Analyze the /search endpoint performance and suggest optimizations."
  }
}
```

**Step 3: Store large report (with chunking)**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "test-agent-workflow",
    "session_id": "workflow-test-001",
    "session_iter": "v1",
    "content": "<Sample 2: Performance Analysis Report>",
    "task_code": "workflow-task-001"
  }
}
```
Record: `WORKFLOW_REPORT_ID`

**Step 4: Search at FINE granularity**
```json
{
  "tool": "search_reports_specific_chunks",
  "parameters": {
    "query": "search endpoint performance",
    "session_id": "workflow-test-001",
    "limit": 5
  }
}
```

**Step 5: Search at MEDIUM granularity**
```json
{
  "tool": "search_reports_section_context",
  "parameters": {
    "query": "search endpoint performance",
    "session_id": "workflow-test-001",
    "limit": 3
  }
}
```

**Step 6: Get full document**
```json
{
  "tool": "search_reports_full_documents",
  "parameters": {
    "query": "performance",
    "session_id": "workflow-test-001",
    "limit": 1
  }
}
```

**Step 7: Reconstruct document**
```json
{
  "tool": "reconstruct_document",
  "parameters": {
    "memory_id": "<WORKFLOW_REPORT_ID>"
  }
}
```

**Step 8: Load session context**
```json
{
  "tool": "load_session_context_for_task",
  "parameters": {
    "session_id": "workflow-test-001",
    "session_iter": "v1"
  }
}
```

**Step 9: Get session stats**
```json
{
  "tool": "get_session_stats",
  "parameters": {
    "session_id": "workflow-test-001"
  }
}
```

#### Validation Phase

**Validation Checks:**
1.  All steps succeed
2.  FINE search returns chunks mentioning "/search endpoint"
3.  MEDIUM search returns "API Response Times" section
4.  COARSE search returns full document
5.  Reconstruction matches original content
6.  Session context loaded correctly
7.  Session stats show all stored items

**Performance Checks:**
- Step 3 (store with chunking): < 3 seconds
- Step 4 (FINE search): < 500ms
- Step 5 (MEDIUM search): < 1500ms
- Step 6 (COARSE search): < 200ms
- Step 7 (reconstruction): < 500ms

---

### Test 10.2: Multi-Agent Collaboration Simulation

**Category:** Integration - Multi-Agent
**Purpose:** Simulate multiple agents working on same session

#### Workflow Steps

**Agent 1: Store initial findings**
```json
{
  "tool": "store_working_memory",
  "parameters": {
    "agent_id": "agent-database-analyst",
    "session_id": "collab-test-001",
    "content": "Found 12 missing indexes. Analyzing impact on query performance...",
    "task_code": "analyze-db"
  }
}
```

**Agent 2: Store complementary analysis**
```json
{
  "tool": "store_working_memory",
  "parameters": {
    "agent_id": "agent-api-analyzer",
    "session_id": "collab-test-001",
    "content": "Traced /search endpoint slowness to database queries. Need index recommendations from DB team.",
    "task_code": "analyze-api"
  }
}
```

**Agent 1: Store final report**
```json
{
  "tool": "store_report",
  "parameters": {
    "agent_id": "agent-database-analyst",
    "session_id": "collab-test-001",
    "content": "Database analysis complete. Recommend adding composite index on (email, created_at).",
    "task_code": "analyze-db"
  }
}
```

**Agent 2: Add observation**
```json
{
  "tool": "store_report_observation",
  "parameters": {
    "agent_id": "agent-api-analyzer",
    "session_id": "collab-test-001",
    "content": "Confirmed that suggested index would improve /search by 75% based on query patterns.",
    "task_code": "analyze-api"
  }
}
```

**Search across all agents**
```json
{
  "tool": "search_working_memory_specific_chunks",
  "parameters": {
    "query": "missing indexes",
    "session_id": "collab-test-001",
    "limit": 10
  }
}
```

**Search specific agent**
```json
{
  "tool": "search_working_memory_specific_chunks",
  "parameters": {
    "query": "missing indexes",
    "agent_id": "agent-database-analyst",
    "session_id": "collab-test-001",
    "limit": 10
  }
}
```

#### Validation Phase

**Validation Checks:**
1.  Both agents' memories stored separately
2.  Cross-agent search (no agent_id filter) finds both
3.  Agent-specific search finds only that agent's data
4.  Session stats show 2 unique agents
5.  Report observation links to parent (if parent_report_id used)

---

### Test 10.3: Session Continuation Workflow

**Category:** Integration - Session Resume
**Purpose:** Simulate agent resuming work on a session

#### Workflow Steps

**Initial Session (v1):**

**Step 1: Store initial context**
```json
{
  "tool": "store_session_context",
  "parameters": {
    "session_id": "continuation-test-001",
    "session_iter": "v1",
    "content": "Session started. User asked for performance analysis."
  }
}
```

**Step 2: Store work in progress**
```json
{
  "tool": "store_working_memory",
  "parameters": {
    "agent_id": "test-agent-continuation",
    "session_id": "continuation-test-001",
    "session_iter": "v1",
    "content": "Analyzed first 3 endpoints. Need to continue with remaining 7 endpoints."
  }
}
```

**Simulate Session Pause**

**Session Resume (v1):**

**Step 3: Load previous context**
```json
{
  "tool": "load_session_context_for_task",
  "parameters": {
    "session_id": "continuation-test-001",
    "session_iter": "v1"
  }
}
```

**Step 4: Search previous working memory**
```json
{
  "tool": "search_working_memory_full_documents",
  "parameters": {
    "query": "endpoints",
    "session_id": "continuation-test-001",
    "limit": 10
  }
}
```

**Step 5: Continue work**
```json
{
  "tool": "store_working_memory",
  "parameters": {
    "agent_id": "test-agent-continuation",
    "session_id": "continuation-test-001",
    "session_iter": "v1",
    "content": "Resumed analysis. Completed remaining 7 endpoints. Ready for final report."
  }
}
```

#### Validation Phase

**Validation Checks:**
1.  Load context returns initial session state
2.  Search finds previous working memory
3.  New working memory stored correctly
4.  Session stats show cumulative progress
5.  All items have same session_iter ("v1")

---

## 11. Validation Criteria

### Success Criteria

A test passes if ALL of the following are true:

1. **Functional Correctness:**
   - MCP tool call returns expected response structure
   - `success` field is true (unless testing error case)
   - All required fields present in response
   - Data values are correct and reasonable

2. **Semantic Search Accuracy:**
   - Top results are semantically relevant to query
   - Similarity scores > 0.5 for good matches
   - Results ordered by relevance (similarity DESC)
   - No false positives in top 5 results

3. **Chunking Integrity:**
   - Markdown headers preserved in `header_path`
   - Chunk boundaries don't split sentences mid-word (where possible)
   - Reconstructed document matches original
   - Code blocks and special formatting intact

4. **Granularity Behavior:**
   - FINE returns individual chunks (~400 tokens)
   - MEDIUM returns sections with full context
   - COARSE returns complete documents
   - No mixing of granularities in single result set

5. **Performance:**
   - Storage with chunking: < 3 seconds
   - FINE search: < 500ms
   - MEDIUM search: < 1500ms
   - COARSE search: < 200ms
   - No "SLOW QUERY" warnings for normal operations

6. **Memory Isolation:**
   - Agent-scoped filters work correctly
   - Session-scoped filters work correctly
   - No cross-contamination between agents or sessions
   - Knowledge base accessible across all sessions

7. **Error Handling:**
   - Invalid inputs return clear error messages
   - No crashes or unhandled exceptions
   - Error responses have proper structure
   - Logs contain useful debugging information

### Failure Criteria

A test fails if ANY of the following occur:

1. **Critical Failures:**
   - MCP tool call throws unhandled exception
   - Server crashes or becomes unresponsive
   - Database corruption detected
   - Data loss (stored data cannot be retrieved)

2. **Functional Failures:**
   - Wrong response structure
   - Missing required fields
   - Incorrect data values
   - `success` is true but operation actually failed

3. **Search Quality Failures:**
   - Top results completely irrelevant to query
   - Similarity scores inverted (low scores ranked first)
   - Empty results when matches should exist
   - Wrong granularity returned

4. **Chunking Failures:**
   - Headers missing or corrupted
   - Reconstruction produces different content
   - Code blocks split mid-function
   - Chunk boundaries corrupt sentences

5. **Performance Failures:**
   - Operations exceed 3× expected time
   - Timeouts occur
   - Memory leaks detected (increasing memory over time)
   - Database locks persist

6. **Security Failures:**
   - Agent A can access Agent B's private memories
   - Session isolation broken
   - Unauthorized data exposure in logs

### Partial Pass Criteria

Some tests may partially pass with caveats:

- **Performance Warning:** Operation succeeds but slower than expected
- **Minor Content Differences:** Reconstructed document has whitespace differences but content identical
- **Expected Empty Results:** Search returns no results due to legitimate absence of matching content
- **Graceful Degradation:** Feature disabled (e.g., embedding model unavailable) but error handled correctly

---

## 12. Test Execution Guide

### Before You Begin

1. **Verify Server Running:**
   ```bash
   tail -f /Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/logs/mcp_server.log
   ```
   Look for: "MCP SERVER STARTUP" and "Embedding model warmed up"

2. **Check Database Path:**
   ```bash
   ls -lh /Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db
   ```
   Verify database exists and is writable

3. **Clear Old Test Data (Optional):**
   - Use `cleanup_old_memories` with dry_run=false if needed
   - Or accept that test data will accumulate

### Execution Order

**Phase 1: Storage Functions (Tests 5.1-5.9)**
- Execute in order
- Record all memory_ids
- Verify chunks_created values

**Phase 2: Search Functions (Tests 6.1-6.10)**
- Depends on Phase 1
- Test each granularity level
- Validate semantic relevance

**Phase 3: Chunking Tests (Tests 7.1-7.5)**
- Depends on Phase 1
- Focus on content integrity
- Manual inspection recommended

**Phase 4: Utility Functions (Tests 8.1-8.8)**
- Can run anytime after Phase 1
- Test in order for dependencies

**Phase 5: Edge Cases (Tests 9.1-9.7)**
- Independent tests
- Can run in any order

**Phase 6: Integration (Tests 10.1-10.3)**
- Comprehensive workflows
- Run after all individual function tests pass

### Logging Best Practices

1. **Monitor Logs During Tests:**
   ```bash
   tail -f logs/mcp_server.log | grep -E "(TIMING|SLOW QUERY|ERROR|Iterative)"
   ```

2. **Key Log Patterns to Watch:**
   - `[TIMING]`  Operation timing
   - `SLOW QUERY`  Operations exceeding threshold
   - `Iterative search performance`  Search statistics
   - `Generated {N} chunk embeddings`  Chunking operations
   - `ERROR` or `CRITICAL`  Problems

3. **Save Log Snapshots:**
   - Before starting: `cp logs/mcp_server.log logs/before_test.log`
   - After completion: `cp logs/mcp_server.log logs/after_test.log`
   - Compare for anomalies

### Recording Results

For each test, record:

1. **Test ID:** (e.g., Test 5.1)
2. **Status:** PASS / FAIL / PARTIAL
3. **Memory IDs:** Any memory_ids created
4. **Timing:** Actual execution time
5. **Notes:** Observations, anomalies, deviations
6. **Log Excerpts:** Relevant log entries

**Example Record:**
```
Test 5.4: store_report (With Auto-Chunking)
Status: PASS
Memory ID: 42
Chunks Created: 18
Execution Time: 2.1s
Notes: Chunking worked correctly. All headers preserved.
Log: "Generated 18 chunk embeddings"
```

### Troubleshooting

**Problem: Embedding model timeout**
- Solution: Verify warmup in logs
- Check: Model loaded during server startup
- Workaround: First operation may take 7+ seconds

**Problem: Search returns no results**
- Verify: Data actually stored (check memory_id)
- Verify: Filters not too restrictive
- Check: Embedding model available

**Problem: Chunking creates too few/many chunks**
- Check: Content length (< 450 tokens ’ no chunking)
- Verify: Chunk size config (default 450 tokens)
- Review: Markdown structure (headers affect chunking)

**Problem: Reconstruction differs from original**
- Check: YAML frontmatter extraction
- Verify: Code blocks not split incorrectly
- Review: Whitespace handling at chunk boundaries

**Problem: "SLOW QUERY" warnings**
- Acceptable for: Large documents (> 20 chunks), first embedding generation
- Investigate if: Repeated slow queries, search > 2 seconds
- Check: Database locked, high system load

### Test Report Template

```markdown
# Vector Memory MCP Server Test Report

**Date:** 2025-01-15
**Tester:** [Name]
**Server Version:** 1.0.0
**Database:** agent_session_memory.db

## Summary

- Total Tests: 50
- Passed: 48
- Failed: 2
- Partial: 0

## Detailed Results

### Storage Functions
- Test 5.1: PASS (0.05s)
- Test 5.2: PASS (0.04s)
- Test 5.4: PASS (2.1s, 18 chunks)
- ...

### Search Functions
- Test 6.1: PASS (0.12s)
- Test 6.4: PASS (0.42s)
- ...

### Failed Tests

#### Test 9.2: Very Large Content
- Status: FAIL
- Issue: Timeout after 10 seconds
- Error: "Database locked"
- Root Cause: Large content (500K chars) exceeded processing capacity
- Recommendation: Implement streaming for very large documents

## Performance Summary

- Average Storage Time: 1.2s
- Average Search Time (FINE): 0.38s
- Average Search Time (MEDIUM): 0.91s
- Average Search Time (COARSE): 0.15s

## Recommendations

1. Optimize large document handling
2. Add streaming support for files > 100K chars
3. Improve database lock handling under load
```

---

## Appendix A: Quick Reference

### Storage Functions Cheat Sheet

| Function | Auto-Chunk | Fixed Agent | Session Required |
|----------|------------|-------------|------------------|
| store_session_context | No | main-orchestrator | Yes |
| store_input_prompt | No | main-orchestrator | Yes |
| store_system_memory | No | Any | Yes |
| store_report | Yes | Any | Yes |
| store_report_observation | No | Any | Yes |
| store_working_memory | Yes | Any | Yes |
| store_knowledge_base | Yes | Any | No |

### Search Functions Cheat Sheet

| Function | Vector Search | Granularity | Typical Limit |
|----------|---------------|-------------|---------------|
| search_session_context | No (scoped) | N/A | 5 |
| search_input_prompts | No (scoped) | N/A | 5 |
| search_system_memory | Yes | N/A | 10 |
| search_reports_specific_chunks | Yes | FINE | 10 |
| search_reports_section_context | Yes | MEDIUM | 5 |
| search_reports_full_documents | No (scoped) | COARSE | 3 |
| search_working_memory_* | Yes/No | FINE/MEDIUM/COARSE | 10/5/3 |
| search_knowledge_base_* | Yes/No | FINE/MEDIUM/COARSE | 10/5/3 |

### Expected Response Times

| Operation | Expected Time | Slow Threshold |
|-----------|---------------|----------------|
| store (no chunk) | < 100ms | 200ms |
| store (with chunk) | < 3s | 5s |
| search (FINE) | < 500ms | 1s |
| search (MEDIUM) | < 1500ms | 3s |
| search (COARSE) | < 200ms | 500ms |
| reconstruct | < 500ms | 1s |
| delete | < 100ms | 200ms |

---

## Appendix B: Sample Data Library

All sample data is provided in [Section 4: Test Data Samples](#4-test-data-samples).

Samples included:
- Sample 1: Small content (no chunking)
- Sample 2: Large multi-section document (15-20 chunks)
- Sample 3: Code-heavy content (2-4 chunks)
- Sample 4: YAML frontmatter content
- Sample 5: Minimal content (single line)
- Sample 6: Empty content (edge case)

---

## Appendix C: Log Analysis Guide

### Key Log Patterns

**Storage Operation:**
```
Memory stored successfully with ID: 42
Chunking document...
Generated 18 chunk embeddings
[TIMING] store_report: 2.150s
```

**Search Operation:**
```
FINE GRANULARITY SEARCH STARTING
Iterative search performance: elapsed=0.456s, fetched=500, kept=10, target=10
FINAL RESULTS: Total results: 10 chunks
[TIMING] vector_search_fine: 0.456s
```

**Slow Query Warning:**
```
SLOW QUERY: search_reports_specific_chunks took 2.145s (threshold: 2.0s)
```

**Semantic Search:**
```
Generated query embedding
Using iterative search to handle low selectivity filters
Fetched batch: offset=0, size=100, total_fetched=100
```

### Log Levels

- **DEBUG:** Detailed operation information
- **INFO:** Normal operation logs
- **WARNING:** Potential issues (slow queries, missing embeddings)
- **ERROR:** Operation failures
- **CRITICAL:** Server-level failures

---

## Conclusion

This test design document provides a comprehensive, step-by-step plan for validating all functionality of the Vector Memory MCP Server using only MCP tool calls. By following this plan systematically, a test execution agent can:

1. Verify all 28 MCP functions work correctly
2. Validate semantic search accuracy and relevance ranking
3. Confirm chunking preserves document structure
4. Test all three granularity levels (FINE, MEDIUM, COARSE)
5. Ensure memory isolation and security
6. Handle edge cases and error conditions gracefully
7. Validate performance under various workloads

**Next Steps:**
1. Execute tests in order (Phases 1-6)
2. Record results using provided template
3. Analyze logs for anomalies
4. Report failures with detailed context
5. Re-test after fixes
6. Maintain test data for regression testing

**Success Metric:** All tests pass with acceptable performance (< 3s for storage, < 500ms for search).

---

**End of Test Design Document**
