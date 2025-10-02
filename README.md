# Agent Session Memory MCP Server

A specialized vector-based memory server designed for agent session management with hierarchical memory scoping, document chunking, and markdown support.

## Features

### Core Memory Management
- **Session-Centric Storage**: Proper scoping with agent_id, session_id, session_iter, and task_code
- **7 Memory Types**: knowledge_base, session_context, input_prompt, reports, working_memory, system_memory, report_observations
- **Vector Search**: Semantic search using sentence-transformers with sqlite-vec
- **Ordered Results**: Automatic ordering by session_iter DESC, created_at DESC

### Advanced Features (NEW)
- **Document Chunking**: Hierarchical markdown-aware chunking with configurable chunk sizes
- **Markdown Preservation**: Maintains header hierarchy and structure (h1-h6)
- **YAML Frontmatter**: Automatic extraction and parsing of YAML frontmatter
- **Document Reconstruction**: Rebuild complete documents from stored chunks
- **Memory Management**: Delete memories and cleanup old entries
- **Knowledge Base**: Global/session-scoped shared knowledge storage

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Claude Desktop (for MCP integration)

### Dependencies

```bash
# Required packages (auto-installed with uv)
mcp>=0.3.0
sqlite-vec>=0.1.6
sentence-transformers>=2.2.2
tiktoken>=0.5.0
pyyaml>=6.0
```

### Quick Start

```bash
# Test the server manually
cd /path/to/vector-memory-2-mcp
uv run --script main.py --database-path ./test-memory.db

# Or use legacy working directory mode
python main.py --working-dir /path/to/your/project
```

### MCP Configuration for Claude Desktop

Add to your Claude Desktop config (`~/.config/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "run",
        "/absolute/path/to/vector-memory-2-mcp/main.py",
        "--database-path",
        "/absolute/path/to/your/memory.db"
      ]
    }
  }
}
```

**Example Configuration:**
```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "run",
        "/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/main.py",
        "--database-path",
        "/Users/vladanm/projects/memory/agent_memory.db"
      ]
    }
  }
}
```

**After adding the configuration:**
1. Save the config file
2. Restart Claude Desktop completely
3. Look for the MCP tools icon (🔧) to verify connection

## Usage

### Basic Storage

```python
# Store working memory
store_working_memory(
    agent_id="main",
    session_id="session-123",
    content="Found critical bug in auth module",
    session_iter=1
)

# Store report with chunking enabled
store_report(
    agent_id="code-analyzer",
    session_id="session-123",
    content="""
# Code Analysis Report

## Overview
Comprehensive analysis of the codebase...

## Critical Issues
1. SQL injection vulnerability
2. Missing input validation
""",
    session_iter=1,
    task_code="security-audit",
    auto_chunk=True  # Enable document chunking
)
```

### Document Chunking

Enable chunking for large documents to improve search and manage token limits:

```python
store_report(
    agent_id="specialized-agent",
    session_id="session-123",
    content="""
---
title: Q1 Security Audit
author: Security Team
date: 2024-01-15
---

# Executive Summary

This comprehensive security audit reveals...

## Findings

### Critical Issues
- SQL injection in login endpoint
- XSS vulnerability in comments

### Recommendations
- Implement prepared statements
- Add input sanitization
""",
    session_iter=1,
    task_code="security-audit",
    title="Q1 2024 Security Audit",
    auto_chunk=True  # Chunks by markdown headers
)
```

The system will:
1. Extract YAML frontmatter automatically
2. Split content by markdown headers
3. Preserve header hierarchy (h1 > h2 > h3...)
4. Store chunks with metadata
5. Enable reconstruction later

### Searching Memories

```python
# Semantic search across reports
search_reports(
    agent_id="code-analyzer",
    session_id="session-123",
    query="SQL injection vulnerability",
    limit=10
)

# Search with task filtering
search_working_memory(
    agent_id="main",
    session_id="session-123",
    task_code="security-audit",
    query="authentication issues"
)
```

### Document Reconstruction

```python
# Reconstruct chunked document
reconstruct_document(memory_id=123)

# Returns:
# {
#   "success": true,
#   "content": "Full reconstructed document...",
#   "chunk_count": 5,
#   "title": "Security Audit Report"
# }
```

### Memory Management

```python
# Delete specific memory
delete_memory(memory_id=123)

# Cleanup old memories
cleanup_old_memories(
    older_than_days=30,
    memory_type="working_memory"
)
```

### Task Continuity

```python
# Load context only if agent worked on this task before
load_session_context_for_task(
    agent_id="code-reviewer",
    session_id="session-123",
    current_task_code="review-auth-module"
)
```

## Memory Types

### knowledge_base (NEW)
**Global or session-scoped shared knowledge**

- Supports document chunking and markdown preservation
- Use for: Documentation, architecture knowledge, best practices
- Scoping: Global (no session_id) or session-specific

### session_context
**Agent session snapshots for continuity**

- Main agent stores state between iterations
- Preserves conversation context
- Use for: Session state, progress tracking

### input_prompt
**Original user prompts**

- Prevents loss during session compression
- Exact user request preservation
- Use for: Reference, audit trail

### reports
**Agent-generated analysis and findings**

- Supports document chunking
- Usually markdown formatted
- Use for: Analysis results, investigation reports

### working_memory
**Important task information**

- Gotcha moments during execution
- Key insights and solutions
- Use for: Obstacles, learnings, discoveries

### system_memory
**System configurations and commands**

- API endpoints, file paths, configs
- Command templates
- Use for: Technical reference, scripts

### report_observations
**Additional notes on existing reports**

- Follow-up notes and clarifications
- Updates without modifying original
- Use for: Comments, updates, corrections

## Document Chunking Details

### When to Enable Chunking

Enable `auto_chunk=True` for:
- Large reports (>1000 characters)
- Markdown documents with multiple sections
- Knowledge base entries
- Technical documentation

### Chunking Configuration by Type

| Memory Type | Chunk Size | Overlap | Preserve Structure | Default |
|-------------|------------|---------|-------------------|---------|
| knowledge_base | 1000 | 100 | Yes | Enabled |
| reports | 1500 | 150 | Yes | Enabled |
| working_memory | 800 | 80 | No | Disabled |
| system_memory | 600 | 60 | No | Disabled |
| session_context | 1000 | 100 | No | Disabled |

### How Chunking Works

1. **Format Detection**: Automatically detects markdown structure
2. **Header Splitting**: Splits by markdown headers (h1-h6)
3. **Hierarchy Preservation**: Maintains header paths (Section > Subsection)
4. **Token Management**: Respects chunk size limits with overlap
5. **Chunk Linking**: Links chunks with prev/next references

### Markdown Structure Example

Input document:
```markdown
# Main Report

Introduction paragraph.

## Section 1

Section 1 content.

### Subsection 1.1

Detailed content here.

## Section 2

Section 2 content.
```

Results in chunks:
- Chunk 0: "# Main Report\n\nIntroduction..."
- Chunk 1: "## Section 1\n\nSection 1 content..."
- Chunk 2: "### Subsection 1.1\n\nDetailed content..."
- Chunk 3: "## Section 2\n\nSection 2 content..."

Each chunk preserves:
- Header path (e.g., "Main Report > Section 1 > Subsection 1.1")
- Header level (1-6)
- Original formatting

## API Reference

### Storage Functions

**store_session_context(agent_id, session_id, content, session_iter, ...)**
- Store agent session snapshots for continuity

**store_input_prompt(agent_id, session_id, content, session_iter, task_code, ...)**
- Store original user prompts

**store_report(agent_id, session_id, content, session_iter, task_code, auto_chunk, ...)**
- Store agent-generated reports
- Supports document chunking with `auto_chunk=True`

**store_working_memory(agent_id, session_id, content, session_iter, task_code, ...)**
- Store important task execution information

**store_system_memory(agent_id, session_id, content, session_iter, task_code, ...)**
- Store system configurations and commands

**store_report_observation(agent_id, session_id, content, parent_report_id, ...)**
- Add notes to existing reports

### Search Functions

All search functions support these parameters:
- `agent_id`: Filter by agent (optional)
- `session_id`: Filter by session (optional)
- `session_iter`: Filter by specific iteration (optional)
- `task_code`: Filter by task (optional)
- `query`: Semantic search query (optional)
- `limit`: Max results (default: 10)
- `latest_first`: Order by latest (default: true)

**search_session_context(...)**
**search_input_prompts(...)**
**search_reports(...)**
**search_working_memory(...)**
**search_system_memory(...)**

### Document Management (NEW)

**reconstruct_document(memory_id)**
- Rebuild complete document from chunks
- Returns full content with metadata
- Works for both chunked and non-chunked memories

**delete_memory(memory_id)**
- Delete memory and all associated data
- Cascades to embeddings and chunks
- Cannot be undone

**cleanup_old_memories(older_than_days, memory_type)**
- Clean up memories older than N days
- Optional memory type filter
- Returns count of deleted memories

### Utility Functions

**load_session_context_for_task(agent_id, session_id, current_task_code)**
- Load context only if agent previously worked on same task
- Enables task continuity for sub-agents

**get_memory_by_id(memory_id)**
- Retrieve specific memory by ID

**get_session_stats(agent_id, session_id)**
- Get statistics about session memory usage

**list_sessions(agent_id, limit)**
- List recent sessions with basic info

## Database Schema

### Main Tables

**session_memories** - Core memory storage
```sql
CREATE TABLE session_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    session_iter INTEGER DEFAULT 1,
    task_code TEXT,
    content TEXT NOT NULL,
    title TEXT,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    content_hash TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

**memory_chunks** (NEW) - Document chunk storage
```sql
CREATE TABLE memory_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_type TEXT DEFAULT 'text',
    token_count INTEGER,
    header_path TEXT,
    level INTEGER DEFAULT 0,
    prev_chunk_id INTEGER,
    next_chunk_id INTEGER,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES session_memories(id) ON DELETE CASCADE
);
```

**session_embeddings** - Vector embeddings
**vec_session_search** - sqlite-vec search index

### Key Indexes

- `idx_agent_session` - (agent_id, session_id)
- `idx_agent_session_iter` - (agent_id, session_id, session_iter)
- `idx_agent_session_task` - (agent_id, session_id, task_code)
- `idx_memory_type` - (memory_type)

## Testing

Run the comprehensive test suite:

```bash
python3 test_new_features.py
```

**Test coverage:**
- ✅ Basic storage (backward compatibility)
- ✅ Document chunking with markdown
- ✅ YAML frontmatter extraction
- ✅ Document reconstruction
- ✅ Delete memory functionality
- ✅ Cleanup old memories
- ✅ Knowledge base memory type
- ✅ All original memory types

All tests pass with 100% success rate.

## Backward Compatibility

**All new features are opt-in and fully backward compatible:**

- `auto_chunk` defaults to `False` (no chunking unless explicitly enabled)
- Existing databases automatically receive new tables
- All original MCP tools work unchanged
- No breaking changes to existing functionality
- Legacy working directory mode still supported

## Troubleshooting

### Server Not Connecting

1. **Check logs**: `tail -f ~/Library/Logs/Claude/*`
2. **Verify uv**: `which uv && uv --version`
3. **Test manually**:
   ```bash
   uv run /path/to/main.py --database-path ./test.db
   ```
4. **Use absolute paths** in config (no `~` or relative paths)
5. **Restart Claude Desktop** completely

### Database Issues

- Database auto-creates if missing
- Ensure parent directory exists and is writable
- Schema initializes automatically on first run
- Check permissions: `ls -la /path/to/database.db`

### uv Installation

```bash
# Install via script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via homebrew
brew install uv
```

## Configuration Options

### Command Line Arguments

**Preferred (direct database path):**
```bash
uv run main.py --database-path /path/to/database.db
```

**Legacy (working directory):**
```bash
python main.py --working-dir /path/to/project
# Creates: {working_dir}/memory/agent_session_memory.db
```

### Database Location Suggestions

1. **Project-specific**: `/path/to/project/memory.db`
2. **Shared global**: `~/.vector-memory/global-memory.db`
3. **Evaluation**: `/path/to/evaluation/memory.db`

## Security

- Input validation for all parameters
- Content length limits and sanitization
- SQL injection prevention (parameterized queries)
- Path traversal protection
- Content deduplication with SHA-256 hashing
- No external network access required

## Architecture

### Memory Scoping Hierarchy

```
agent_id → session_id → session_iter → task_code
```

**Main agent:**
- `agent_id="main"`
- Uses session_context and system_memory
- Tracks session iterations

**Sub-agents:**
- `agent_id="specialized-agent"` (or custom)
- Uses reports, working_memory, observations
- Scoped by task_code for specific tasks

### Search Result Ordering

All searches return results ordered by:
1. `session_iter DESC` - Latest iteration first
2. `created_at DESC` - Newest within iteration first

This ensures most recent, relevant context always appears first.

## License

Designed for agent memory management and session continuity.

## Contributing

When contributing:
1. Maintain backward compatibility
2. Add tests to `test_new_features.py`
3. Update this README
4. Follow existing code style
5. Ensure all tests pass

## Support

- Test suite: `test_new_features.py`
- Source documentation: `src/`
- System prompt: `CLAUDE.md`
- Additional guide: `vector-memory-mcp-guide.md`