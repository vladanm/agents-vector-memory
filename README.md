# Agent Session Memory MCP Server

A specialized MCP server designed specifically for agent session management with proper scoping, ordering, and task continuity support.

## Key Features

### 🎯 Session-Centric Design
- **Main Agent Support**: `session_context` and `system_memory` scoped by `session_id`/`session_iter`
- **Sub-Agent Support**: All memory types scoped by `agent_id` + `session_id` + optional(`session_iter`, `task_code`)
- **Proper Ordering**: Results returned in `session_iter DESC, created_at DESC` order

### 🧠 Memory Types
- **`session_context`**: Agent session snapshots for continuity across iterations
- **`input_prompt`**: Original user prompts to prevent loss during session compression
- **`reports`**: Agent-generated analysis and findings (usually MD files)
- **`working_memory`**: Important information during task execution (gotcha moments)
- **`system_memory`**: System configs, commands, scripts, endpoints, DB connections
- **`report_observations`**: Additional notes on existing reports

### 🔍 Advanced Search & Scoping
- Semantic search with vector embeddings
- Precise filtering by `agent_id`, `session_id`, `session_iter`, `task_code`
- Conditional loading for task continuity
- Proper session iteration ordering

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- Claude Desktop (for MCP integration)

### Quick Start

```bash
# Test the server manually
cd /path/to/vector-memory-2-mcp
uv run --script main.py --database-path ./test-memory.db

# Or use legacy working directory mode
python main.py --working-dir /path/to/your/project
```

### MCP Configuration for Claude Desktop

Add to your Claude Desktop config (`~/.config/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "run",
        "--script",
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
        "--script",
        "/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/main.py",
        "--database-path",
        "/Users/vladanm/projects/agents_evaluation/memory.db"
      ]
    }
  }
}
```

**After adding the configuration:**
1. Save the config file
2. Restart Claude Desktop completely
3. Look for the MCP tools icon (🔧) to verify connection

## Usage Examples

### Main Agent Session Context
```python
# Store session context for main agent
store_session_context(
    agent_id="main",
    session_id="session_123",
    content="Current analysis of user requirements...",
    session_iter=5,
    title="Requirements Analysis Context"
)

# Search for session context
search_session_context(
    agent_id="main", 
    session_id="session_123"
)
```

### Sub-Agent Reports
```python  
# Store agent report
store_report(
    agent_id="specialized-agent",
    session_id="session_123",
    content="## Analysis Report\n\nFound 15 issues...",
    task_code="code-analysis",
    session_iter=3,
    title="Code Analysis Report"
)

# Search reports for specific task
search_reports(
    agent_id="specialized-agent",
    session_id="session_123", 
    task_code="code-analysis"
)
```

### Task Continuity
```python
# Load context only if agent worked on this task before
load_session_context_for_task(
    agent_id="specialized-agent",
    session_id="session_123",
    current_task_code="security-audit"
)
```

## API Functions

### Storage Functions
- `store_session_context()` - Session snapshots for continuity
- `store_input_prompt()` - Original prompts to prevent loss
- `store_system_memory()` - System configs and commands
- `store_report()` - Agent-generated reports
- `store_report_observation()` - Additional notes on reports
- `store_working_memory()` - Important task execution info

### Search Functions  
- `search_session_context()` - Search session contexts with scoping
- `search_system_memory()` - Search system memory with scoping
- `search_reports()` - Search reports with scoping
- `search_working_memory()` - Search working memory with scoping
- `search_input_prompts()` - Search input prompts with scoping

### Utility Functions
- `load_session_context_for_task()` - Conditional loading for continuity
- `get_memory_by_id()` - Retrieve specific memory
- `get_session_stats()` - Session statistics
- `list_sessions()` - List recent sessions

## Database Schema

The system uses SQLite with sqlite-vec for vector search:

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
    updated_at TEXT NOT NULL,
    accessed_at TEXT,
    access_count INTEGER DEFAULT 0
);
```

Key indexes for efficient scoped searches:
- `idx_agent_session` - (agent_id, session_id)
- `idx_agent_session_iter` - (agent_id, session_id, session_iter) 
- `idx_agent_session_task` - (agent_id, session_id, task_code)

## Requirements vs. Current MCP

### ✅ What This Solves
- **Missing scoping fields**: Now has `agent_id`, `session_id`, `session_iter`, `task_code`
- **Search scoping impossible**: Now supports filtered searches by all scope fields
- **Ordering requirements**: Implements `session_iter DESC, created_at DESC` ordering
- **Missing memory types**: Adds `input_prompt` and proper session-centric types
- **Task continuity**: Implements conditional loading for task matching

### 🎯 Architecture Match
- **Session-based memory system** vs. flat document storage
- **Agent-centric interface** with proper scoping
- **Hierarchical session management** with iteration tracking

## Configuration

### Command Line Arguments

**Preferred Method:**
```bash
uv run --script main.py --database-path /path/to/database.db
```
- Direct path to SQLite database file
- Database auto-created if doesn't exist
- Single file, easy to backup/move
- **Recommended for MCP integration**

**Legacy Method:**
```bash
python main.py --working-dir /path/to/project
```
- Creates: `{working_dir}/memory/agent_session_memory.db`
- Maintains compatibility with older configs

### Environment Variables

- `EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)
- No other environment variables required

### Database Location Options

1. **Project-specific**: `/path/to/your-project/memory.db`
2. **Shared global**: `~/.vector-memory/global-memory.db`
3. **Evaluation/testing**: `/path/to/agents_evaluation/memory.db`

## Troubleshooting

### Server Not Connecting to Claude Desktop

1. **Check logs**: `tail -f ~/Library/Logs/Claude/*`
2. **Verify uv installation**: `which uv` and `uv --version`
3. **Test manually**:
   ```bash
   uv run --script /path/to/main.py --database-path ./test.db
   ```
4. **Use absolute paths** in Claude Desktop config (no `~` or relative paths)
5. **Restart Claude Desktop** completely after config changes

### Database Issues

- **Auto-creation**: Database file is created automatically if it doesn't exist
- **Permissions**: Ensure parent directory is writable
- **Check path**: Verify directory exists: `ls -la /path/to/`
- **Database schema**: Auto-initialized on first run

### uv Installation

If `uv` is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via homebrew:
```bash
brew install uv
```

## Security

- Input validation for all agent IDs, session IDs, task codes
- Content length limits and sanitization
- SQL injection prevention via parameterized queries
- Path traversal protection
- Content deduplication with SHA-256 hashing
- No external network access required