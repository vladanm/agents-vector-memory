# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-10-15

### Added

- **store_session_context**: Added optional `agent_id` parameter with default value `"main-orchestrator"`
  - Backward compatible: Existing code continues to work without changes
  - Enables sub-agents to maintain their own session contexts
  - Provides flexibility for complex multi-agent workflows

- **search_session_context**: Added optional `agent_id` parameter for filtering
  - Pass specific agent_id to filter by agent
  - Pass `None` (default) to search across all agents

### Changed

- Session context storage and search now support multi-agent scenarios
- Better consistency with other memory storage tools
- Updated docstrings to reflect new capabilities

### Backward Compatibility

- ✅ **100% Backward Compatible**: All existing code works without modification
- Default behavior unchanged: `agent_id="main-orchestrator"` when not specified
- No migration required

---

## [2.1.0] - 2025-10-15

### Breaking Changes

- **store_input_prompt**: Added required `agent_id` parameter as first argument
  - Removes hard-coded `"main-orchestrator"` agent_id
  - Enables proper multi-agent workflows with attributed input prompts
  - Migration required: Add `agent_id` parameter to all existing calls

### Added

- **search_input_prompts**: Added optional `agent_id` parameter for filtering
  - Pass specific agent_id to filter by agent
  - Pass `None` to search across all agents (default behavior)

### Changed

- Input prompts are now properly attributed to the agent that stored them
- Better consistency with other storage tools (store_system_memory, store_report, etc.)

### Migration

**Before (v2.0.0):**
```python
result = store_input_prompt(
    session_id="session-123",
    session_iter="v1",
    content="Analyze the code"
)
```

**After (v2.1.0):**
```python
result = store_input_prompt(
    agent_id="main-orchestrator",  # NEW: Required parameter
    session_id="session-123",
    session_iter="v1",
    content="Analyze the code"
)
```

For detailed migration guide, see [MIGRATION_v2.0_to_v2.1.md](./MIGRATION_v2.0_to_v2.1.md)

---

## [2.0.0] - 2025-XX-XX

### Added
- Initial release with vector-based session memory
- Multi-agent support with agent_id scoping
- Semantic search with sqlite-vec
- Session context management
- Report and working memory storage
- Knowledge base support

---
