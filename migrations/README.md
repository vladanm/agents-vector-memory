# Database Migrations

## Overview

This directory contains database migration tools and scripts for the Vector Memory MCP server.

**IMPORTANT**: As of the latest version, **migrations are no longer needed for new installations**. The complete schema (including all granularity fields) is created automatically during database initialization in `SessionMemoryStore._init_database()`.

## When to Use Migrations

Migrations are **only needed** for:

1. **Legacy databases** - Databases created before the granularity features were added to the core schema
2. **Schema upgrades** - Future changes that need to be applied to existing databases

## New Installations (Recommended)

For new installations, simply initialize the database normally - no migration required:

```python
from src.session_memory_store import SessionMemoryStore

# Database is automatically created with complete schema
store = SessionMemoryStore(db_path="/path/to/memory.db")
```

The initialization automatically creates:
- All tables with complete column definitions
- All indexes for optimal performance
- Vector search tables for semantic queries

## Verifying Your Schema

To verify that your database has the complete schema, use the verification script:

```bash
python verify_schema.py /path/to/your/database.db
```

Expected output:
```
🔍 Verifying schema for: /path/to/your/database.db

📋 Checking session_memories table...
  ✅ All 20 columns present

📋 Checking memory_chunks table...
  ✅ All 25 columns present

📋 Checking indexes...
  ✅ All 11 indexes present

✅ Schema verification PASSED
```

## Upgrading Legacy Databases

If you have an existing database from an older version, you can either:

### Option 1: Automatic Migration (Recommended)

The improved migration runner will safely upgrade your schema:

```bash
python run_migration.py /path/to/database.db 001_add_granularity_metadata.sql
```

The migration runner:
- ✅ Skips columns that already exist (idempotent)
- ✅ Creates missing indexes automatically
- ✅ Provides detailed progress reporting
- ✅ Safely handles partial migrations

### Option 2: Fresh Database

For a clean start, create a new database and migrate your data:

```python
from src.session_memory_store import SessionMemoryStore
import sqlite3

# Create new database with complete schema
new_store = SessionMemoryStore(db_path="/path/to/new_memory.db")

# Copy data from old database (implement data migration as needed)
# ... your data migration logic ...
```

## Schema Comparison

### Legacy Schema (Pre-Granularity)
- **session_memories**: 16 columns
- **memory_chunks**: 14 columns
- **indexes**: 6 indexes

### Current Schema (With Granularity)
- **session_memories**: 20 columns (+4)
  - Added: `document_structure`, `document_summary`, `estimated_tokens`, `chunk_strategy`
- **memory_chunks**: 25 columns (+11)
  - Added: `parent_title`, `section_hierarchy`, `granularity_level`, `chunk_position_ratio`,
    `sibling_count`, `depth_level`, `contains_code`, `contains_table`, `keywords`,
    `original_content`, `is_contextually_enriched`
- **indexes**: 11 indexes (+5)
  - Added: `idx_memory_type_iter`, `idx_chunks_granularity`, `idx_chunks_section`,
    `idx_chunks_parent_title`, `idx_chunks_contains_code`

## Available Scripts

### `verify_schema.py`
Checks if your database has the complete schema.

**Usage**:
```bash
python verify_schema.py /path/to/database.db
```

**Output**: Reports missing columns, extra columns, and index status.

### `run_migration.py`
Applies SQL migration files to upgrade legacy databases.

**Usage**:
```bash
python run_migration.py /path/to/database.db 001_add_granularity_metadata.sql
```

**Features**:
- Idempotent execution (safe to re-run)
- Detailed progress reporting
- Error handling for duplicate columns/indexes
- Transaction rollback on failure

## Migration Files

### `001_add_granularity_metadata.sql`
Adds three-tier granularity search support to existing databases.

**What it does**:
- Adds 4 columns to `session_memories` table
- Adds 11 columns to `memory_chunks` table
- Creates 5 new performance indexes

**Status**: ✅ Complete and tested

## Troubleshooting

### "Column already exists" errors
This is normal for idempotent migrations. The script will skip existing columns and continue.

### "No such column" errors during migration
This indicates the migration file statements are executing out of order. Use the improved `run_migration.py` which properly parses SQL comments and statements.

### Partial migration
If a migration fails partway through:
1. Run `verify_schema.py` to see what's missing
2. Re-run the migration (it's idempotent)
3. Or manually apply missing columns using the SQL from the migration file

### Performance issues after migration
Run `ANALYZE` on your database to update query optimizer statistics:

```bash
sqlite3 /path/to/database.db "ANALYZE;"
```

## Best Practices

1. **Always backup** before running migrations
2. **Verify schema** after migrations with `verify_schema.py`
3. **Test queries** to ensure granularity search works correctly
4. **Monitor performance** - large databases may need `VACUUM` after migrations

## Future Migrations

When adding new migrations:

1. Update `SessionMemoryStore._init_database()` **first** with the new schema
2. Create migration SQL for upgrading existing databases
3. Test on a copy of a production database
4. Update this README with migration details
5. Bump version number and add changelog entry

This ensures new installations get the complete schema while existing databases can upgrade safely.
