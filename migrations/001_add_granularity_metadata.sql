-- Migration: Add Granularity and Context Metadata
-- Version: 001
-- Date: 2025-01-30
-- Description: Adds metadata fields for three-tier granularity search and contextual enrichment

-- ============================================
-- EXTEND session_memories TABLE
-- ============================================

-- Document structure and summary
ALTER TABLE session_memories ADD COLUMN document_structure TEXT DEFAULT NULL;
ALTER TABLE session_memories ADD COLUMN document_summary TEXT DEFAULT NULL;
ALTER TABLE session_memories ADD COLUMN estimated_tokens INTEGER DEFAULT 0;
ALTER TABLE session_memories ADD COLUMN chunk_strategy TEXT DEFAULT 'hierarchical';

-- ============================================
-- EXTEND memory_chunks TABLE
-- ============================================

-- Core context fields
ALTER TABLE memory_chunks ADD COLUMN parent_title TEXT DEFAULT NULL;
ALTER TABLE memory_chunks ADD COLUMN section_hierarchy TEXT DEFAULT NULL;
ALTER TABLE memory_chunks ADD COLUMN granularity_level TEXT DEFAULT 'medium';

-- Position and relationship fields
ALTER TABLE memory_chunks ADD COLUMN chunk_position_ratio REAL DEFAULT 0.5;
ALTER TABLE memory_chunks ADD COLUMN sibling_count INTEGER DEFAULT 1;
ALTER TABLE memory_chunks ADD COLUMN depth_level INTEGER DEFAULT 0;

-- Content type indicators
ALTER TABLE memory_chunks ADD COLUMN contains_code BOOLEAN DEFAULT 0;
ALTER TABLE memory_chunks ADD COLUMN contains_table BOOLEAN DEFAULT 0;
ALTER TABLE memory_chunks ADD COLUMN keywords TEXT DEFAULT '[]';

-- Original content (for display without context header)
ALTER TABLE memory_chunks ADD COLUMN original_content TEXT DEFAULT NULL;
ALTER TABLE memory_chunks ADD COLUMN is_contextually_enriched BOOLEAN DEFAULT 0;

-- ============================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX IF NOT EXISTS idx_chunks_granularity ON memory_chunks(granularity_level);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON memory_chunks(section_hierarchy);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_title ON memory_chunks(parent_title);
CREATE INDEX IF NOT EXISTS idx_chunks_contains_code ON memory_chunks(contains_code);
CREATE INDEX IF NOT EXISTS idx_memory_type_iter ON session_memories(memory_type, session_iter);

-- ============================================
-- Note: UPDATE EXISTING DATA
-- ============================================
-- These updates will be handled by a separate Python script
-- after the schema migration is complete to avoid column reference issues
-- during the migration transaction.
