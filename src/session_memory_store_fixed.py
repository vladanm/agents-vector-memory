# Fixed search_with_granularity implementation
# This shows the CORRECT approach where granularity controls expansion behavior, not filtering

def search_with_granularity_FIXED(
    self,
    query: str,
    memory_type: str,
    granularity: str,
    agent_id: str = None,
    session_id: str = None,
    session_iter: int = None,
    task_code: str = None,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    auto_merge_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Three-tier granularity search with expansion behavior control.

    CORRECT DESIGN:
    - granularity controls WHAT IS RETURNED (expansion behavior)
    - NOT what is filtered/searched

    Behaviors:
    - 'fine': Returns matching chunks AS-IS (no expansion)
    - 'medium': Returns matching chunks + their siblings (section expansion)
    - 'coarse': Returns complete reconstructed documents

    Args:
        query: Semantic search query
        memory_type: Memory type (knowledge_base or reports)
        granularity: 'fine', 'medium', or 'coarse'
        limit: Maximum results
        similarity_threshold: Minimum similarity score (0.0-1.0)
        auto_merge_threshold: For medium, merge if ≥60% siblings match

    Returns:
        Dict with search results at specified granularity level
    """
    try:
        valid_granularities = ['fine', 'medium', 'coarse']
        if granularity not in valid_granularities:
            return {
                "success": False,
                "error": f"Invalid granularity. Must be one of: {', '.join(valid_granularities)}"
            }

        conn = self._get_connection()
        query_embedding = self.embedding_model.encode([query.strip()])[0]

        # Build WHERE conditions for filters
        where_conditions = ["m.memory_type = ?"]
        params = [memory_type]

        if agent_id:
            where_conditions.append("m.agent_id = ?")
            params.append(agent_id)

        if session_id:
            where_conditions.append("m.session_id = ?")
            params.append(session_id)

        if session_iter is not None:
            where_conditions.append("m.session_iter = ?")
            params.append(session_iter)

        if task_code:
            where_conditions.append("m.task_code = ?")
            params.append(task_code)

        where_clause_main = "AND " + " AND ".join(where_conditions)
        distance_threshold = 2.0 - similarity_threshold

        # ========================================
        # COARSE GRANULARITY: Full Documents
        # ========================================
        if granularity == 'coarse':
            # Search document embeddings, return full documents
            vector_query = f"""
                SELECT m.*, v.distance
                FROM session_memories m
                JOIN (
                    SELECT memory_id, distance
                    FROM vec_session_search
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance ASC
                ) v ON m.id = v.memory_id
                WHERE v.distance < ?
                {where_clause_main}
                ORDER BY v.distance ASC
                LIMIT ?
            """
            final_params = [query_embedding.tobytes(), limit * 2, distance_threshold] + params + [limit]

            rows = conn.execute(vector_query, final_params).fetchall()

            # Return full documents
            results = []
            for row in rows:
                result = {
                    "memory_id": row[0],
                    "memory_type": row[1],
                    "agent_id": row[2],
                    "session_id": row[3],
                    "session_iter": row[4],
                    "task_code": row[5],
                    "content": row[6],  # Full document content
                    "title": row[7],
                    "description": row[8],
                    "tags": json.loads(row[9]) if row[9] else [],
                    "metadata": json.loads(row[10]) if row[10] else {},
                    "content_hash": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                    "accessed_at": row[14],
                    "access_count": row[15],
                    "similarity": max(0.0, 2.0 - row[16]) if row[16] is not None else 0.0,
                    "source_type": "document",
                    "granularity": "coarse"
                }
                results.append(result)

            conn.close()
            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "granularity": granularity
            }

        # ========================================
        # FINE & MEDIUM GRANULARITY: Chunk Search
        # ========================================
        # Search chunk embeddings (NO filtering by granularity_level!)
        # ⚠️ CRITICAL: Removed granularity_level filter - we search ALL chunks
        vector_query = f"""
            SELECT
                m.id, m.memory_type, m.agent_id, m.session_id,
                m.session_iter, m.task_code,
                m.title, m.description, m.tags, m.metadata,
                m.content_hash, m.created_at, m.updated_at,
                m.accessed_at, m.access_count,
                v.distance, mc.chunk_index, mc.id as chunk_id,
                COALESCE(mc.original_content, mc.content) as chunk_content,
                mc.header_path, mc.token_count, mc.sibling_count
            FROM session_memories m
            JOIN memory_chunks mc ON m.id = mc.parent_id
            JOIN (
                SELECT chunk_id, distance
                FROM vec_chunk_search
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
            ) v ON mc.id = v.chunk_id
            WHERE v.distance < ?
            {where_clause_main}
            ORDER BY v.distance ASC
            LIMIT ?
        """
        final_params = [query_embedding.tobytes(), limit * 3, distance_threshold] + params + [limit * 2]

        rows = conn.execute(vector_query, final_params).fetchall()

        # Format matched chunks
        matched_chunks = []
        for row in rows:
            chunk = {
                "memory_id": row[0],
                "memory_type": row[1],
                "agent_id": row[2],
                "session_id": row[3],
                "session_iter": row[4],
                "task_code": row[5],
                "title": row[6],
                "description": row[7],
                "tags": json.loads(row[8]) if row[8] else [],
                "metadata": json.loads(row[9]) if row[9] else {},
                "content_hash": row[10],
                "created_at": row[11],
                "updated_at": row[12],
                "accessed_at": row[13],
                "access_count": row[14],
                "similarity": max(0.0, 2.0 - row[15]) if row[15] is not None else 0.0,
                "chunk_index": row[16],
                "chunk_id": row[17],
                "chunk_content": row[18],
                "header_path": row[19],
                "token_count": row[20],
                "sibling_count": row[21],
                "source_type": "chunk",
                "granularity": granularity
            }
            matched_chunks.append(chunk)

        # ========================================
        # FINE GRANULARITY: Return chunks as-is
        # ========================================
        if granularity == 'fine':
            results = []
            for chunk in matched_chunks[:limit]:
                result = chunk.copy()
                # Set content to the actual chunk content (not full document!)
                result["content"] = chunk["chunk_content"]
                results.append(result)

            conn.close()
            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "granularity": "fine"
            }

        # ========================================
        # MEDIUM GRANULARITY: Expand to sections
        # ========================================
        elif granularity == 'medium':
            results = self._expand_to_sections_FIXED(matched_chunks, auto_merge_threshold, conn, limit)

            conn.close()
            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query,
                "granularity": "medium"
            }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": "Granularity search failed",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


def _expand_to_sections_FIXED(
    self,
    matched_chunks: List[Dict],
    merge_threshold: float,
    conn,
    limit: int
) -> List[Dict]:
    """
    Expand matched chunks to include their siblings (section-level context).

    Logic:
    1. Group matched chunks by (parent_id, header_path)
    2. For each section, fetch ALL sibling chunks
    3. Merge all siblings into section-level content
    4. Return expanded sections (limit applied)
    """
    # Group matched chunks by section
    sections = {}
    for chunk in matched_chunks:
        parent_id = chunk["memory_id"]
        section_path = chunk.get("header_path", "")
        key = (parent_id, section_path)

        if key not in sections:
            sections[key] = {
                "matched_chunks": [],
                "representative": chunk  # Keep one chunk as template
            }
        sections[key]["matched_chunks"].append(chunk)

    # Expand each section to include all siblings
    expanded_results = []
    for (parent_id, section_path), section_data in sections.items():
        # Fetch ALL chunks from this section
        all_section_chunks = conn.execute("""
            SELECT chunk_index, COALESCE(original_content, content) as content, token_count
            FROM memory_chunks
            WHERE parent_id = ? AND header_path = ?
            ORDER BY chunk_index ASC
        """, (parent_id, section_path)).fetchall()

        if not all_section_chunks:
            continue

        # Merge all sibling chunks
        merged_content = "\n\n".join([chunk[1] for chunk in all_section_chunks])
        total_tokens = sum([chunk[2] for chunk in all_section_chunks])

        # Create expanded result
        representative = section_data["representative"]
        result = {
            "memory_id": parent_id,
            "memory_type": representative["memory_type"],
            "agent_id": representative["agent_id"],
            "session_id": representative["session_id"],
            "session_iter": representative["session_iter"],
            "task_code": representative["task_code"],
            "title": representative["title"],
            "description": representative["description"],
            "tags": representative["tags"],
            "metadata": representative["metadata"],
            "content_hash": representative["content_hash"],
            "created_at": representative["created_at"],
            "updated_at": representative["updated_at"],
            "accessed_at": representative["accessed_at"],
            "access_count": representative["access_count"],
            "similarity": representative["similarity"],
            "content": merged_content,  # Expanded section content
            "source_type": "expanded_section",
            "granularity": "medium",
            "section_path": section_path,
            "matched_chunk_count": len(section_data["matched_chunks"]),
            "total_chunk_count": len(all_section_chunks),
            "merged": True,
            "token_count": total_tokens
        }
        expanded_results.append(result)

    # Sort by similarity and limit
    expanded_results.sort(key=lambda x: x["similarity"], reverse=True)
    return expanded_results[:limit]
