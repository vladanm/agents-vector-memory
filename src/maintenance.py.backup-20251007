"""
Maintenance Operations
=====================

Statistics, health checks, and database maintenance.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MaintenanceOperations:
    """Maintenance operations for session memories."""

    def __init__(self, store):
        """Initialize with reference to parent store."""
        self.store = store

    def get_session_stats(
        self,
        agent_id: str | None = None,
        session_id: str | None = None
    ) -> dict[str, Any]:
        """Get statistics about session memory usage."""
        return self.store._get_session_stats_impl(agent_id, session_id)

    def list_sessions(
        self,
        agent_id: str | None = None,
        limit: int = 20
    ) -> dict[str, Any]:
        """List recent sessions with basic info."""
        return self.store._list_sessions_impl(agent_id, limit)

    def cleanup_old_memories(
        self,
        days_old: int = 90,
        memory_type: str | None = None
    ) -> dict[str, Any]:
        """Cleanup old memories (placeholder for future implementation)."""
        return {
            "success": False,
            "error": "Not implemented",
            "message": "Cleanup functionality not yet implemented"
        }

    def vacuum_database(self) -> dict[str, Any]:
        """
        Optimize database storage with VACUUM operation.

        VACUUM rebuilds the database file, reclaiming unused space and
        optimizing vec0 vector indexes. This operation can be slow on
        large databases and requires disk space equal to the database size.

        Returns:
            dict: Operation result with success status and details
        """
        try:
            conn = self.store._get_connection()

            # Get database size before VACUUM
            size_before = conn.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            ).fetchone()[0]

            # Execute VACUUM (reclaims space, optimizes indexes including vec0)
            logger.info("Starting VACUUM operation...")
            conn.execute("VACUUM")
            conn.commit()

            # Get database size after VACUUM
            size_after = conn.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            ).fetchone()[0]

            space_reclaimed = size_before - size_after

            conn.close()

            logger.info(f"VACUUM completed. Space reclaimed: {space_reclaimed} bytes")

            return {
                "success": True,
                "operation": "VACUUM",
                "size_before_bytes": size_before,
                "size_after_bytes": size_after,
                "space_reclaimed_bytes": space_reclaimed,
                "message": f"VACUUM completed successfully. Reclaimed {space_reclaimed} bytes."
            }

        except Exception as e:
            logger.error(f"VACUUM operation failed: {e}")
            return {
                "success": False,
                "operation": "VACUUM",
                "error": str(e),
                "message": f"VACUUM failed: {e}"
            }

    def analyze_database(self) -> dict[str, Any]:
        """
        Update query planner statistics with ANALYZE operation.

        ANALYZE gathers statistics about table and index content to help
        the query planner choose optimal query plans. This includes statistics
        for vec0 vector indexes, improving vector search performance.

        This operation is generally fast and should be run periodically,
        especially after large batch inserts or deletes.

        Returns:
            dict: Operation result with success status and details
        """
        try:
            conn = self.store._get_connection()

            # Execute ANALYZE (updates query planner statistics)
            logger.info("Starting ANALYZE operation...")
            conn.execute("ANALYZE")
            conn.commit()

            # Get statistics about analyzed tables
            stats = conn.execute("""
                SELECT COUNT(*) as table_count
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """).fetchone()

            table_count = stats[0] if stats else 0

            conn.close()

            logger.info(f"ANALYZE completed for {table_count} tables")

            return {
                "success": True,
                "operation": "ANALYZE",
                "tables_analyzed": table_count,
                "message": f"ANALYZE completed successfully for {table_count} tables."
            }

        except Exception as e:
            logger.error(f"ANALYZE operation failed: {e}")
            return {
                "success": False,
                "operation": "ANALYZE",
                "error": str(e),
                "message": f"ANALYZE failed: {e}"
            }

    def optimize_database(self) -> dict[str, Any]:
        """
        Perform full database optimization (VACUUM + ANALYZE).

        This is a convenience method that runs both VACUUM and ANALYZE
        operations sequentially for complete database optimization.

        Note: VACUUM can be slow on large databases and requires disk space.
        Consider running during maintenance windows.

        Returns:
            dict: Combined operation results
        """
        logger.info("Starting full database optimization (VACUUM + ANALYZE)...")

        vacuum_result = self.vacuum_database()
        analyze_result = self.analyze_database()

        success = vacuum_result.get("success", False) and analyze_result.get("success", False)

        return {
            "success": success,
            "operation": "OPTIMIZE (VACUUM + ANALYZE)",
            "vacuum_result": vacuum_result,
            "analyze_result": analyze_result,
            "message": "Full optimization completed." if success else "Optimization had errors."
        }
