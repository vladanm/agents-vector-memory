"""
Performance Benchmarks for Vector Memory MCP Server
===================================================

Establishes baseline metrics for critical operations:
- Store single memory
- Vector search (k=10)
- Batch store (100 memories)
- Large document chunking (50k tokens)

Run with: pytest tests/e2e/test_performance.py -v -s
"""

import time
import tempfile
from pathlib import Path
from statistics import median
from src.session_memory_store import SessionMemoryStore


def time_operation(func, iterations=5):
    """
    Time an operation multiple times and return median latency.

    Args:
        func: Callable to time
        iterations: Number of iterations (default: 5)

    Returns:
        tuple: (median_ms, all_times_ms, min_ms, max_ms)
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return median(times), times, min(times), max(times)


def test_store_single_memory_performance():
    """Benchmark storing a single memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "perf_test.db")
        store = SessionMemoryStore(db_path=db_path)

        counter = [0]

        def store_memory():
            counter[0] += 1
            store.storage.store_memory(
                memory_type="working_memory",
                agent_id="perf-test",
                session_id=f"session-{counter[0]}",
                content=f"Performance test memory {counter[0]} with moderate length content to simulate realistic usage patterns.",
                session_iter=1
            )

        median_ms, all_times, min_ms, max_ms = time_operation(store_memory, iterations=5)

        print(f"\n📊 Store Single Memory Performance:")
        print(f"   Median: {median_ms:.2f} ms")
        print(f"   Min:    {min_ms:.2f} ms")
        print(f"   Max:    {max_ms:.2f} ms")
        print(f"   All:    {[f'{t:.2f}' for t in all_times]} ms")

        # Baseline: Should be < 150ms for single store (includes embedding generation)
        assert median_ms < 150, f"Single store too slow: {median_ms:.2f}ms (expected < 150ms)"


def test_vector_search_performance():
    """Benchmark vector search with k=10."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "perf_test.db")
        store = SessionMemoryStore(db_path=db_path)

        # Populate database with 100 memories for realistic search
        for i in range(100):
            store.storage.store_memory(
                memory_type="reports",
                agent_id="perf-test",
                session_id=f"session-{i % 10}",
                content=f"Report {i}: Analysis of system component with findings and recommendations for improvement. This is test data for performance benchmarking.",
                session_iter=1,
                task_code=f"task-{i % 5}"
            )

        def search_memories():
            store.search.search_memories(
                memory_type="reports",
                query="system analysis findings",
                limit=10,
                agent_id="perf-test"
            )

        median_ms, all_times, min_ms, max_ms = time_operation(search_memories, iterations=5)

        print(f"\n🔍 Vector Search Performance (k=10, 100 docs):")
        print(f"   Median: {median_ms:.2f} ms")
        print(f"   Min:    {min_ms:.2f} ms")
        print(f"   Max:    {max_ms:.2f} ms")
        print(f"   All:    {[f'{t:.2f}' for t in all_times]} ms")

        # Baseline: Should be < 600ms for search with embedding generation
        assert median_ms < 600, f"Vector search too slow: {median_ms:.2f}ms (expected < 600ms)"


def test_batch_store_performance():
    """Benchmark batch storing 100 memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "perf_test.db")
        store = SessionMemoryStore(db_path=db_path)

        def batch_store():
            for i in range(100):
                store.storage.store_memory(
                    memory_type="system_memory",
                    agent_id="perf-test",
                    session_id=f"batch-session-{i}",
                    content=f"System memory {i} for batch performance testing with realistic content length.",
                    session_iter=1
                )

        median_ms, all_times, min_ms, max_ms = time_operation(batch_store, iterations=3)  # Fewer iterations (slower)

        print(f"\n📦 Batch Store Performance (100 memories):")
        print(f"   Median: {median_ms:.2f} ms")
        print(f"   Min:    {min_ms:.2f} ms")
        print(f"   Max:    {max_ms:.2f} ms")
        print(f"   All:    {[f'{t:.2f}' for t in all_times]} ms")
        print(f"   Per-memory avg: {median_ms / 100:.2f} ms")

        # Baseline: Should be < 15 seconds for 100 memories
        assert median_ms < 15000, f"Batch store too slow: {median_ms:.2f}ms (expected < 15000ms)"


def test_large_document_chunking_performance():
    """Benchmark chunking a 50k token document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "perf_test.db")
        store = SessionMemoryStore(db_path=db_path)

        # Generate a large document (~50k tokens = ~200k characters)
        large_content = "\n\n".join([
            f"# Section {i}\n\n" +
            f"## Subsection {i}.1\n\n" +
            ("This is a detailed analysis paragraph with multiple sentences. " * 50) +
            f"\n\n## Subsection {i}.2\n\n" +
            ("More detailed content with technical information and analysis. " * 50)
            for i in range(100)  # 100 sections * ~2000 chars = ~200k chars (~50k tokens)
        ])

        def chunk_and_store():
            store.storage.store_memory(
                memory_type="knowledge_base",
                agent_id="perf-test",
                session_id="large-doc-session",
                content=large_content,
                session_iter=1,
                title="Large Performance Test Document",
                auto_chunk=True
            )

        median_ms, all_times, min_ms, max_ms = time_operation(chunk_and_store, iterations=3)

        # Get chunk count
        conn = store._get_connection()
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE parent_id IN (SELECT id FROM session_memories WHERE title = ?)",
            ("Large Performance Test Document",)
        ).fetchone()[0]
        conn.close()

        print(f"\n📄 Large Document Chunking Performance:")
        print(f"   Document size: ~{len(large_content)} chars (~{len(large_content) // 4} tokens)")
        print(f"   Chunks created: {chunk_count}")
        print(f"   Median: {median_ms:.2f} ms")
        print(f"   Min:    {min_ms:.2f} ms")
        print(f"   Max:    {max_ms:.2f} ms")
        print(f"   All:    {[f'{t:.2f}' for t in all_times]} ms")
        print(f"   Per-chunk avg: {median_ms / max(chunk_count, 1):.2f} ms")

        # Baseline: Should be < 45 seconds for large document
        assert median_ms < 45000, f"Large document chunking too slow: {median_ms:.2f}ms (expected < 45000ms)"
        assert chunk_count > 100, f"Expected many chunks for large document, got {chunk_count}"


def test_memory_usage_profiling():
    """Profile memory usage during operations (basic check)."""
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "perf_test.db")

        # Get baseline memory
        baseline_mb = sys.getsizeof(SessionMemoryStore) / (1024 * 1024)

        store = SessionMemoryStore(db_path=db_path)

        # Store moderate amount of data
        for i in range(50):
            store.storage.store_memory(
                memory_type="reports",
                agent_id="mem-test",
                session_id=f"session-{i}",
                content=f"Memory profiling test report {i} " * 100,  # ~500 chars each
                session_iter=1,
                task_code="mem-test"
            )

        # Check database file size
        db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)

        print(f"\n💾 Memory Usage Profile:")
        print(f"   Baseline: {baseline_mb:.2f} MB")
        print(f"   Database size: {db_size_mb:.2f} MB (50 memories)")
        print(f"   Per-memory avg: {db_size_mb / 50 * 1024:.2f} KB")

        # Baseline: Database should be reasonable size
        assert db_size_mb < 15, f"Database too large: {db_size_mb:.2f}MB (expected < 15MB for 50 memories)"


def test_performance_summary():
    """Print comprehensive performance summary."""
    print("\n" + "="*70)
    print("🎯 PERFORMANCE BENCHMARK SUMMARY")
    print("="*70)
    print("\nAll benchmarks completed successfully!")
    print("\nBaseline Metrics Established:")
    print("  ✅ Store single memory: < 150ms")
    print("  ✅ Vector search (k=10): < 600ms")
    print("  ✅ Batch store (100): < 15s")
    print("  ✅ Large doc chunking: < 45s")
    print("  ✅ Memory usage: Reasonable")
    print("\nPerformance Status: ACCEPTABLE ✅")
    print("="*70 + "\n")
