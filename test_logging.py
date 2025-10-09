#!/usr/bin/env python3
"""
Test script to verify logging instrumentation works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import after logging is configured in main.py
os.chdir(Path(__file__).parent)

# Configure logging before importing anything else
import logging
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(logs_dir / "test_logging.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("TESTING LOGGING INSTRUMENTATION")
print("=" * 80)
print(f"Log file: {logs_dir / 'test_logging.log'}")
print()

# Now import the store
from src.session_memory_store import SessionMemoryStore

# Create test database
test_db = Path(__file__).parent / "test_logging.db"
if test_db.exists():
    test_db.unlink()

print(f"Creating SessionMemoryStore with test database: {test_db}")
store = SessionMemoryStore(db_path=str(test_db))

print("\n" + "=" * 80)
print("TEST 1: Store simple memory (no chunking)")
print("=" * 80)

result1 = store.store_memory(
    memory_type="working_memory",
    agent_id="test-agent",
    session_id="test-session-1",
    content="This is a short test memory that should NOT be chunked.",
    session_iter=1,
    task_code="test-task-1",
    title="Test Memory 1",
    description="Simple test without chunking",
    auto_chunk=False
)

print(f"\nResult 1: success={result1['success']}, memory_id={result1.get('memory_id')}, chunks={result1.get('chunks_created')}")

print("\n" + "=" * 80)
print("TEST 2: Store longer memory WITH chunking (will trigger embedding model)")
print("=" * 80)

long_content = """
# Test Document

## Section 1: Introduction
This is a longer test document that will be chunked. The purpose is to trigger
the chunking and embedding pipeline to see all the instrumentation logs.

## Section 2: Details
Here we add more content to ensure the document is large enough to be chunked
into multiple pieces. Each section will become a separate chunk.

## Section 3: Conclusion
Finally, we wrap up with a conclusion section to complete our test document.
This should give us at least 3 chunks to work with.
"""

result2 = store.store_memory(
    memory_type="reports",
    agent_id="test-agent",
    session_id="test-session-2",
    content=long_content,
    session_iter=1,
    task_code="test-task-2",
    title="Test Memory 2 with Chunking",
    description="Test with auto-chunking enabled",
    auto_chunk=True
)

print(f"\nResult 2: success={result2['success']}, memory_id={result2.get('memory_id')}, chunks={result2.get('chunks_created')}")

print("\n" + "=" * 80)
print("LOGGING TEST COMPLETE")
print("=" * 80)
print(f"\n✓ Check the log file for detailed instrumentation: {logs_dir / 'test_logging.log'}")
print(f"✓ You should see:")
print("  - Entry/exit logs for each operation")
print("  - Timing information for each step")
print("  - Chunking details (for test 2)")
print("  - Embedding model loading (for test 2)")
print("  - Database operations timing")

# Cleanup
if test_db.exists():
    test_db.unlink()
    print(f"\n✓ Cleaned up test database: {test_db}")
