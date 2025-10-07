#!/usr/bin/env python3
"""
Basic tests for Agent Session Memory MCP Server
===============================================

Tests the core functionality of the session-centric memory system.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore
from src.config import Config


def test_basic_storage_and_retrieval():
    """Test basic memory storage and retrieval"""
    print("🧪 Testing basic storage and retrieval...")
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SessionMemoryStore(db_path)
        
        # Test storing session context for main agent
        result = store.store_memory(
            memory_type="session_context",
            agent_id="main",
            session_id="test_session_001",
            content="This is a test session context for main agent",
            session_iter=1,
            title="Test Context",
            tags=["test", "main-agent"]
        )
        
        assert result["success"] == True, f"Storage failed: {result}"
        memory_id = result["memory_id"]
        print(f"✅ Stored memory with ID: {memory_id}")
        
        # Test retrieval
        retrieved = store.get_memory(memory_id)
        assert retrieved["success"] == True, f"Retrieval failed: {retrieved}"
        
        memory = retrieved["memory"]
        assert memory["agent_id"] == "main"
        assert memory["session_id"] == "test_session_001"
        assert memory["memory_type"] == "session_context"
        print("✅ Memory retrieved successfully")


def test_scoped_search_ordering():
    """Test scoped search with proper ordering"""
    print("\n🧪 Testing scoped search with ordering...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SessionMemoryStore(db_path)
        
        # Store multiple memories with different iterations
        memories = [
            ("session_context", "main", "sess_123", 1, None, "Context iter 1"),
            ("session_context", "main", "sess_123", 5, None, "Context iter 5"),
            ("session_context", "main", "sess_123", 3, None, "Context iter 3"),
            ("reports", "specialized-agent", "sess_123", 2, "task_A", "Report task A iter 2"),
            ("reports", "specialized-agent", "sess_123", 4, "task_A", "Report task A iter 4"),
            ("working_memory", "specialized-agent", "sess_123", 1, "task_B", "Working memory task B"),
        ]
        
        memory_ids = []
        for memory_type, agent_id, session_id, session_iter, task_code, content in memories:
            result = store.store_memory(
                memory_type=memory_type,
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                session_iter=session_iter,
                task_code=task_code
            )
            assert result["success"] == True
            memory_ids.append(result["memory_id"])
        
        print(f"✅ Stored {len(memory_ids)} memories")
        
        # Test scoped search for main agent session context
        main_results = store.search_memories(
            memory_type="session_context",
            agent_id="main",
            session_id="sess_123"
        )
        
        assert main_results["success"] == True
        results = main_results["results"]
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # Verify ordering: session_iter DESC
        iterations = [r["session_iter"] for r in results]
        assert iterations == [5, 3, 1], f"Wrong ordering: {iterations}"
        print("✅ Session context search properly ordered by session_iter DESC")
        
        # Test scoped search for specialized agent with task_code
        task_results = store.search_memories(
            memory_type="reports",
            agent_id="specialized-agent",
            session_id="sess_123",
            task_code="task_A"
        )
        
        assert task_results["success"] == True
        task_res = task_results["results"]
        assert len(task_res) == 2, f"Expected 2 task A reports, got {len(task_res)}"
        
        # Verify ordering
        task_iterations = [r["session_iter"] for r in task_res]
        assert task_iterations == [4, 2], f"Wrong task ordering: {task_iterations}"
        print("✅ Task-scoped search properly ordered")


def test_task_continuity():
    """Test conditional loading for task continuity"""
    print("\n🧪 Testing task continuity...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SessionMemoryStore(db_path)
        
        # Store session context for specific task
        result = store.store_memory(
            memory_type="session_context",
            agent_id="specialized-agent",
            session_id="sess_456",
            content="Previous work on security audit task",
            session_iter=2,
            task_code="security-audit",
            title="Security Audit Context"
        )
        assert result["success"] == True
        print("✅ Stored task-specific session context")
        
        # Test loading context for same task - should find it
        found_result = store.load_session_context_for_task(
            agent_id="specialized-agent",
            session_id="sess_456",
            current_task_code="security-audit"
        )
        
        assert found_result["success"] == True
        assert found_result["found_previous_context"] == True
        assert found_result["context"]["task_code"] == "security-audit"
        print("✅ Found previous context for same task")
        
        # Test loading context for different task - should not find it
        not_found_result = store.load_session_context_for_task(
            agent_id="specialized-agent", 
            session_id="sess_456",
            current_task_code="performance-audit"
        )
        
        assert not_found_result["success"] == True
        assert not_found_result["found_previous_context"] == False
        print("✅ Correctly did not find context for different task")


def test_session_stats():
    """Test session statistics"""
    print("\n🧪 Testing session statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SessionMemoryStore(db_path)
        
        # Store various memories
        memories = [
            ("session_context", "main", "sess_001", 1, None),
            ("system_memory", "main", "sess_001", 1, None),
            ("reports", "specialized-agent", "sess_001", 1, "task_X"),
            ("working_memory", "specialized-agent", "sess_001", 2, "task_Y"),
            ("input_prompt", "main", "sess_002", 1, None),
        ]
        
        for memory_type, agent_id, session_id, session_iter, task_code in memories:
            result = store.store_memory(
                memory_type=memory_type,
                agent_id=agent_id,
                session_id=session_id,
                content=f"Test content for {memory_type}",
                session_iter=session_iter,
                task_code=task_code
            )
            assert result["success"] == True
        
        print("✅ Stored test memories for stats")
        
        # Get overall stats
        stats = store.get_session_stats()
        assert stats["success"] == True
        assert stats["total_memories"] == 5
        assert stats["unique_sessions"] == 2
        assert "session_context" in stats["memory_type_breakdown"]
        print("✅ Overall statistics correct")
        
        # Get filtered stats for specific session
        session_stats = store.get_session_stats(session_id="sess_001")
        assert session_stats["success"] == True
        assert session_stats["total_memories"] == 4  # 4 memories in sess_001
        print("✅ Session-filtered statistics correct")


def test_memory_types_validation():
    """Test memory type validation"""
    print("\n🧪 Testing memory type validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        store = SessionMemoryStore(db_path)
        
        # Test valid memory types
        valid_types = Config.MEMORY_TYPES
        for memory_type in valid_types:
            result = store.store_memory(
                memory_type=memory_type,
                agent_id="main",
                session_id="test_session",
                content=f"Test content for {memory_type}"
            )
            assert result["success"] == True, f"Valid memory type {memory_type} failed"
        
        print(f"✅ All {len(valid_types)} valid memory types accepted")
        
        # Test invalid memory type
        invalid_result = store.store_memory(
            memory_type="invalid_type",
            agent_id="main",
            session_id="test_session",
            content="Test content"
        )
        assert invalid_result["success"] == False
        assert "Invalid memory type" in invalid_result["error"]
        print("✅ Invalid memory type correctly rejected")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Agent Session Memory Tests")
    print("=" * 50)
    
    try:
        test_basic_storage_and_retrieval()
        test_scoped_search_ordering() 
        test_task_continuity()
        test_session_stats()
        test_memory_types_validation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("Agent Session Memory MCP Server is working correctly")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()