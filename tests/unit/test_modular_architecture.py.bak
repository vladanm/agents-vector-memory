"""
Unit tests for modular architecture (Phase 2 feature).

Tests that SessionMemoryStore delegates to operation modules correctly.
"""

import pytest


@pytest.mark.unit
def test_storage_module_exists():
    """Verify storage module is importable."""
    from src.storage import StorageOperations
    assert StorageOperations is not None


@pytest.mark.unit
def test_search_module_exists():
    """Verify search module is importable."""
    from src.search import SearchOperations
    assert SearchOperations is not None


@pytest.mark.unit
def test_maintenance_module_exists():
    """Verify maintenance module is importable."""
    from src.maintenance import MaintenanceOperations
    assert MaintenanceOperations is not None


@pytest.mark.unit
def test_chunking_storage_module_exists():
    """Verify chunking storage module is importable."""
    from src.chunking_storage import ChunkingStorageOperations
    assert ChunkingStorageOperations is not None


@pytest.mark.unit
def test_store_delegates_to_modules(store):
    """Verify SessionMemoryStore delegates to operation modules."""
    assert hasattr(store, 'storage')
    assert hasattr(store, 'search')
    assert hasattr(store, 'maintenance')
    assert hasattr(store, 'chunking')

    from src.storage import StorageOperations
    from src.search import SearchOperations
    from src.maintenance import MaintenanceOperations
    from src.chunking_storage import ChunkingStorageOperations

    assert isinstance(store.storage, StorageOperations)
    assert isinstance(store.search, SearchOperations)
    assert isinstance(store.maintenance, MaintenanceOperations)
    assert isinstance(store.chunking, ChunkingStorageOperations)


@pytest.mark.unit
def test_public_api_delegates(store, sample_memory):
    """Verify public API methods delegate to modules."""
    # store_memory should delegate to storage module
    result = store.store_memory(**sample_memory)
    assert result["success"]

    # search_memories should delegate to search module
    result = store.search_memories(memory_type="session_context")
    assert result["success"]

    # get_session_stats should delegate to maintenance module
    result = store.get_session_stats()
    assert result["success"]


@pytest.mark.unit
def test_no_circular_imports():
    """Verify no circular import issues between modules."""
    # If these all import successfully, there are no circular imports
    from src.session_memory_store import SessionMemoryStore
    from src.storage import StorageOperations
    from src.search import SearchOperations
    from src.maintenance import MaintenanceOperations
    from src.chunking_storage import ChunkingStorageOperations

    # All modules loaded successfully
    assert True


@pytest.mark.unit
def test_modules_share_connection(store):
    """Verify all operation modules share the same database connection."""
    # All modules should use the same _get_connection method
    assert store.storage._get_connection == store._get_connection
    assert store.search._get_connection == store._get_connection
    assert store.maintenance._get_connection == store._get_connection
    assert store.chunking._get_connection == store._get_connection
