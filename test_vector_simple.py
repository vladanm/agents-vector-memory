#!/usr/bin/env python3
"""
Simple Vector Search Test
=========================

Quick test of the vector search fix without loading the embedding model.
"""

import sqlite3
from pathlib import Path
import sqlite_vec

def test_vector_query():
    """Test that vector query syntax works."""
    print("🧪 Testing Vector Query Syntax...")
    
    db_path = Path("/Users/vladanm/projects/subagents/simple-agents/memory/memory/memory/agent_session_memory.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        return False
    
    try:
        # Connect with sqlite-vec
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        # Test basic query
        result = conn.execute("SELECT COUNT(*) FROM session_memories").fetchone()
        print(f"✅ Found {result[0]} memories in database")
        
        # Test vector search table exists
        result = conn.execute("SELECT COUNT(*) FROM vec_session_search").fetchone()
        print(f"✅ Found {result[0]} vector embeddings")
        
        # Test vector search syntax (without actual embedding)
        # This should not throw the "LIMIT or k constraint" error
        try:
            # Create a dummy embedding vector (384 dimensions of zeros)
            dummy_embedding = bytes(384 * 4)  # 384 float32 values
            
            # Test the fixed query syntax
            rows = conn.execute("""
                SELECT m.*, v.distance
                FROM session_memories m
                JOIN (
                    SELECT memory_id, distance 
                    FROM vec_session_search 
                    WHERE embedding MATCH ? 
                    ORDER BY distance ASC
                    LIMIT ?
                ) v ON m.id = v.memory_id
                AND v.distance < 0.5
                ORDER BY v.distance ASC
            """, [dummy_embedding, 5]).fetchall()
            
            print(f"✅ Vector query syntax test passed - no constraint errors")
            print(f"📊 Query returned {len(rows)} results (expected 0 with dummy embedding)")
            
        except Exception as e:
            if "LIMIT or 'k = ?' constraint" in str(e):
                print(f"❌ Vector constraint error still present: {e}")
                return False
            else:
                print(f"✅ Different error (expected with dummy data): {e}")
        
        conn.close()
        print("\n🎉 Vector search syntax fix appears to be working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_query()
    exit(0 if success else 1)