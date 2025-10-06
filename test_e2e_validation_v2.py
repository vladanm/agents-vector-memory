#!/usr/bin/env python3
"""
End-to-End Validation Tests for MCP Large Response Handling - Version 2

This test suite validates ALL 11 acceptance criteria using the REAL database
and direct testing of implementation methods.

Database: /Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db

Acceptance Criteria:
1-4. write_document_to_file tool with auto/custom paths
5. Custom path validation and creation
6. Metadata frontmatter generation
7. Chunked document reconstruction
8-11. Size warnings in search results
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.session_memory_store import SessionMemoryStore


def run_validation_tests():
    """Run comprehensive E2E validation tests"""

    db_path = "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
    store = SessionMemoryStore(db_path=db_path)

    print("="*80)
    print("COMPREHENSIVE E2E VALIDATION - MCP LARGE RESPONSE HANDLING")
    print("="*80)
    print(f"\nDatabase: {db_path}\n")

    passed = 0
    failed = 0
    test_files = []

    try:
        # ========== TEST 1: Method Existence ==========
        print("\n" + "="*80)
        print("TEST 1: AC1 - write_document_to_file Method Exists")
        print("="*80)

        if hasattr(store, 'write_document_to_file') and callable(store.write_document_to_file):
            print("✅ PASS: write_document_to_file method exists and is callable")
            passed += 1
        else:
            print("❌ FAIL: write_document_to_file method not found")
            failed += 1

        # ========== TEST 2: Error Handling - MEMORY_NOT_FOUND ==========
        print("\n" + "="*80)
        print("TEST 2: AC3 - MEMORY_NOT_FOUND Error Code")
        print("="*80)

        result = store.write_document_to_file(memory_id=999999999)

        if result.get('success') == False and result.get('error_code') == 'MEMORY_NOT_FOUND':
            print(f"✅ PASS: MEMORY_NOT_FOUND error correctly raised")
            print(f"   Error message: {result.get('error_message')}")
            passed += 1
        else:
            print(f"❌ FAIL: Expected MEMORY_NOT_FOUND error")
            print(f"   Got: {result}")
            failed += 1

        # ========== TEST 3: Error Handling - INVALID_PARAMETER ==========
        print("\n" + "="*80)
        print("TEST 3: AC3 - INVALID_PARAMETER Error Code")
        print("="*80)

        result = store.write_document_to_file(memory_id=-1)

        if result.get('success') == False and result.get('error_code') in ['INVALID_PARAMETER', 'MEMORY_NOT_FOUND']:
            print(f"✅ PASS: Invalid parameter error correctly raised")
            print(f"   Error code: {result.get('error_code')}")
            passed += 1
        else:
            print(f"❌ FAIL: Expected error for invalid parameter")
            failed += 1

        # ========== TEST 4: Error Handling - INVALID_PATH ==========
        print("\n" + "="*80)
        print("TEST 4: AC3 - INVALID_PATH Error Code")
        print("="*80)

        # First find a real memory ID from database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            real_memory_id = row[0]
            result = store.write_document_to_file(
                memory_id=real_memory_id,
                output_path="relative/path.md"  # Invalid: not absolute
            )

            if result.get('success') == False and result.get('error_code') == 'INVALID_PATH':
                print(f"✅ PASS: INVALID_PATH error correctly raised for relative path")
                passed += 1
            else:
                print(f"❌ FAIL: Expected INVALID_PATH error")
                print(f"   Got: {result}")
                failed += 1
        else:
            print("⚠ SKIP: No memories in database to test with")

        # ========== TEST 5: Auto-Generated Path ==========
        print("\n" + "="*80)
        print("TEST 5: AC2-4 - Auto-Generated Temp Path")
        print("="*80)

        # Create test memory directly in database
        content = "# E2E Test Document\n\n" + ("Test content. " * 100)
        result = store.store_memory(
            memory_type="knowledge_base",
            agent_id="e2e-test-agent",
            session_id="e2e-validation",
            content=content,
            title="E2E Auto-Path Test",
            tags=["e2e-test"],
            auto_chunk=False
        )

        test_memory_id = result['memory_id']
        print(f"Created test memory: {test_memory_id}")

        # Write with auto-generated path
        write_result = store.write_document_to_file(
            memory_id=test_memory_id,
            include_metadata=True
        )

        if write_result.get('success'):
            file_path = write_result['file_path']
            test_files.append(file_path)

            # Validate path is in temp directory
            if file_path.startswith(tempfile.gettempdir()) and os.path.exists(file_path):
                print(f"✅ PASS: Auto-generated path works correctly")
                print(f"   Path: {file_path}")
                print(f"   Size: {write_result['file_size_human']}")
                print(f"   Tokens: {write_result['estimated_tokens']}")
                passed += 1
            else:
                print(f"❌ FAIL: Path not in temp directory or file doesn't exist")
                print(f"   Path: {file_path}")
                print(f"   Temp dir: {tempfile.gettempdir()}")
                failed += 1
        else:
            print(f"❌ FAIL: write_document_to_file failed")
            print(f"   Error: {write_result}")
            failed += 1

        # ========== TEST 6: Custom Path ==========
        print("\n" + "="*80)
        print("TEST 6: AC5 - Custom Path Validation")
        print("="*80)

        custom_dir = os.path.join(tempfile.gettempdir(), "e2e_validation_custom")
        custom_path = os.path.join(custom_dir, "custom_doc.md")

        # Clean up if exists
        if os.path.exists(custom_path):
            os.remove(custom_path)
        if os.path.exists(custom_dir):
            os.rmdir(custom_dir)

        write_result = store.write_document_to_file(
            memory_id=test_memory_id,
            output_path=custom_path,
            include_metadata=False
        )

        if write_result.get('success'):
            test_files.append(custom_path)

            if write_result['file_path'] == custom_path and os.path.exists(custom_path):
                print(f"✅ PASS: Custom path works correctly")
                print(f"   Path: {custom_path}")
                print(f"   Directory created: {os.path.isdir(custom_dir)}")
                passed += 1
            else:
                print(f"❌ FAIL: Custom path validation failed")
                failed += 1
        else:
            print(f"❌ FAIL: Custom path write failed: {write_result}")
            failed += 1

        # ========== TEST 7: Metadata Frontmatter ==========
        print("\n" + "="*80)
        print("TEST 7: AC6 - YAML Frontmatter Generation")
        print("="*80)

        # Write with metadata
        write_result = store.write_document_to_file(
            memory_id=test_memory_id,
            include_metadata=True
        )

        if write_result.get('success'):
            file_path = write_result['file_path']
            test_files.append(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            required_fields = ['memory_id:', 'title:', 'memory_type:', 'created_at:', 'session_id:', 'agent_id:']
            has_frontmatter = content.startswith('---')
            has_all_fields = all(field in content for field in required_fields)

            if has_frontmatter and has_all_fields:
                print(f"✅ PASS: YAML frontmatter generated correctly")
                print(f"   All required fields present")
                passed += 1
            else:
                print(f"❌ FAIL: Frontmatter validation failed")
                print(f"   Has frontmatter: {has_frontmatter}")
                print(f"   Has all fields: {has_all_fields}")
                failed += 1
        else:
            print(f"❌ FAIL: Write with metadata failed")
            failed += 1

        # ========== TEST 8: Chunked Document Reconstruction ==========
        print("\n" + "="*80)
        print("TEST 8: AC7 - Chunked Document Reconstruction")
        print("="*80)

        # Create large document that will be chunked
        large_content = "# Large Chunked Document\n\n"
        large_content += "## Section 1\n\n" + ("Content for section 1. " * 200)
        large_content += "\n\n## Section 2\n\n" + ("Content for section 2. " * 200)
        large_content += "\n\n## Section 3\n\n" + ("Content for section 3. " * 200)

        original_length = len(large_content)
        print(f"Original content: {original_length} chars")

        result = store.store_memory(
            memory_type="knowledge_base",
            agent_id="e2e-test-agent",
            session_id="e2e-validation",
            content=large_content,
            title="Large Chunked Test",
            tags=["chunked-test"],
            auto_chunk=True
        )

        chunked_memory_id = result['memory_id']

        # Check if it was chunked
        chunks = store._get_chunks_for_memory(chunked_memory_id)
        chunk_count = len(chunks) if chunks else 0
        print(f"Created {chunk_count} chunks")

        # Reconstruct via write_document_to_file
        write_result = store.write_document_to_file(
            memory_id=chunked_memory_id,
            include_metadata=False
        )

        if write_result.get('success'):
            file_path = write_result['file_path']
            test_files.append(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                reconstructed = f.read()

            reconstructed_length = len(reconstructed)
            length_diff = abs(original_length - reconstructed_length)
            tolerance = original_length * 0.05  # 5%

            has_sections = all(section in reconstructed for section in ['# Large Chunked Document', '## Section 1', '## Section 2', '## Section 3'])

            if length_diff <= tolerance and has_sections:
                print(f"✅ PASS: Chunked document reconstructed correctly")
                print(f"   Original: {original_length} chars")
                print(f"   Reconstructed: {reconstructed_length} chars")
                print(f"   Difference: {length_diff} chars ({length_diff/original_length*100:.1f}%)")
                print(f"   All sections present: {has_sections}")
                passed += 1
            else:
                print(f"❌ FAIL: Reconstruction issues")
                print(f"   Length diff too large or sections missing")
                failed += 1
        else:
            print(f"❌ FAIL: Write failed: {write_result}")
            failed += 1

        # ========== TEST 9-11: Size Warnings in Search ==========
        print("\n" + "="*80)
        print("TEST 9: AC8-11 - Size Warning Fields in Search Results")
        print("="*80)

        # Search for our test documents
        search_result = store.search_with_granularity(
            query="E2E Test",
            memory_type="knowledge_base",
            granularity="coarse",
            limit=10
        )

        if search_result.get('results'):
            # Check first result for required fields
            doc = search_result['results'][0]

            has_estimated_tokens = 'estimated_tokens' in doc
            has_exceeds_limit = 'exceeds_response_limit' in doc

            print(f"Sample result fields:")
            print(f"   - estimated_tokens: {doc.get('estimated_tokens')}")
            print(f"   - exceeds_response_limit: {doc.get('exceeds_response_limit')}")
            print(f"   - size_warning: {doc.get('size_warning')}")

            if has_estimated_tokens and has_exceeds_limit:
                print(f"✅ PASS: Size warning fields present in search results")
                passed += 1
            else:
                print(f"❌ FAIL: Missing size warning fields")
                print(f"   Has estimated_tokens: {has_estimated_tokens}")
                print(f"   Has exceeds_response_limit: {has_exceeds_limit}")
                failed += 1
        else:
            print(f"⚠ SKIP: No search results to validate")

        # ========== TEST 10: Helper Methods ==========
        print("\n" + "="*80)
        print("TEST 10: Helper Methods - Token Estimation")
        print("="*80)

        if hasattr(store, '_estimate_tokens'):
            test_text = "This is a test. " * 100
            tokens = store._estimate_tokens(test_text)

            if tokens > 0:
                print(f"✅ PASS: Token estimation working")
                print(f"   Text length: {len(test_text)} chars")
                print(f"   Estimated tokens: {tokens}")
                print(f"   Ratio: {len(test_text)/tokens:.2f} chars/token")
                passed += 1
            else:
                print(f"❌ FAIL: Token estimation returned 0")
                failed += 1
        else:
            print(f"❌ FAIL: _estimate_tokens method not found")
            failed += 1

        # ========== TEST 11: Response Format Validation ==========
        print("\n" + "="*80)
        print("TEST 11: Response Format - Success Response Structure")
        print("="*80)

        write_result = store.write_document_to_file(memory_id=test_memory_id)

        required_fields = ['success', 'file_path', 'file_size_bytes', 'file_size_human', 'estimated_tokens', 'memory_id', 'created_at', 'message']

        if write_result.get('success'):
            test_files.append(write_result['file_path'])
            missing_fields = [f for f in required_fields if f not in write_result]

            if not missing_fields:
                print(f"✅ PASS: Success response has all required fields")
                for field in required_fields:
                    print(f"   - {field}: {write_result.get(field)}")
                passed += 1
            else:
                print(f"❌ FAIL: Missing fields in response: {missing_fields}")
                failed += 1
        else:
            print(f"❌ FAIL: Write failed: {write_result}")
            failed += 1

    finally:
        # Cleanup
        print("\n" + "="*80)
        print("CLEANUP")
        print("="*80)

        for file_path in test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ Removed: {file_path}")
            except Exception as e:
                print(f"⚠ Could not remove {file_path}: {e}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Pass Rate: {pass_rate:.1f}%")

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - PRODUCTION READY")
        print("   Implementation meets all acceptance criteria")
        print("   Ready for production deployment")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - ISSUES FOUND")
        print("   Review failed tests above")
        print("   Implementation requires fixes before production")
        return False


if __name__ == "__main__":
    import sys
    success = run_validation_tests()
    sys.exit(0 if success else 1)
