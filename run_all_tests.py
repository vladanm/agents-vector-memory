#!/usr/bin/env python3
"""Run all test files and capture results."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# List of all test files
TEST_FILES = [
    "test_chunking.py",
    "test_chunking_debug.py",
    "test_comprehensive_mcp.py",
    "test_connection_leak_fix.py",
    "test_consolidated_search.py",
    "test_direct_chunking.py",
    "test_e2e_final.py",
    "test_e2e_validation.py",
    "test_e2e_validation_v2.py",
    "test_langchain_chunking.py",
    "test_large_response_handling.py",
    "test_new_features.py",
    "test_post_restart.py",
    "test_production_fix.py",
    "test_semantic_search_qa_comprehensive.py",
    "test_server_startup.py",
    "test_session_memory.py",
    "test_size_warnings.py",
    "test_store_with_chunks.py",
    "test_vector_search.py",
    "test_vector_simple.py",
    "test_write_tool_verification.py",
    "test_yaml_frontmatter.py",
]

def run_test(test_file: str, timeout: int = 60) -> tuple[str, int, str]:
    """Run a single test file."""
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return ("PASS" if result.returncode == 0 else "FAIL", result.returncode, result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", -1, f"Test exceeded {timeout} second timeout")
    except Exception as e:
        return ("ERROR", -2, str(e))

def main():
    print("=" * 70)
    print("Phase 1 Test Suite Validation")
    print(f"Running {len(TEST_FILES)} test files")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print()

    results = {}
    passed = 0
    failed = 0
    timeouts = 0
    errors = 0

    for i, test_file in enumerate(TEST_FILES, 1):
        print(f"[{i}/{len(TEST_FILES)}] Running: {test_file}")

        status, exit_code, output = run_test(test_file)
        results[test_file] = (status, exit_code, output)

        if status == "PASS":
            print(f"  ✅ PASS")
            passed += 1
        elif status == "TIMEOUT":
            print(f"  ⏱️  TIMEOUT")
            timeouts += 1
            failed += 1
        elif status == "ERROR":
            print(f"  ⚠️  ERROR")
            errors += 1
            failed += 1
        else:
            print(f"  ❌ FAIL (exit code: {exit_code})")
            failed += 1
        print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(TEST_FILES)}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"  - Test Failures: {failed - timeouts - errors}")
    print(f"  - Timeouts:      {timeouts}")
    print(f"  - Errors:        {errors}")
    print()
    pass_rate = (passed * 100) // len(TEST_FILES) if TEST_FILES else 0
    print(f"Pass Rate: {pass_rate}%")
    print()

    # Detailed results
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for test_file in TEST_FILES:
        status, exit_code, _ = results[test_file]
        if status == "PASS":
            print(f"✅ {test_file}")
        elif status == "TIMEOUT":
            print(f"⏱️  {test_file} (timeout)")
        elif status == "ERROR":
            print(f"⚠️  {test_file} (error)")
        else:
            print(f"❌ {test_file} (exit {exit_code})")
    print()

    # Failed test details
    if failed > 0:
        print("=" * 70)
        print("FAILED TEST DETAILS")
        print("=" * 70)
        for test_file in TEST_FILES:
            status, exit_code, output = results[test_file]
            if status != "PASS":
                print(f"\n--- {test_file} ({status}) ---")
                # Print last 30 lines of output
                lines = output.strip().split('\n')
                if len(lines) > 30:
                    print("... (truncated) ...")
                    print('\n'.join(lines[-30:]))
                else:
                    print(output)
                print()

    # Exit with failure if any tests failed
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
