#!/bin/bash
# Run all test files and capture results

cd /Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp

echo "======================================================================"
echo "Phase 1 Test Suite Validation"
echo "Running all 23 test files"
echo "======================================================================"
echo ""

# Array to track results
declare -A test_results
total_tests=0
passed_tests=0
failed_tests=0

# List of all test files (root level)
tests=(
    "test_chunking.py"
    "test_chunking_debug.py"
    "test_comprehensive_mcp.py"
    "test_connection_leak_fix.py"
    "test_consolidated_search.py"
    "test_direct_chunking.py"
    "test_e2e_final.py"
    "test_e2e_validation.py"
    "test_e2e_validation_v2.py"
    "test_langchain_chunking.py"
    "test_large_response_handling.py"
    "test_new_features.py"
    "test_post_restart.py"
    "test_production_fix.py"
    "test_semantic_search_qa_comprehensive.py"
    "test_server_startup.py"
    "test_session_memory.py"
    "test_size_warnings.py"
    "test_store_with_chunks.py"
    "test_vector_search.py"
    "test_vector_simple.py"
    "test_write_tool_verification.py"
    "test_yaml_frontmatter.py"
)

# Run each test
for test in "${tests[@]}"; do
    total_tests=$((total_tests + 1))
    echo "[$total_tests/23] Running: $test"

    # Run test with timeout (60 seconds max)
    timeout 60 python "$test" > "/tmp/${test}.log" 2>&1
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  ✅ PASS"
        test_results[$test]="PASS"
        passed_tests=$((passed_tests + 1))
    elif [ $exit_code -eq 124 ]; then
        echo "  ⏱️  TIMEOUT"
        test_results[$test]="TIMEOUT"
        failed_tests=$((failed_tests + 1))
    else
        echo "  ❌ FAIL (exit code: $exit_code)"
        test_results[$test]="FAIL"
        failed_tests=$((failed_tests + 1))
    fi
    echo ""
done

# Summary
echo "======================================================================"
echo "TEST SUMMARY"
echo "======================================================================"
echo "Total Tests:  $total_tests"
echo "Passed:       $passed_tests"
echo "Failed:       $failed_tests"
echo ""
echo "Pass Rate: $((passed_tests * 100 / total_tests))%"
echo ""

# Detailed results
echo "======================================================================"
echo "DETAILED RESULTS"
echo "======================================================================"
for test in "${tests[@]}"; do
    status="${test_results[$test]}"
    if [ "$status" == "PASS" ]; then
        echo "✅ $test"
    elif [ "$status" == "TIMEOUT" ]; then
        echo "⏱️  $test (timeout)"
    else
        echo "❌ $test"
    fi
done
echo ""

# Exit with failure if any tests failed
if [ $failed_tests -gt 0 ]; then
    exit 1
else
    exit 0
fi
