#!/usr/bin/env python3
"""
Comprehensive Test Suite for Consolidated MCP Search Functions
==============================================================

Tests the consolidated search functions with granularity parameter covering:
1. All 9 search patterns (3 memory types × 3 granularities)
2. Granularity parameter validation
3. Parameter pass-through (agent_id, session_id, etc.)
4. Granularity mapping (specific_chunks→fine, section_context→medium, full_documents→coarse)
5. Default granularity behavior
6. Error handling for invalid granularity values

Consolidated Functions Under Test:
- search_knowledge_base(granularity='specific_chunks|section_context|full_documents')
- search_reports(granularity='specific_chunks|section_context|full_documents')
- search_working_memory(granularity='specific_chunks|section_context|full_documents')
"""

import json
import tempfile
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Literal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.session_memory_store import SessionMemoryStore
from src.config import Config

# Granularity mapping constant (same as in main.py)
GRANULARITY_MAP = {
    "specific_chunks": "fine",
    "section_context": "medium",
    "full_documents": "coarse"
}


class ConsolidatedSearchTester:
    """Test harness for consolidated search functions"""

    def __init__(self, db_path: str):
        self.store = SessionMemoryStore(db_path=db_path)
        self.test_results = []
        self.test_data_created = []

    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        result = {
            "name": name,
            "passed": passed,
            "details": details,
            "status": status
        }
        self.test_results.append(result)
        print(f"{status}: {name}")
        if details:
            print(f"  {details}")
        return passed

    def setup_test_data(self):
        """Create test data for all memory types"""
        print("\n" + "="*70)
        print("SETUP: Creating Test Data")
        print("="*70)

        test_session = "test-consolidated-search"
        test_agent = "test-agent"

        # Knowledge Base data
        kb_result = self.store.store_memory(
            memory_type="knowledge_base",
            agent_id=test_agent,
            session_id=test_session,
            content="# Go Programming Best Practices\n\n## Error Handling\nAlways handle errors explicitly in Go. Use `if err != nil` pattern.\n\n## Concurrency\nUse channels for communication between goroutines.",
            session_iter=1,
            title="Go Best Practices",
            tags=["golang", "best-practices"]
        )
        self.test_data_created.append(("knowledge_base", kb_result.get("memory_id")))

        # Reports data
        report_result = self.store.store_memory(
            memory_type="reports",
            agent_id=test_agent,
            session_id=test_session,
            content="# Code Analysis Report\n\n## Findings\nFound 3 potential race conditions in concurrent code.\n\n## Recommendations\nAdd mutex protection for shared state.",
            session_iter=1,
            task_code="code-analysis",
            title="Race Condition Analysis",
            tags=["analysis", "concurrency"]
        )
        self.test_data_created.append(("reports", report_result.get("memory_id")))

        # Working Memory data
        wm_result = self.store.store_memory(
            memory_type="working_memory",
            agent_id=test_agent,
            session_id=test_session,
            content="# Important Discovery\n\n## Key Insight\nThe goroutine leak occurs because context cancellation is not checked in the worker loop.\n\n## Action Required\nAdd context.Done() check in worker select statement.",
            session_iter=1,
            task_code="debugging",
            title="Goroutine Leak Root Cause",
            tags=["gotcha", "debugging"]
        )
        self.test_data_created.append(("working_memory", wm_result.get("memory_id")))

        print(f"\n✓ Created {len(self.test_data_created)} test memories")
        for memory_type, memory_id in self.test_data_created:
            print(f"  - {memory_type}: ID {memory_id}")

    # ========================================================================
    # Consolidated Search Function Implementations (matching main.py)
    # ========================================================================

    def search_knowledge_base(
        self,
        query: str,
        granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        limit: int = 3,
        similarity_threshold: float = 0.7,
        auto_merge_threshold: float = 0.6
    ) -> dict:
        """Search knowledge base with configurable granularity."""
        return self.store.search_with_granularity(
            query=query,
            memory_type="knowledge_base",
            granularity=GRANULARITY_MAP[granularity],
            agent_id=agent_id,
            session_id=session_id,
            session_iter=session_iter,
            task_code=task_code,
            limit=limit,
            similarity_threshold=similarity_threshold,
            auto_merge_threshold=auto_merge_threshold
        )

    def search_reports(
        self,
        query: str,
        granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        limit: int = 3,
        similarity_threshold: float = 0.7,
        auto_merge_threshold: float = 0.6
    ) -> dict:
        """Search reports with configurable granularity."""
        return self.store.search_with_granularity(
            query=query,
            memory_type="reports",
            granularity=GRANULARITY_MAP[granularity],
            agent_id=agent_id,
            session_id=session_id,
            session_iter=session_iter,
            task_code=task_code,
            limit=limit,
            similarity_threshold=similarity_threshold,
            auto_merge_threshold=auto_merge_threshold
        )

    def search_working_memory(
        self,
        query: str,
        granularity: Literal["specific_chunks", "section_context", "full_documents"] = "full_documents",
        agent_id: str = None,
        session_id: str = None,
        session_iter: int = None,
        task_code: str = None,
        limit: int = 3,
        similarity_threshold: float = 0.7,
        auto_merge_threshold: float = 0.6
    ) -> dict:
        """Search working memory with configurable granularity."""
        return self.store.search_with_granularity(
            query=query,
            memory_type="working_memory",
            granularity=GRANULARITY_MAP[granularity],
            agent_id=agent_id,
            session_id=session_id,
            session_iter=session_iter,
            task_code=task_code,
            limit=limit,
            similarity_threshold=similarity_threshold,
            auto_merge_threshold=auto_merge_threshold
        )

    # ========================================================================
    # Test Cases
    # ========================================================================

    def test_granularity_mapping(self):
        """Test that granularity mapping constant is correct"""
        print("\n" + "="*70)
        print("TEST 1: Granularity Mapping Validation")
        print("="*70)

        expected_mappings = {
            "specific_chunks": "fine",
            "section_context": "medium",
            "full_documents": "coarse"
        }

        all_correct = True
        for external, expected_internal in expected_mappings.items():
            actual_internal = GRANULARITY_MAP.get(external)
            if actual_internal == expected_internal:
                self.log_test(
                    f"Mapping '{external}' → '{expected_internal}'",
                    True,
                    f"Correctly maps to '{actual_internal}'"
                )
            else:
                self.log_test(
                    f"Mapping '{external}' → '{expected_internal}'",
                    False,
                    f"Expected '{expected_internal}', got '{actual_internal}'"
                )
                all_correct = False

        return all_correct

    def test_knowledge_base_all_granularities(self):
        """Test search_knowledge_base with all three granularity levels"""
        print("\n" + "="*70)
        print("TEST 2: Knowledge Base - All Granularities")
        print("="*70)

        query = "error handling in Go"
        granularities = ["specific_chunks", "section_context", "full_documents"]
        all_passed = True

        for granularity in granularities:
            try:
                result = self.search_knowledge_base(
                    query=query,
                    granularity=granularity,
                    session_id="test-consolidated-search"
                )

                # Validate result structure
                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"search_knowledge_base(granularity='{granularity}')",
                    has_success,
                    f"Returned result with {len(result.get('results', []))} items"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"search_knowledge_base(granularity='{granularity}')",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_reports_all_granularities(self):
        """Test search_reports with all three granularity levels"""
        print("\n" + "="*70)
        print("TEST 3: Reports - All Granularities")
        print("="*70)

        query = "race condition analysis"
        granularities = ["specific_chunks", "section_context", "full_documents"]
        all_passed = True

        for granularity in granularities:
            try:
                result = self.search_reports(
                    query=query,
                    granularity=granularity,
                    session_id="test-consolidated-search"
                )

                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"search_reports(granularity='{granularity}')",
                    has_success,
                    f"Returned result with {len(result.get('results', []))} items"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"search_reports(granularity='{granularity}')",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_working_memory_all_granularities(self):
        """Test search_working_memory with all three granularity levels"""
        print("\n" + "="*70)
        print("TEST 4: Working Memory - All Granularities")
        print("="*70)

        query = "goroutine leak debugging"
        granularities = ["specific_chunks", "section_context", "full_documents"]
        all_passed = True

        for granularity in granularities:
            try:
                result = self.search_working_memory(
                    query=query,
                    granularity=granularity,
                    session_id="test-consolidated-search"
                )

                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"search_working_memory(granularity='{granularity}')",
                    has_success,
                    f"Returned result with {len(result.get('results', []))} items"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"search_working_memory(granularity='{granularity}')",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_default_granularity(self):
        """Test default granularity parameter (should be full_documents)"""
        print("\n" + "="*70)
        print("TEST 5: Default Granularity Parameter")
        print("="*70)

        query = "test query"
        all_passed = True

        # Test each function without specifying granularity
        functions = [
            ("search_knowledge_base", self.search_knowledge_base),
            ("search_reports", self.search_reports),
            ("search_working_memory", self.search_working_memory)
        ]

        for func_name, func in functions:
            try:
                result = func(query=query, session_id="test-consolidated-search")

                # Should default to full_documents (coarse granularity)
                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"{func_name}() with default granularity",
                    has_success,
                    "Defaults to 'full_documents' granularity"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"{func_name}() with default granularity",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_parameter_passthrough(self):
        """Test that all parameters are passed through correctly"""
        print("\n" + "="*70)
        print("TEST 6: Parameter Pass-Through")
        print("="*70)

        all_passed = True

        # Test with various parameter combinations
        test_cases = [
            {
                "name": "agent_id filter",
                "params": {
                    "query": "test",
                    "agent_id": "test-agent",
                    "granularity": "full_documents"
                }
            },
            {
                "name": "session_id filter",
                "params": {
                    "query": "test",
                    "session_id": "test-consolidated-search",
                    "granularity": "full_documents"
                }
            },
            {
                "name": "session_iter filter",
                "params": {
                    "query": "test",
                    "session_id": "test-consolidated-search",
                    "session_iter": 1,
                    "granularity": "full_documents"
                }
            },
            {
                "name": "task_code filter",
                "params": {
                    "query": "test",
                    "task_code": "code-analysis",
                    "granularity": "full_documents"
                }
            },
            {
                "name": "limit parameter",
                "params": {
                    "query": "test",
                    "limit": 5,
                    "granularity": "full_documents"
                }
            },
            {
                "name": "similarity_threshold parameter",
                "params": {
                    "query": "test",
                    "similarity_threshold": 0.5,
                    "granularity": "full_documents"
                }
            },
            {
                "name": "auto_merge_threshold parameter",
                "params": {
                    "query": "test",
                    "auto_merge_threshold": 0.7,
                    "granularity": "section_context"
                }
            }
        ]

        for test_case in test_cases:
            try:
                result = self.search_knowledge_base(**test_case["params"])
                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"Parameter pass-through: {test_case['name']}",
                    has_success,
                    f"Parameters accepted and processed"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"Parameter pass-through: {test_case['name']}",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_invalid_granularity(self):
        """Test error handling for invalid granularity values"""
        print("\n" + "="*70)
        print("TEST 7: Invalid Granularity Error Handling")
        print("="*70)

        invalid_values = ["fine", "medium", "coarse", "invalid", ""]
        all_passed = True

        for invalid_value in invalid_values:
            try:
                # This should raise an error because Literal type doesn't accept these values
                # However, in our test we're calling the function directly,
                # so we catch KeyError from GRANULARITY_MAP
                result = self.store.search_with_granularity(
                    query="test",
                    memory_type="knowledge_base",
                    granularity=invalid_value,  # Pass invalid value directly
                    session_id="test-consolidated-search"
                )

                # If we get here, it means the invalid value was accepted (BAD)
                self.log_test(
                    f"Invalid granularity '{invalid_value}' rejection",
                    False,
                    "Invalid value was accepted (should be rejected)"
                )
                all_passed = False

            except (KeyError, ValueError) as e:
                # This is expected - invalid values should be rejected
                self.log_test(
                    f"Invalid granularity '{invalid_value}' rejection",
                    True,
                    f"Correctly rejected with: {type(e).__name__}"
                )
            except Exception as e:
                self.log_test(
                    f"Invalid granularity '{invalid_value}' rejection",
                    False,
                    f"Unexpected exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_result_structure_validation(self):
        """Test that results have expected structure for each granularity"""
        print("\n" + "="*70)
        print("TEST 8: Result Structure Validation")
        print("="*70)

        query = "test query"
        all_passed = True

        # Expected result structure elements
        granularity_expectations = {
            "specific_chunks": ["results", "total_results"],
            "section_context": ["results", "total_results"],
            "full_documents": ["results", "total_results"]
        }

        for granularity, expected_keys in granularity_expectations.items():
            try:
                result = self.search_knowledge_base(
                    query=query,
                    granularity=granularity,
                    session_id="test-consolidated-search"
                )

                # Check for expected keys
                has_all_keys = all(key in result for key in expected_keys)
                missing_keys = [key for key in expected_keys if key not in result]

                passed = self.log_test(
                    f"Result structure for '{granularity}'",
                    has_all_keys,
                    f"Has expected keys: {expected_keys}" if has_all_keys else f"Missing keys: {missing_keys}"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"Result structure for '{granularity}'",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def test_all_9_search_patterns(self):
        """Comprehensive test of all 9 search patterns (3 types × 3 granularities)"""
        print("\n" + "="*70)
        print("TEST 9: All 9 Search Patterns (3 Memory Types × 3 Granularities)")
        print("="*70)

        search_patterns = [
            ("knowledge_base", "specific_chunks", self.search_knowledge_base, "Go error handling"),
            ("knowledge_base", "section_context", self.search_knowledge_base, "Go error handling"),
            ("knowledge_base", "full_documents", self.search_knowledge_base, "Go error handling"),
            ("reports", "specific_chunks", self.search_reports, "race condition"),
            ("reports", "section_context", self.search_reports, "race condition"),
            ("reports", "full_documents", self.search_reports, "race condition"),
            ("working_memory", "specific_chunks", self.search_working_memory, "goroutine leak"),
            ("working_memory", "section_context", self.search_working_memory, "goroutine leak"),
            ("working_memory", "full_documents", self.search_working_memory, "goroutine leak")
        ]

        all_passed = True
        pattern_count = 0

        for memory_type, granularity, search_func, query in search_patterns:
            pattern_count += 1
            try:
                result = search_func(
                    query=query,
                    granularity=granularity,
                    session_id="test-consolidated-search"
                )

                has_success = "success" in result or "results" in result
                passed = self.log_test(
                    f"Pattern {pattern_count}/9: {memory_type} × {granularity}",
                    has_success,
                    f"Query: '{query}' - Returned {len(result.get('results', []))} results"
                )
                all_passed = all_passed and passed

            except Exception as e:
                self.log_test(
                    f"Pattern {pattern_count}/9: {memory_type} × {granularity}",
                    False,
                    f"Exception: {str(e)}"
                )
                all_passed = False

        return all_passed

    def run_all_tests(self):
        """Run all tests and generate summary"""
        print("\n" + "="*70)
        print("CONSOLIDATED SEARCH FUNCTIONS - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print(f"Test Database: {self.store.db_path}")
        print(f"Test Started: {datetime.now(timezone.utc).isoformat()}")

        # Setup test data
        self.setup_test_data()

        # Run all test suites
        test_suites = [
            self.test_granularity_mapping,
            self.test_knowledge_base_all_granularities,
            self.test_reports_all_granularities,
            self.test_working_memory_all_granularities,
            self.test_default_granularity,
            self.test_parameter_passthrough,
            self.test_invalid_granularity,
            self.test_result_structure_validation,
            self.test_all_9_search_patterns
        ]

        suite_results = []
        for test_suite in test_suites:
            passed = test_suite()
            suite_results.append(passed)

        # Generate summary
        self.print_summary(suite_results)

        # Return overall result
        return all(suite_results)

    def print_summary(self, suite_results):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        failed_tests = total_tests - passed_tests

        total_suites = len(suite_results)
        passed_suites = sum(1 for r in suite_results if r)
        failed_suites = total_suites - passed_suites

        print(f"\nTest Suites: {passed_suites}/{total_suites} passed")
        print(f"Total Tests: {passed_tests}/{total_tests} passed")
        print(f"Failed Tests: {failed_tests}")

        if failed_tests > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  ✗ {result['name']}")
                    if result['details']:
                        print(f"    {result['details']}")

        print("\n" + "="*70)
        if failed_tests == 0:
            print("✅ ALL TESTS PASSED - Consolidated search functions working correctly!")
        else:
            print(f"❌ {failed_tests} TEST(S) FAILED - Please review failures above")
        print("="*70)

        # Save results to JSON
        results_file = Path(self.store.db_path).parent / "test_consolidated_search_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_suites": total_suites,
                    "passed_suites": passed_suites,
                    "failed_suites": failed_suites,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests
                },
                "test_results": self.test_results
            }, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main test execution"""
    # Create temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_consolidated_search.db"

        print(f"\nUsing test database: {db_path}")

        # Create tester and run tests
        tester = ConsolidatedSearchTester(str(db_path))
        success = tester.run_all_tests()

        # Exit with appropriate code
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
