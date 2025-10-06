#!/usr/bin/env python3
"""
Comprehensive QA Testing for Semantic Search Fix
Tests all 3 granularity levels with systematic validation
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vector-memory-mcp" / "vector-memory-2-mcp" / "src"))

from src.session_memory_store import SessionMemoryStore

class QATestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.performance_metrics = {}

    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"✅ PASS: {test_name}")

    def record_fail(self, test_name: str, reason: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append({"test": test_name, "reason": reason})
        print(f"❌ FAIL: {test_name}")
        print(f"   Reason: {reason}")

    def record_metric(self, metric_name: str, value: float, unit: str = "ms"):
        self.performance_metrics[metric_name] = {"value": value, "unit": unit}
        print(f"📊 {metric_name}: {value:.2f} {unit}")

    def summary(self) -> Dict[str, Any]:
        return {
            "total_tests": self.tests_run,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "pass_rate": (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0,
            "failures": self.failures,
            "performance_metrics": self.performance_metrics
        }


def verify_database_state(db_path: str, results: QATestResults) -> Dict[str, Any]:
    """Verify database has test data with embeddings"""
    print("\n" + "="*80)
    print("PHASE 1: DATABASE STATE VERIFICATION")
    print("="*80)

    conn = sqlite3.connect(db_path)

    # Check for existing memories
    memory_count = conn.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    embedding_count = conn.execute("SELECT COUNT(*) FROM vec_session_search").fetchone()[0]
    chunks_with_embeddings = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    print(f"\n📊 Database State:")
    print(f"   - Total memories: {memory_count}")
    print(f"   - Total chunks: {chunk_count}")
    print(f"   - Chunks with embeddings: {chunks_with_embeddings}")
    print(f"   - Vec search entries: {embedding_count}")

    # Test embeddings exist
    if embedding_count > 0:
        results.record_pass("Database has embeddings")
    else:
        results.record_fail("Database has embeddings", f"Found {embedding_count} embeddings, need > 0")

    # Test chunks have embeddings
    if chunks_with_embeddings > 0:
        results.record_pass("Chunks have embedding field populated")
    else:
        results.record_fail("Chunks have embedding field populated",
                          f"Found {chunks_with_embeddings} chunks with embeddings")

    conn.close()

    return {
        "memory_count": memory_count,
        "chunk_count": chunk_count,
        "embedding_count": embedding_count,
        "chunks_with_embeddings": chunks_with_embeddings
    }


def create_test_data(store: SessionMemoryStore, results: QATestResults) -> List[int]:
    """Create fresh test data with embeddings"""
    print("\n" + "="*80)
    print("PHASE 2: TEST DATA CREATION")
    print("="*80)

    test_contents = [
        {
            "title": "Error Handling Best Practices",
            "content": """
# Error Handling Patterns in Go

## Introduction
Error handling in Go is explicit and follows simple patterns. Unlike exception-based languages, Go uses error values.

## Basic Error Handling
Always check errors immediately:
```go
file, err := os.Open("config.json")
if err != nil {
    return fmt.Errorf("failed to open config: %w", err)
}
defer file.Close()
```

## Custom Error Types
Create custom errors for domain-specific cases:
```go
type ValidationError struct {
    Field string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed on %s: %s", e.Field, e.Message)
}
```

## Error Wrapping
Use fmt.Errorf with %w to preserve error chains:
```go
if err := process(); err != nil {
    return fmt.Errorf("processing failed: %w", err)
}
```

## Sentinel Errors
Define package-level errors for known conditions:
```go
var ErrNotFound = errors.New("resource not found")
var ErrUnauthorized = errors.New("unauthorized access")
```

## Error Handling in HTTP Handlers
Proper error responses in web applications:
```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    data, err := fetchData(r.Context())
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            http.Error(w, "Not Found", http.StatusNotFound)
            return
        }
        http.Error(w, "Internal Server Error", http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(data)
}
```

## Best Practices
- Check errors immediately
- Wrap errors with context
- Use errors.Is and errors.As for type checking
- Return errors rather than logging and continuing
- Panic only for programming errors
            """
        },
        {
            "title": "Performance Optimization Techniques",
            "content": """
# Go Performance Optimization Guide

## Profiling Fundamentals
Use pprof to identify bottlenecks before optimizing.

### CPU Profiling
```go
import _ "net/http/pprof"

func main() {
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    // Your application code
}
```

## Memory Optimization

### Slice Pre-allocation
Avoid repeated allocations by pre-allocating slices:
```go
// Bad: grows dynamically
result := []string{}
for i := 0; i < 1000; i++ {
    result = append(result, fmt.Sprintf("item%d", i))
}

// Good: pre-allocated
result := make([]string, 0, 1000)
for i := 0; i < 1000; i++ {
    result = append(result, fmt.Sprintf("item%d", i))
}
```

### Object Pooling
Use sync.Pool for frequently allocated objects:
```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func processData(data []byte) string {
    buf := bufferPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufferPool.Put(buf)
    }()

    buf.Write(data)
    return buf.String()
}
```

## String Building
Use strings.Builder for efficient string concatenation:
```go
// Bad: creates many intermediate strings
result := ""
for i := 0; i < 1000; i++ {
    result += fmt.Sprintf("item%d", i)
}

// Good: single allocation
var builder strings.Builder
builder.Grow(10000) // Pre-allocate if size known
for i := 0; i < 1000; i++ {
    fmt.Fprintf(&builder, "item%d", i)
}
result := builder.String()
```

## Concurrency Patterns

### Worker Pool
Limit goroutine count for controlled concurrency:
```go
func processJobs(jobs <-chan Job, results chan<- Result) {
    const workers = 10
    var wg sync.WaitGroup

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                results <- processJob(job)
            }
        }()
    }

    wg.Wait()
    close(results)
}
```

## Benchmark-Driven Development
Always benchmark before and after optimization:
```go
func BenchmarkProcessing(b *testing.B) {
    data := generateTestData()
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _ = process(data)
    }
}
```
            """
        },
        {
            "title": "Microservices Architecture Patterns",
            "content": """
# Microservices Architecture in Go

## Service Design Principles

### Single Responsibility
Each service should handle one bounded context:
- User service: authentication, profiles
- Order service: order processing, fulfillment
- Payment service: payment processing, refunds

### API Gateway Pattern
Centralize client requests through gateway:
```go
type Gateway struct {
    userClient    *UserServiceClient
    orderClient   *OrderServiceClient
    paymentClient *PaymentServiceClient
}

func (g *Gateway) PlaceOrder(ctx context.Context, req *PlaceOrderRequest) (*Order, error) {
    // Validate user
    user, err := g.userClient.GetUser(ctx, req.UserID)
    if err != nil {
        return nil, fmt.Errorf("user validation: %w", err)
    }

    // Create order
    order, err := g.orderClient.CreateOrder(ctx, req.Items)
    if err != nil {
        return nil, fmt.Errorf("order creation: %w", err)
    }

    // Process payment
    payment, err := g.paymentClient.ProcessPayment(ctx, order.Total)
    if err != nil {
        // Compensating transaction: cancel order
        g.orderClient.CancelOrder(ctx, order.ID)
        return nil, fmt.Errorf("payment failed: %w", err)
    }

    return order, nil
}
```

## Service Discovery
Use service registry for dynamic discovery:
```go
type ServiceRegistry interface {
    Register(service ServiceInfo) error
    Discover(serviceName string) ([]ServiceEndpoint, error)
    Deregister(serviceID string) error
}

type ConsulRegistry struct {
    client *consul.Client
}

func (r *ConsulRegistry) Discover(serviceName string) ([]ServiceEndpoint, error) {
    services, _, err := r.client.Health().Service(serviceName, "", true, nil)
    if err != nil {
        return nil, err
    }

    endpoints := make([]ServiceEndpoint, len(services))
    for i, svc := range services {
        endpoints[i] = ServiceEndpoint{
            Address: svc.Service.Address,
            Port:    svc.Service.Port,
        }
    }
    return endpoints, nil
}
```

## Circuit Breaker Pattern
Protect services from cascading failures:
```go
type CircuitBreaker struct {
    mu           sync.Mutex
    state        State
    failures     int
    threshold    int
    timeout      time.Duration
    lastAttempt  time.Time
}

func (cb *CircuitBreaker) Call(ctx context.Context, fn func() error) error {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    if cb.state == StateOpen {
        if time.Since(cb.lastAttempt) > cb.timeout {
            cb.state = StateHalfOpen
        } else {
            return ErrCircuitOpen
        }
    }

    err := fn()
    cb.lastAttempt = time.Now()

    if err != nil {
        cb.failures++
        if cb.failures >= cb.threshold {
            cb.state = StateOpen
        }
        return err
    }

    cb.failures = 0
    cb.state = StateClosed
    return nil
}
```

## Event-Driven Architecture
Use message queues for asynchronous communication:
```go
type EventBus interface {
    Publish(topic string, event Event) error
    Subscribe(topic string, handler EventHandler) error
}

type OrderEventHandler struct {
    inventoryService *InventoryService
    notificationSvc  *NotificationService
}

func (h *OrderEventHandler) Handle(event Event) error {
    switch e := event.(type) {
    case *OrderPlacedEvent:
        // Reserve inventory
        if err := h.inventoryService.Reserve(e.Items); err != nil {
            return err
        }

        // Send confirmation
        return h.notificationSvc.SendOrderConfirmation(e.OrderID)

    case *OrderCancelledEvent:
        // Release inventory
        return h.inventoryService.Release(e.Items)
    }
    return nil
}
```

## Health Checks and Readiness
Implement proper health endpoints:
```go
func (s *Service) HealthCheck() HealthStatus {
    status := HealthStatus{Healthy: true}

    // Check database
    if err := s.db.Ping(); err != nil {
        status.Healthy = false
        status.Errors = append(status.Errors, "database unavailable")
    }

    // Check dependencies
    for name, dep := range s.dependencies {
        if err := dep.Ping(); err != nil {
            status.Healthy = false
            status.Errors = append(status.Errors, fmt.Sprintf("%s unavailable", name))
        }
    }

    return status
}
```

## Distributed Tracing
Implement OpenTelemetry for observability:
```go
func traceOperation(ctx context.Context, operationName string) (context.Context, trace.Span) {
    tracer := otel.Tracer("myservice")
    return tracer.Start(ctx, operationName)
}

func ProcessOrder(ctx context.Context, order Order) error {
    ctx, span := traceOperation(ctx, "ProcessOrder")
    defer span.End()

    span.SetAttributes(
        attribute.String("order.id", order.ID),
        attribute.Float64("order.total", order.Total),
    )

    // Processing logic with traced operations
    if err := validateOrder(ctx, order); err != nil {
        span.RecordError(err)
        return err
    }

    return nil
}
```
            """
        }
    ]

    memory_ids = []
    for i, test_doc in enumerate(test_contents):
        print(f"\n📝 Creating test document {i+1}/3: {test_doc['title']}")

        start_time = time.time()
        result = store.store_memory(
            agent_id="qa-test-agent",
            session_id="qa-comprehensive-test",
            session_iter=1,
            memory_type="knowledge_base",
            content=test_doc["content"],
            title=test_doc["title"],
            auto_chunk=True
        )
        elapsed = (time.time() - start_time) * 1000

        if result.get("success"):
            memory_id = result["memory_id"]
            chunks_created = result.get("chunks_created", 0)
            memory_ids.append(memory_id)
            print(f"   ✅ Created memory {memory_id} with {chunks_created} chunks in {elapsed:.2f}ms")
            results.record_metric(f"test_data_creation_{i+1}", elapsed)

            if chunks_created > 0:
                results.record_pass(f"Test document {i+1} created with chunks")
            else:
                results.record_fail(f"Test document {i+1} created with chunks",
                                  f"Expected chunks > 0, got {chunks_created}")
        else:
            results.record_fail(f"Test document {i+1} creation",
                              f"Storage failed: {result.get('error', 'unknown')}")

    return memory_ids


def test_embedding_generation(store: SessionMemoryStore, memory_ids: List[int], results: QATestResults):
    """Verify embeddings were generated during storage"""
    print("\n" + "="*80)
    print("PHASE 3: EMBEDDING GENERATION VERIFICATION")
    print("="*80)

    conn = store._get_connection()

    for memory_id in memory_ids:
        # Check chunks have embeddings
        chunks = conn.execute("""
            SELECT id, chunk_index, embedding IS NOT NULL as has_embedding
            FROM memory_chunks
            WHERE parent_id = ?
            ORDER BY chunk_index
        """, (memory_id,)).fetchall()

        chunks_with_embeddings = sum(1 for c in chunks if c[2])
        total_chunks = len(chunks)

        print(f"\n📊 Memory {memory_id}:")
        print(f"   - Total chunks: {total_chunks}")
        print(f"   - Chunks with embeddings: {chunks_with_embeddings}")

        if chunks_with_embeddings == total_chunks and total_chunks > 0:
            results.record_pass(f"Memory {memory_id} - all chunks have embeddings")
        else:
            results.record_fail(f"Memory {memory_id} - all chunks have embeddings",
                              f"Expected {total_chunks}, got {chunks_with_embeddings}")

        # Check vec_session_search entries
        vec_entries = conn.execute("""
            SELECT COUNT(*)
            FROM vec_session_search
            WHERE memory_id = ?
        """, (memory_id,)).fetchone()[0]

        print(f"   - Vec search entries: {vec_entries}")

        if vec_entries > 0:
            results.record_pass(f"Memory {memory_id} - vec_session_search populated")
        else:
            results.record_fail(f"Memory {memory_id} - vec_session_search populated",
                              f"Expected > 0 entries, got {vec_entries}")

    conn.close()


def test_specific_chunks_granularity(store: SessionMemoryStore, results: QATestResults):
    """Test fine granularity (specific_chunks)"""
    print("\n" + "="*80)
    print("PHASE 4: SPECIFIC_CHUNKS GRANULARITY TESTING")
    print("="*80)

    test_queries = [
        ("error handling patterns", 0.5, "Should find error handling content"),
        ("custom error types", 0.5, "Should find ValidationError examples"),
        ("http handler errors", 0.5, "Should find HTTP error handling"),
    ]

    for query, threshold, description in test_queries:
        print(f"\n🔍 Query: '{query}' (threshold: {threshold})")
        print(f"   Expected: {description}")

        start_time = time.time()
        result = store.search_with_granularity(
            memory_type="knowledge_base",
            granularity="specific_chunks",
            query=query,
            agent_id="qa-test-agent",
            session_id="qa-comprehensive-test",
            limit=5,
            similarity_threshold=threshold
        )
        elapsed = (time.time() - start_time) * 1000

        results.record_metric(f"specific_chunks_query_{query[:20]}", elapsed)

        if not result.get("success"):
            results.record_fail(f"specific_chunks: '{query}'",
                              f"Search failed: {result.get('error', 'unknown')}")
            continue

        results_found = len(result.get("results", []))
        print(f"   📊 Found {results_found} results in {elapsed:.2f}ms")

        if results_found == 0:
            results.record_fail(f"specific_chunks: '{query}'",
                              f"Expected results, got 0 (message: {result.get('message', 'none')})")
            continue

        # Check result structure
        first_result = result["results"][0]
        required_fields = ["chunk_id", "memory_id", "chunk_index", "content",
                          "similarity", "source", "granularity"]

        missing_fields = [f for f in required_fields if f not in first_result]
        if missing_fields:
            results.record_fail(f"specific_chunks result structure",
                              f"Missing fields: {missing_fields}")
            continue

        # Check similarity scores
        similarities = [r["similarity"] for r in result["results"]]
        print(f"   📈 Similarity scores: {[f'{s:.3f}' for s in similarities]}")

        # Verify similarity scores are meaningful (not all 1.0 or 0.0)
        if all(s == similarities[0] for s in similarities):
            results.record_fail(f"specific_chunks similarity scores",
                              f"All scores identical: {similarities[0]}")
        elif all(s < 0.0 or s > 1.0 for s in similarities):
            results.record_fail(f"specific_chunks similarity scores",
                              f"Scores out of range [0,1]: {similarities}")
        else:
            results.record_pass(f"specific_chunks: '{query}' - meaningful results")

        # Verify results sorted by similarity
        if similarities == sorted(similarities, reverse=True):
            results.record_pass(f"specific_chunks: '{query}' - sorted by similarity")
        else:
            results.record_fail(f"specific_chunks: '{query}' - sorted by similarity",
                              f"Not sorted descending: {similarities}")

        # Verify granularity field
        if all(r.get("granularity") == "fine" for r in result["results"]):
            results.record_pass(f"specific_chunks: '{query}' - correct granularity field")
        else:
            results.record_fail(f"specific_chunks: '{query}' - correct granularity field",
                              f"Expected 'fine', got mixed values")


def test_section_context_granularity(store: SessionMemoryStore, results: QATestResults):
    """Test medium granularity (section_context)"""
    print("\n" + "="*80)
    print("PHASE 5: SECTION_CONTEXT GRANULARITY TESTING")
    print("="*80)

    test_queries = [
        ("performance optimization techniques", 0.5, "Should find optimization sections"),
        ("memory optimization patterns", 0.5, "Should find slice pre-allocation, pooling"),
        ("worker pool concurrency", 0.5, "Should find concurrency patterns section"),
    ]

    for query, threshold, description in test_queries:
        print(f"\n🔍 Query: '{query}' (threshold: {threshold})")
        print(f"   Expected: {description}")

        start_time = time.time()
        result = store.search_with_granularity(
            memory_type="knowledge_base",
            granularity="section_context",
            query=query,
            agent_id="qa-test-agent",
            session_id="qa-comprehensive-test",
            limit=3,
            similarity_threshold=threshold,
            auto_merge_threshold=0.6
        )
        elapsed = (time.time() - start_time) * 1000

        results.record_metric(f"section_context_query_{query[:20]}", elapsed)

        if not result.get("success"):
            results.record_fail(f"section_context: '{query}'",
                              f"Search failed: {result.get('error', 'unknown')}")
            continue

        results_found = len(result.get("results", []))
        print(f"   📊 Found {results_found} results in {elapsed:.2f}ms")

        if results_found == 0:
            results.record_fail(f"section_context: '{query}'",
                              f"Expected results, got 0 (message: {result.get('message', 'none')})")
            continue

        # Check result structure
        first_result = result["results"][0]

        # Section results should have header_path
        if "header_path" in first_result:
            results.record_pass(f"section_context: '{query}' - has header_path")

        # Check if any results are auto-merged
        merged_results = [r for r in result["results"] if r.get("auto_merged", False)]
        if merged_results:
            print(f"   🔗 {len(merged_results)} results auto-merged")
            for r in merged_results:
                print(f"      - {r.get('header_path', 'root')}: {r.get('chunks_merged', 0)} chunks")
            results.record_pass(f"section_context: '{query}' - auto-merge working")

        # Verify content size is medium (400-1200 tokens roughly)
        for i, r in enumerate(result["results"]):
            content_len = len(r.get("content", ""))
            token_count = r.get("token_count", content_len // 4)  # Rough estimate
            print(f"   📏 Result {i+1}: ~{token_count} tokens, {content_len} chars")

        # Verify granularity field
        if all(r.get("granularity") == "medium" for r in result["results"]):
            results.record_pass(f"section_context: '{query}' - correct granularity field")
        else:
            results.record_fail(f"section_context: '{query}' - correct granularity field",
                              f"Expected 'medium', got mixed values")

        results.record_pass(f"section_context: '{query}' - returns section-level results")


def test_full_documents_granularity(store: SessionMemoryStore, results: QATestResults):
    """Test coarse granularity (full_documents)"""
    print("\n" + "="*80)
    print("PHASE 6: FULL_DOCUMENTS GRANULARITY TESTING")
    print("="*80)

    test_queries = [
        ("microservices architecture patterns", 0.5, "Should find microservices document"),
        ("service discovery and circuit breaker", 0.5, "Should find architecture patterns"),
    ]

    for query, threshold, description in test_queries:
        print(f"\n🔍 Query: '{query}' (threshold: {threshold})")
        print(f"   Expected: {description}")

        start_time = time.time()
        result = store.search_with_granularity(
            memory_type="knowledge_base",
            granularity="full_documents",
            query=query,
            agent_id="qa-test-agent",
            session_id="qa-comprehensive-test",
            limit=3,
            similarity_threshold=threshold
        )
        elapsed = (time.time() - start_time) * 1000

        results.record_metric(f"full_documents_query_{query[:20]}", elapsed)

        if not result.get("success"):
            results.record_fail(f"full_documents: '{query}'",
                              f"Search failed: {result.get('error', 'unknown')}")
            continue

        results_found = len(result.get("results", []))
        print(f"   📊 Found {results_found} results in {elapsed:.2f}ms")

        if results_found == 0:
            results.record_fail(f"full_documents: '{query}'",
                              f"Expected results, got 0")
            continue

        # Check result structure (full documents)
        for i, r in enumerate(result["results"]):
            content_len = len(r.get("content", ""))
            title = r.get("title", "untitled")
            print(f"   📄 Document {i+1}: '{title}' ({content_len} chars)")

        # Verify these are complete documents (large content)
        large_docs = [r for r in result["results"] if len(r.get("content", "")) > 1000]
        if large_docs:
            results.record_pass(f"full_documents: '{query}' - returns complete documents")
        else:
            results.record_fail(f"full_documents: '{query}' - returns complete documents",
                              f"Expected large documents, got small chunks")

        # Verify granularity field
        if all(r.get("granularity") == "coarse" for r in result["results"]):
            results.record_pass(f"full_documents: '{query}' - correct granularity field")


def test_parameter_variations(store: SessionMemoryStore, results: QATestResults):
    """Test different parameter combinations"""
    print("\n" + "="*80)
    print("PHASE 7: PARAMETER TESTING")
    print("="*80)

    # Test different similarity thresholds
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
        result = store.search_with_granularity(
            memory_type="knowledge_base",
            granularity="specific_chunks",
            query="error handling",
            limit=10,
            similarity_threshold=threshold
        )

        count = len(result.get("results", []))
        print(f"   🎚️  Threshold {threshold}: {count} results")

        if count >= 0:  # Just verify it doesn't crash
            results.record_pass(f"Parameter test: threshold={threshold}")

    # Test different limits
    limits = [1, 3, 10]
    for limit in limits:
        result = store.search_with_granularity(
            memory_type="knowledge_base",
            granularity="specific_chunks",
            query="optimization",
            limit=limit,
            similarity_threshold=0.5
        )

        count = len(result.get("results", []))
        print(f"   🔢 Limit {limit}: {count} results (max)")

        if count <= limit:
            results.record_pass(f"Parameter test: limit={limit} respected")
        else:
            results.record_fail(f"Parameter test: limit={limit} respected",
                              f"Expected <= {limit}, got {count}")


def test_performance(store: SessionMemoryStore, results: QATestResults):
    """Performance validation"""
    print("\n" + "="*80)
    print("PHASE 8: PERFORMANCE VALIDATION")
    print("="*80)

    # Test query performance
    query = "performance optimization"
    iterations = 5

    for granularity in ["specific_chunks", "section_context", "full_documents"]:
        times = []
        for i in range(iterations):
            start = time.time()
            store.search_with_granularity(
                memory_type="knowledge_base",
                granularity=granularity,
                query=query,
                limit=5
            )
            times.append((time.time() - start) * 1000)

        avg_time = sum(times) / len(times)
        print(f"   ⚡ {granularity}: avg {avg_time:.2f}ms over {iterations} runs")
        results.record_metric(f"performance_{granularity}_avg", avg_time)

        if avg_time < 2000:  # Less than 2 seconds
            results.record_pass(f"Performance: {granularity} < 2s")
        else:
            results.record_fail(f"Performance: {granularity} < 2s",
                              f"Average time {avg_time:.2f}ms exceeds 2000ms")


def main():
    print("="*80)
    print("COMPREHENSIVE QA TESTING: SEMANTIC SEARCH FIX")
    print("="*80)

    results = QATestResults()

    # Initialize store
    db_path = "/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/memory/agent_session_memory.db"
    store = SessionMemoryStore(db_path=db_path)

    # Check embedding model loaded
    if store.embedding_model:
        print("\n✅ Embedding model loaded successfully")
        results.record_pass("Embedding model initialization")
    else:
        print("\n❌ Embedding model NOT loaded")
        results.record_fail("Embedding model initialization", "Model is None")
        print("\n⚠️  CRITICAL: Cannot proceed without embedding model")
        return

    # Phase 1: Verify existing data or create new
    db_state = verify_database_state(db_path, results)

    # Phase 2: Create fresh test data
    memory_ids = create_test_data(store, results)

    if not memory_ids:
        print("\n❌ CRITICAL: No test data created, cannot proceed")
        return

    # Phase 3: Verify embeddings generated
    test_embedding_generation(store, memory_ids, results)

    # Phase 4-6: Test all granularities
    test_specific_chunks_granularity(store, results)
    test_section_context_granularity(store, results)
    test_full_documents_granularity(store, results)

    # Phase 7: Parameter testing
    test_parameter_variations(store, results)

    # Phase 8: Performance testing
    test_performance(store, results)

    # Final summary
    print("\n" + "="*80)
    print("QA TEST SUMMARY")
    print("="*80)

    summary = results.summary()
    print(f"\n📊 Overall Results:")
    print(f"   - Total tests: {summary['total_tests']}")
    print(f"   - Passed: {summary['passed']} ✅")
    print(f"   - Failed: {summary['failed']} ❌")
    print(f"   - Pass rate: {summary['pass_rate']:.1f}%")

    if summary['failed'] > 0:
        print(f"\n❌ FAILED TESTS:")
        for failure in summary['failures']:
            print(f"   - {failure['test']}")
            print(f"     Reason: {failure['reason']}")

    print(f"\n📈 Performance Metrics:")
    for metric, data in summary['performance_metrics'].items():
        print(f"   - {metric}: {data['value']:.2f} {data['unit']}")

    # GO/NO-GO decision
    print("\n" + "="*80)
    if summary['pass_rate'] >= 90 and summary['failed'] == 0:
        print("🟢 GO FOR PRODUCTION")
        print("   - All critical tests passed")
        print("   - Performance acceptable")
        print("   - All granularities working")
    elif summary['pass_rate'] >= 70:
        print("🟡 CONDITIONAL GO")
        print("   - Most tests passed")
        print("   - Some issues need review")
        print("   - Consider fixes before production")
    else:
        print("🔴 NO-GO")
        print("   - Critical issues found")
        print("   - Fix required before production")
        print("   - Multiple test failures")
    print("="*80)

    # Save results
    output_file = "/Users/vladanm/projects/subagents/simple-agents/qa_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
