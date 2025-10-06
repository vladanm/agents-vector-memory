#!/usr/bin/env python3
"""Create comprehensive test data with embeddings for semantic search verification."""

import sys
import time
sys.path.insert(0, '/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/src')

from session_memory_store import SessionMemoryStore

def main():
    # Initialize store
    store = SessionMemoryStore()

    # Create Knowledge Base: Go Programming Best Practices
    kb_content = '''# Go Programming Best Practices and Patterns

## Introduction to Idiomatic Go

Go (Golang) is a statically typed, compiled programming language designed at Google. Understanding idiomatic patterns is crucial for writing maintainable, performant code. This guide covers essential best practices that every Go developer should master.

## Error Handling Strategies

### The Error Interface

Go's error handling is explicit and values-based. The built-in error interface is simple:

```go
type error interface {
    Error() string
}
```

This minimalist design encourages explicit error checking at every level. Unlike exceptions in other languages, Go errors are regular values that can be inspected, wrapped, and propagated with full control.

### Error Wrapping with fmt.Errorf

Since Go 1.13, error wrapping has become standard practice. Use the %w verb to wrap errors while preserving the error chain:

```go
if err := os.Open(filename); err != nil {
    return fmt.Errorf("failed to open config file: %w", err)
}
```

This allows downstream code to use errors.Is() and errors.As() to inspect the error chain. Wrapping provides context while maintaining the original error type, enabling sophisticated error handling strategies.

### Custom Error Types

For domain-specific errors, create custom types that implement the error interface:

```go
type ValidationError struct {
    Field string
    Value interface{}
    Constraint string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field '%s': value %v violates %s",
        e.Field, e.Value, e.Constraint)
}
```

Custom error types enable type-safe error handling and rich error information that can be programmatically inspected.

## Concurrency Patterns

### Goroutines and Channel Communication

Goroutines are lightweight threads managed by the Go runtime. Channels provide type-safe communication between goroutines following the principle "Don't communicate by sharing memory; share memory by communicating."

```go
func worker(jobs <-chan int, results chan<- int) {
    for job := range jobs {
        results <- process(job)
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // Start 3 workers
    for w := 0; w < 3; w++ {
        go worker(jobs, results)
    }

    // Send work
    for j := 0; j < 9; j++ {
        jobs <- j
    }
    close(jobs)
}
```

### Context for Cancellation

The context package provides cancellation signals and deadlines across API boundaries and goroutines:

```go
func processRequest(ctx context.Context, req *Request) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    select {
    case <-ctx.Done():
        return ctx.Err()
    case result := <-process(req):
        return result
    }
}
```

Context propagation is essential for graceful shutdowns, request timeouts, and distributed tracing.

### Sync Package Primitives

For protecting shared state, use sync.Mutex and sync.RWMutex:

```go
type SafeCounter struct {
    mu sync.Mutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}
```

Use sync.Once for one-time initialization, sync.Pool for object reuse, and sync.WaitGroup for goroutine coordination.

## Memory Management and Performance

### Understanding Escape Analysis

Go's compiler performs escape analysis to determine whether variables should be stack or heap allocated:

```go
// Stack allocated - pointer doesn't escape
func stackAlloc() int {
    x := 42
    return x
}

// Heap allocated - pointer escapes via return
func heapAlloc() *int {
    x := 42
    return &x
}
```

Use `go build -gcflags='-m'` to see escape analysis decisions. Minimizing heap allocations improves performance by reducing GC pressure.

### Slice Capacity Management

Pre-allocate slices when size is known to avoid repeated allocations:

```go
// Inefficient - grows incrementally
results := []Result{}
for _, item := range items {
    results = append(results, process(item))
}

// Efficient - single allocation
results := make([]Result, 0, len(items))
for _, item := range items {
    results = append(results, process(item))
}
```

### String Building Performance

For concatenating multiple strings, use strings.Builder to avoid allocating intermediate strings:

```go
var builder strings.Builder
builder.Grow(estimatedSize) // Pre-allocate if size known
for _, s := range strings {
    builder.WriteString(s)
}
result := builder.String()
```

This is orders of magnitude faster than repeated string concatenation with the + operator.

## Testing Methodology

### Table-Driven Tests

Go's testing style emphasizes table-driven tests for comprehensive coverage:

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive", 2, 3, 5},
        {"negative", -1, -1, -2},
        {"mixed", 5, -3, 2},
        {"zero", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            if got != tt.want {
                t.Errorf("Add(%d, %d) = %d; want %d",
                    tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

This pattern provides clear test organization, easy addition of cases, and descriptive failure messages.

### Benchmarking

Go's built-in benchmarking measures performance precisely:

```go
func BenchmarkStringConcat(b *testing.B) {
    for i := 0; i < b.N; i++ {
        result := ""
        for j := 0; j < 100; j++ {
            result += "test"
        }
    }
}

func BenchmarkStringBuilder(b *testing.B) {
    for i := 0; i < b.N; i++ {
        var builder strings.Builder
        for j := 0; j < 100; j++ {
            builder.WriteString("test")
        }
        _ = builder.String()
    }
}
```

Run with `go test -bench=. -benchmem` to see allocations and memory usage.

## Interface Design Principles

### Accept Interfaces, Return Structs

This principle promotes flexibility and testability:

```go
// Bad - forces implementation details on caller
func ProcessFile(f *os.File) error {
    // ...
}

// Good - accepts any reader
func ProcessFile(r io.Reader) error {
    // ...
}
```

Callers can pass files, network connections, buffers, or mock implementations.

### Small, Focused Interfaces

The smaller the interface, the easier it is to implement:

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
```

Go's standard library demonstrates this principle throughout - most interfaces have 1-3 methods.

## Conclusion

Mastering these Go best practices enables writing efficient, maintainable, idiomatic code. Focus on explicit error handling, thoughtful concurrency, performance-conscious memory management, comprehensive testing, and interface-based design. These patterns have been battle-tested across millions of Go applications and represent the community's collective wisdom.
'''

    print('Creating Knowledge Base document...')
    kb_result = store.store_knowledge_base(
        agent_id='test-agent-final',
        session_id='semantic-search-fix-20251006',
        session_iter=1,
        task_code='test-data-creation',
        content=kb_content,
        title='Go Programming Best Practices',
        description='Comprehensive guide to idiomatic Go patterns',
        tags=['go', 'best-practices', 'programming'],
        auto_chunk=True
    )
    print(f'Knowledge Base stored: memory_id={kb_result["memory_id"]}')
    time.sleep(0.5)

    # Create Report: System Performance Analysis
    report_content = '''# Production System Performance Analysis Report

## Executive Summary

This report analyzes the performance characteristics of our Go-based microservices infrastructure serving 50M daily active users. Key findings indicate memory allocation patterns and goroutine management require optimization to meet our 99.9% availability SLA.

## System Architecture Overview

### Service Topology

Our production environment consists of 45 microservices deployed across 3 AWS regions:

- API Gateway Layer: 12 instances handling 100K req/sec
- Business Logic Services: 25 services with domain-specific responsibilities
- Data Access Layer: 8 services managing database interactions
- Background Workers: Queue processors and scheduled jobs

All services communicate via gRPC with circuit breakers for fault tolerance. Service mesh provides observability through distributed tracing and metrics collection.

### Technology Stack

- Runtime: Go 1.21.5 on Ubuntu 22.04 LTS
- Container Platform: Kubernetes 1.28 with custom operators
- Database: PostgreSQL 15 for transactional data, Redis 7.2 for caching
- Message Queue: Apache Kafka for event streaming
- Observability: Prometheus + Grafana + Jaeger

## Performance Metrics Analysis

### Response Time Distribution

Analyzed 7 days of production traffic (500M requests total):

P50 Latency: 45ms (target: <50ms) - GOOD
- Median response time meets SLA
- Indicates healthy performance for majority of requests
- No significant degradation under normal load

P95 Latency: 180ms (target: <200ms) - GOOD
- 95th percentile within acceptable range
- Some requests experiencing increased latency
- Suggests minor optimization opportunities

P99 Latency: 850ms (target: <500ms) - NEEDS ATTENTION
- Tail latency exceeds target by 70%
- Critical issue affecting 5M requests per day
- Primary focus area for optimization

P99.9 Latency: 2.8s (target: <1s) - CRITICAL ISSUE
- Severe tail latency outliers
- Impacts high-value user transactions
- Requires immediate investigation

### CPU Utilization Patterns

Average CPU usage across all services: 42% (healthy baseline)

Problematic Services:
- UserAuthService: 78% average, 95% peak (inefficient JWT validation)
- PaymentProcessor: 82% average, 98% peak (CPU-bound cryptographic operations)
- RecommendationEngine: 71% average, 88% peak (excessive JSON marshaling)

Root Causes Identified:
1. Synchronous cryptographic operations blocking goroutines
2. Repeated parsing of identical JWT tokens (missing cache)
3. JSON serialization in hot paths (consider protobuf)

### Memory Allocation Behavior

Heap Memory Profile:
- Total allocated: 145GB across all service instances
- GC pressure: 18% of CPU time spent in garbage collection
- Allocation rate: 2.1GB/sec sustained during peak hours

Top Allocators (pprof analysis):

1. JSON Encoding (35% of allocations)
   - 720MB/sec in encoding.Marshal operations
   - Repeated allocation of intermediate buffers
   - Recommendation: Use sync.Pool for encoder reuse

2. String Concatenation (22% of allocations)
   - 460MB/sec from + operator string building
   - Hot paths identified in logging and error messages
   - Recommendation: Use strings.Builder with pre-allocation

3. Slice Growth (18% of allocations)
   - 378MB/sec from append operations without capacity hints
   - Database result sets and API response building
   - Recommendation: Pre-allocate slices with make([]T, 0, capacity)

4. HTTP Request/Response Buffers (15% of allocations)
   - 315MB/sec from http.Request and http.Response objects
   - Connection pooling parameters need tuning
   - Recommendation: Increase http.Transport.MaxIdleConnsPerHost

### Goroutine Management

Goroutine Count Analysis:
- Average goroutines per service: 2,400 (baseline)
- Peak goroutines observed: 45,000 (during traffic spike)
- Goroutine leak detected in 3 services

Identified Issues:

1. Goroutine Leaks in UserService
   - 12,000 goroutines stuck waiting on closed channels
   - Missing context cancellation in background workers
   - Leaked goroutines accumulate 2GB memory over 72 hours

2. Unbounded Goroutine Creation
   - PaymentProcessor spawns goroutine per request without limit
   - No worker pool or semaphore limiting concurrency
   - Under load, creates 30K+ goroutines causing scheduler thrashing

3. Missing Timeouts
   - HTTP clients lacking context.WithTimeout
   - Database queries without query timeout configuration
   - Goroutines blocked indefinitely on slow operations

## Database Query Performance

### Slow Query Analysis

Identified 127 queries exceeding 500ms execution time:

Top Offenders:

1. User Timeline Query (2.3s average)
   - Full table scan on 500M row table
   - Missing composite index on (user_id, created_at)
   - Executes 50K times per minute during peak

2. Analytics Aggregation (1.8s average)
   - Complex GROUP BY with multiple joins
   - No materialized view or pre-computed rollups
   - Blocks connection pool threads

3. Search Functionality (1.2s average)
   - LIKE queries on text columns without full-text index
   - Should migrate to Elasticsearch for text search

### Connection Pool Exhaustion

Database connection pools frequently saturated:

- Max connections: 100 per service instance
- Average wait time for connection: 120ms
- Connection timeouts during traffic spikes: 2.4% of requests

Recommendations:
- Increase connection pool size to 200
- Implement exponential backoff for retries
- Consider read replicas for read-heavy queries

## Network I/O Patterns

### gRPC Communication Overhead

Inter-service communication analysis reveals inefficiencies:

Chattiness Issues:
- Average requests per user transaction: 14 service calls
- N+1 query pattern detected in 8 API endpoints
- Excessive network round-trips inflating latency

Example: User Profile Load requires 7 sequential calls adding 280ms latency. Should batch into 2-3 calls.

Serialization Overhead:
- JSON marshaling: 8ms average per request
- Protobuf alternative: 0.8ms (10x faster)
- Recommendation: Migrate internal services to protobuf

### HTTP Client Configuration

Default http.Client settings causing issues:

- Default timeout: 30s (too long for synchronous calls)
- No connection reuse (creating new TCP connection per request)
- Missing keep-alive configuration

## Recommendations and Action Items

### Priority 1: Critical Issues (Implement within 1 week)

1. Fix Goroutine Leaks
   - Add context cancellation to all background workers
   - Implement timeout for all blocking operations
   - Expected impact: -2GB memory, +5% stability

2. Database Index Optimization
   - Create composite index on timeline query
   - Expected impact: 2.3s to 50ms query time

3. Implement Worker Pools
   - Limit concurrent goroutines with semaphore
   - Expected impact: -60% goroutine count, +20% throughput

### Priority 2: Performance Optimizations (Implement within 1 month)

1. Memory Allocation Reduction
   - Use sync.Pool for JSON encoders
   - Replace string concatenation with strings.Builder
   - Expected impact: -35% allocations, -8% GC time

2. Connection Pool Tuning
   - Double database connection limits
   - Optimize HTTP client keep-alive settings
   - Expected impact: -70% connection timeouts

3. Caching Layer Enhancement
   - Cache JWT validation results (5min TTL)
   - Cache database query results for read-heavy endpoints
   - Expected impact: -30% database load, -40ms P95 latency

### Priority 3: Architectural Improvements (Plan for next quarter)

1. Service Mesh Optimization
   - Batch API calls to reduce network round-trips
   - Implement GraphQL federation for client-side queries

2. Migration to Protobuf
   - Replace JSON with protobuf for inter-service communication
   - Expected impact: -85% serialization overhead

3. Elasticsearch for Text Search
   - Migrate LIKE queries to Elasticsearch
   - Expected impact: 1.2s to 50ms search latency

## Conclusion

Performance analysis reveals several optimization opportunities across our Go microservices infrastructure. Critical issues around goroutine leaks and database query performance require immediate attention. Memory allocation patterns show room for 35% reduction through better resource pooling. Implementing the Priority 1 recommendations should improve P99 latency from 850ms to under 500ms, meeting our SLA targets.

Next steps: Create JIRA tickets for all action items and schedule implementation sprints.
'''

    print('Creating Report document...')
    report_result = store.store_report(
        agent_id='test-agent-final',
        session_id='semantic-search-fix-20251006',
        session_iter=1,
        task_code='test-data-creation',
        content=report_content,
        title='Production System Performance Analysis',
        description='Performance analysis of Go microservices with optimization recommendations',
        tags=['performance', 'analysis', 'go', 'microservices'],
        auto_chunk=True
    )
    print(f'Report stored: memory_id={report_result["memory_id"]}')
    time.sleep(0.5)

    # Create Working Memory: Debugging Insights (shortened to avoid issues)
    wm_content = '''# Debugging Session: Race Condition in Payment Service

## Problem Statement

Production incident on 2025-10-05 23:47 UTC: PaymentService experiencing intermittent data corruption affecting 0.3% of transactions (approximately 450 payments per hour). Race detector triggered in staging environment but not consistently reproducible.

## Initial Investigation

### Symptom Analysis

Observed Behavior:
- Payment records showing incorrect amounts (off by 1-2 cents)
- Transaction status flipping between completed and pending
- Rare panics with "concurrent map write" errors
- Issues only occur under load (>500 req/sec)

Error Logs show concurrent map write panics from multiple goroutines accessing shared payment cache.

### Code Review of Payment Handler

Original Implementation (BUGGY):
The service maintained an unprotected map of payment objects that multiple goroutines could access simultaneously. The cache map lacked any mutex protection, and individual Payment objects were modified by multiple goroutines without synchronization.

Race Conditions Identified:

1. Concurrent Map Access: Multiple goroutines reading/writing cache without lock
2. Shared Payment Object: Payment struct modified by multiple goroutines
3. Missing Synchronization: No mutex protecting shared state

## Debugging Techniques Applied

### Race Detector Usage

Compiled service with race detector enabled using go build -race flag. The race detector pinpointed exact lines where concurrent access occurred, specifically identifying that the Status field was being written by two goroutines simultaneously.

This was our "gotcha" moment - realizing the Status field race condition was the root cause.

### Load Testing with Race Detection

Created test that reliably reproduces the race by spawning 100 concurrent payment requests with overlapping payment IDs. Running with -race flag consistently triggered race warnings. Key insight: Using modulo arithmetic to create overlapping payment IDs dramatically increased race likelihood.

### pprof Goroutine Analysis

Captured goroutine profile during incident showing 450 goroutines blocked on map access with contention hotspot at cache assignment line. Goroutines waiting average 120ms for access. This visualization showed the severity of the contention issue.

## Root Cause Analysis

Primary Issue: Unprotected concurrent access to shared map and Payment objects

Why It Happens Intermittently:
- Under low load, goroutines rarely overlap on same payment ID
- Under high load (>500 req/sec), probability of concurrent access increases
- Map corruption only occurs with exact timing of simultaneous writes

Why Amount Corruption Occurs:
- Payment.Amount field not atomically updated
- Partial writes when goroutine interrupted mid-write
- Example: Writing 100.50 might get interrupted after writing 100, leaving 100.00

The "Gotcha" Moment:

The real gotcha was realizing that even read-only map access requires synchronization. We thought reads were safe, but Go's map implementation can panic even on concurrent read+write. From Go spec: "If one goroutine is writing to a map, no other goroutine should be reading or writing the map concurrently."

## Solution Implementation

### Fixed Implementation with Proper Synchronization

The fix introduced two levels of mutex protection:

1. Service-level sync.RWMutex protecting the cache map
2. Payment-level sync.Mutex protecting individual payment fields

Key improvements:
- RWMutex allows multiple concurrent readers with exclusive writers
- Payment-level mutex protects individual payment state changes
- Double-check pattern prevents race during payment creation
- Thread-safe accessor methods for all payment field access

Alternative considered: sync.Map for cache operations, which provides lock-free reads and atomic LoadOrStore operation. This is beneficial when reads significantly outnumber writes.

## Testing and Verification

### Race-Free Verification

Ran go test -race -count=100 across all tests. Results: 0 race conditions detected, all 2,347 tests passing, concurrent map access tests stable.

### Load Testing Results

Before Fix:
- Race conditions: 450 per hour under load
- Data corruption: 0.3% of transactions
- Panics: 12 per day

After Fix:
- Race conditions: 0 (confirmed with -race for 72 hours)
- Data corruption: 0%
- Panics: 0

### Performance Impact

Benchmarks showed minimal performance impact:

Before (no locks): 1052 ns/op
After (with RWMutex): 1089 ns/op
Performance Impact: +3.5% latency (37ns per operation)

This minimal overhead is acceptable given the correctness guarantees.

### Production Rollout

Deployment Strategy:
1. Deploy to canary environment (5% traffic) - monitored 24 hours
2. Gradual rollout: 25% to 50% to 100% over 3 days
3. Monitor error rates, latency, and goroutine counts

Production Results (7 days post-deployment):
- Data corruption incidents: 0 (down from 450/hour)
- Average latency: +2ms (+1.9% increase)
- CPU usage: unchanged
- Memory usage: +0.8% (from additional mutexes)

## Lessons Learned

### Key Takeaways

1. Always Use Race Detector in Development
   - go test -race should be part of CI/CD pipeline
   - Catches subtle concurrency bugs before production

2. Map Access Always Requires Synchronization
   - Even reads must be protected if any goroutine writes
   - Go's map is not concurrency-safe by design

3. Shared Mutable State Needs Protection
   - Any struct field modified by multiple goroutines needs mutex
   - Consider immutable data structures where possible

4. Lock Granularity Matters
   - Service-level mutex for cache access
   - Payment-level mutex for payment state
   - Avoids contention across unrelated payments

5. Benchmarks Reveal Performance Impact
   - Measure before assuming locks are "too slow"
   - 3.5% overhead is negligible compared to correctness

### Debugging Techniques That Helped

1. Race Detector: Pinpointed exact source of concurrent access
2. Load Testing: Reliably reproduced intermittent issue
3. pprof Goroutine Analysis: Visualized contention hotspots
4. Structured Logging: Correlated errors with specific payments

### Common Pitfalls to Avoid

Do NOT: Share maps between goroutines without locks
Do NOT: Assume reads are safe without synchronization
Do NOT: Modify struct fields from multiple goroutines unsynchronized
Do NOT: Ignore race detector warnings (they are always real bugs)

DO: Use sync.RWMutex for shared state
DO: Run tests with -race flag regularly
DO: Consider sync.Map for high-contention caches
DO: Use atomic operations for simple counters
DO: Design for immutability when possible

## Conclusion

This debugging session revealed a classic race condition caused by unprotected concurrent access to shared state. The race detector was instrumental in identifying the exact location of the bug. Fixing required adding proper synchronization with mutexes at both service and payment levels. The solution added minimal performance overhead (3.5%) while completely eliminating data corruption issues.

Key Insight: Go's race detector is your best friend for finding concurrency bugs. Use it liberally during development and testing. The small compilation time overhead is worth catching bugs that would be nearly impossible to debug in production.
'''

    print('Creating Working Memory document...')
    wm_result = store.store_working_memory(
        agent_id='test-agent-final',
        session_id='semantic-search-fix-20251006',
        session_iter=1,
        task_code='test-data-creation',
        content=wm_content,
        title='Debugging Race Condition in Payment Service',
        description='Detailed debugging session for race condition with solutions and lessons learned',
        tags=['debugging', 'race-condition', 'concurrency', 'go'],
        auto_chunk=True
    )
    print(f'Working Memory stored: memory_id={wm_result["memory_id"]}')

    print('\n' + '='*60)
    print('ALL 3 TEST DOCUMENTS CREATED SUCCESSFULLY')
    print('='*60)
    print(f'\nMemory IDs:')
    print(f'  Knowledge Base: {kb_result["memory_id"]}')
    print(f'  Report: {report_result["memory_id"]}')
    print(f'  Working Memory: {wm_result["memory_id"]}')
    print(f'\nSession: semantic-search-fix-20251006')
    print(f'Agent: test-agent-final')
    print(f'Task Code: test-data-creation')

if __name__ == '__main__':
    main()
