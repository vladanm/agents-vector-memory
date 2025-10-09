# Vector Memory Semantic Search Analysis Report
**Date**: 2025-10-08
**Purpose**: Expert analysis for research and specialist review
**System**: vector-memory-mcp server with sqlite-vec

---

## 1. Test Case Documentation

### Test Setup
- **Storage**: Working memory document about "Microservices Architecture Best Practices"
- **Document Structure**: 10 sections covering architecture principles
- **Chunking**: Auto-chunked into 11 chunks (average ~200-300 tokens per chunk)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Database**: SQLite with sqlite-vec extension for vector search
- **Session ID**: `comprehensive-search-test-20251008`
- **Agent ID**: `main-orchestrator`
- **Memory ID**: 445

### Document Content Summary
```
# Microservices Architecture Best Practices

## 1. Service Boundaries and Domain Design
Content about service boundaries, bounded contexts, DDD principles...

## 2. Communication Patterns
Content about synchronous/asynchronous communication, REST, gRPC, message queues...

## 3. Data Management
Content about database per service, event sourcing, CQRS, eventual consistency...

## 4. Deployment and Operations
Content about containerization, orchestration, CI/CD...

## 5. Resilience and Fault Tolerance
Content about circuit breakers, retries, timeouts, bulkheads...

## 6. Security
Content about authentication, authorization, API gateways, mTLS...

## 7. Monitoring and Observability
Content about logging, metrics, tracing, distributed tracing...

## 8. Testing Strategies
Content about unit testing, integration testing, contract testing...

## 9. API Design
Content about versioning, documentation, backward compatibility...

## 10. Organizational Aspects
Content about Conway's Law, team structure, DevOps culture...
```

---

## 2. Test Results by Granularity

### A. Fine Granularity (Specific Chunks)
**Purpose**: Search for exact information within small text segments
**Limit**: 10 chunks returned

#### Test Queries and Results

| Query | Expected Match | Result | Status |
|-------|---------------|--------|--------|
| "How should services communicate with each other?" | Section 2: Communication Patterns | ❌ 0 results | **FAILED** |
| "What are the best practices for microservices data management?" | Section 3: Data Management | ✅ Found chunk | **PASSED** |
| "Explain service boundaries in microservices" | Section 1: Service Boundaries | ❌ 0 results | **FAILED** |
| "What security measures should be implemented?" | Section 6: Security | ✅ Found chunk | **PASSED** |
| "How to handle failures in microservices?" | Section 5: Resilience | ✅ Found chunk | **PASSED** |
| "What testing strategies are recommended?" | Section 8: Testing | ❌ 0 results | **FAILED** |
| "How to monitor microservices?" | Section 7: Monitoring | ✅ Found chunk | **PASSED** |
| "API versioning best practices" | Section 9: API Design | ❌ 0 results | **FAILED** |
| "Team structure for microservices" | Section 10: Organizational | ✅ Found chunk | **PASSED** |

**Success Rate**: 5/9 = **55.6%**

#### Failed Query Examples with Context

**Failed Query #1**: "How should services communicate with each other?"
- **Expected**: Should match Section 2 chunks about REST, gRPC, message queues
- **Actual**: 0 results returned
- **Log Evidence**: `Filtered by similarity: 1` → 1 chunk found but rejected by threshold

**Failed Query #3**: "Explain service boundaries in microservices"
- **Expected**: Should match Section 1 chunks about bounded contexts, DDD
- **Actual**: 0 results returned
- **Log Evidence**: `Filtered by similarity: 1` → 1 chunk found but rejected by threshold

### B. Medium Granularity (Section Context)
**Purpose**: Search with expanded context (3 chunks before/after)
**Limit**: 5 results returned

#### Test Queries and Results

| Query | Expected Match | Result | Status |
|-------|---------------|--------|--------|
| "Communication patterns and data management strategies" | Sections 2 & 3 | ✅ Found chunks with context | **PASSED** |
| "Security and monitoring best practices" | Sections 6 & 7 | ✅ Found chunks with context | **PASSED** |
| "Testing and deployment strategies" | Sections 8 & 4 | ✅ Found chunks with context | **PASSED** |
| "Service design and organizational structure" | Sections 1 & 10 | ✅ Found chunks with context | **PASSED** |

**Success Rate**: 4/4 = **100%**

### C. Coarse Granularity (Full Documents)
**Purpose**: Retrieve entire documents matching query
**Limit**: 3 documents returned

#### Test Query and Result

| Query | Expected Match | Result | Status |
|-------|---------------|--------|--------|
| "Microservices architecture patterns and practices" | Full document (all 10 sections) | ✅ Found complete document | **PASSED** |

**Success Rate**: 1/1 = **100%**

---

## 3. Technical Analysis from Logs

### Log Evidence of Filtering Issues

**Example from Failed Fine Granularity Search:**
```
2025-10-08 17:17:41.670 [INFO] FILTERING RESULTS:
  Filtered by memory_type: 28
  Filtered by agent_id: 1
  Filtered by session_id: 0
  Filtered by session_iter: 0
  Filtered by task_code: 0
  Filtered by similarity: 1      ← ONE CHUNK REJECTED
  FINAL RESULTS: 0 chunks
```

**Interpretation**:
1. Vector search found 28 chunks with embedded vectors
2. Metadata filters reduced this to 1 chunk matching all criteria
3. **Similarity threshold rejected that 1 chunk**
4. Final result: 0 chunks returned to user

### Why Medium/Coarse Work But Fine Fails

**Medium Granularity Success Pattern:**
```
2025-10-08 17:18:13.343 [INFO] FILTERING RESULTS:
  Filtered by memory_type: 20
  Filtered by agent_id: 9
  Filtered by session_id: 0
  Filtered by similarity: 3      ← 3 chunks rejected
  FINAL RESULTS: 1 chunks
  Best similarity: 0.5798        ← ABOVE 0.5 threshold
```

**Key Difference**: Medium granularity had multiple candidate chunks (4 total), so even with 3 rejected, 1 passed the threshold.

---

## 4. Root Cause Analysis

### Primary Issue: Incorrect Similarity Formula

**Location**: `/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/src/session_memory_store.py:751`

**Current (Wrong) Implementation:**
```python
similarity = 1.0 - (distance / 2.0)
```

**Correct Implementation:**
```python
similarity = 1.0 - (distance**2 / 2.0)
```

### Mathematical Explanation

For **normalized embeddings** using **L2 distance**:

1. **Cosine Similarity Range**: [-1, 1] but normalized vectors → [0, 1]
2. **L2 Distance Range**: [0, 2] for normalized vectors
3. **Relationship**: `cosine_similarity = 1 - (L2_distance² / 2)`

**Why the formula matters:**

| L2 Distance | Wrong Formula | Correct Formula | Semantic Meaning |
|-------------|---------------|-----------------|------------------|
| 0.0 | 1.0 | 1.0 | Identical vectors |
| 0.5 | 0.75 | 0.875 | Very similar |
| 0.7 | 0.65 | 0.755 | Similar |
| 1.0 | 0.5 | 0.5 | Moderately similar |
| 1.4 | 0.3 | 0.02 | Dissimilar |
| 2.0 | 0.0 | -1.0 | Opposite |

**Impact**: The wrong formula **underestimates similarity scores**, causing semantically relevant chunks to be filtered out by the 0.5 threshold.

### Secondary Issue: Similarity Threshold

**Current Threshold**: 0.5
**Previous Threshold**: 0.7 (changed during debugging)

**Analysis**:
- With correct formula, 0.5 threshold = moderately strict
- With wrong formula, 0.5 threshold = very strict (filtering out relevant results)
- After fixing formula, threshold may need re-tuning based on desired precision/recall

---

## 5. My Suggestions

### Immediate Fixes (Already Applied)

✅ **Fix #1**: Corrected similarity formula on line 751
```python
# Changed from:
similarity = 1.0 - (distance / 2.0)
# To:
similarity = 1.0 - (distance**2 / 2.0)
```

### Recommended Next Steps

**1. Restart Server and Re-test**
- Restart MCP server to load corrected formula
- Re-run comprehensive test suite
- Expected: Fine granularity should improve from 56% to 80-90%+ success rate

**2. Threshold Optimization** (After formula fix)
- Test different thresholds: 0.3, 0.4, 0.5, 0.6
- Measure precision/recall trade-off
- Recommended starting point: 0.4 with correct formula

**3. Additional Enhancements** (Optional)
- Add debug logging for actual similarity scores in failed queries
- Implement A/B testing framework for threshold tuning
- Add similarity score distribution analysis tool

### Risk Assessment

**Low Risk**:
- Formula correction is mathematically correct
- No breaking changes to API or data structures
- Medium/coarse granularity already working

**Testing Required**:
- Verify fine granularity improvement
- Confirm no regression in medium/coarse
- Validate across different content types

---

## 6. Request for MCP-Agent Algorithm Specification

Now requesting comprehensive technical specification from mcp-coding-agent...
