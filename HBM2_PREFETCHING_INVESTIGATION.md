# HBM2 Prefetching Investigation

## Overview

This document details the investigation into HBM2 prefetching regression and the implementation of enhanced memory profiling for gfx906 (Vega20/MI50/Radeon VII).

## HBM2 Characteristics

**gfx906 HBM2 Memory:**
- **Latency**: ~300-400 cycles (~194ns at 1.8GHz) - **Higher than GDDR5** (~100-200 cycles)
- **Bandwidth**: ~1TB/s theoretical peak
- **Cache**: 4MB L2 cache
- **Impact**: Higher latency means prefetching is more critical, but must be done correctly

## Previous Prefetching Implementation

**Status**: DISABLED due to performance regression

**Previous Approach:**
- Used `__builtin_prefetch(addr, 0, 2)` (read, moderate locality)
- Prefetched 1 iteration ahead in MMQ kernels
- **Result**: Caused performance regression (pp512 decreased)

**Possible Causes:**
1. **Cache Pollution**: Prefetching too aggressively, evicting useful data
2. **Wrong Prefetch Distance**: 1 iteration may not be enough to hide 300-400 cycle latency
3. **Stale Data**: Prefetching addresses that won't be used
4. **AMD Compiler Issues**: `__builtin_prefetch` may not be optimized effectively for HBM2
5. **Memory Conflicts**: Prefetching interfering with actual memory access patterns

## New Prefetching Strategy

**Implementation**: Controlled software prefetching

**Key Changes:**
1. **Environment Variable Control**: `GGML_HIP_PREFETCH_ENABLE=1` (experimental)
2. **Software Prefetching**: Use explicit volatile loads instead of `__builtin_prefetch`
3. **Prefetch Distance**: 1-2 iterations ahead (nrows*nwarps)
4. **Selective Prefetching**: Only prefetch when there's a next iteration

**Code Location:**
- `ggml/src/ggml-cuda/common.cuh`: `ggml_cuda_prefetch_hbm2()` function
- `ggml/src/ggml-cuda/mmq.cuh`: Prefetch calls in `load_tiles_q8_0()` and `load_tiles_mxfp4()`

## Enhanced Memory Profiling

**New Profiling Features:**

1. **HBM2-Specific Metrics:**
   - HBM2 latency analysis (~194ns per access)
   - Memory operation counting
   - Estimated memory-bound time (assuming 10% cache miss rate)
   - Bandwidth utilization estimates

2. **Bottleneck Identification:**
   - Cache pressure warnings
   - Memory latency bottleneck detection
   - Access pattern analysis (coalesced vs scattered)
   - Bandwidth utilization warnings

3. **Kernel Timing:**
   - Dispatch overhead measurement
   - Memory operation tracking
   - Per-kernel statistics

**Usage:**
```bash
GGML_HIP_LOG_PERFORMANCE=1 llama-bench -m model.gguf
```

**Output Includes:**
- HBM2 latency analysis per kernel
- Memory bottleneck warnings
- Cache utilization metrics
- Bandwidth utilization estimates

## Investigation Findings

### Memory Access Patterns

From benchmark logs:
- **Working set**: ~77.62 KB per iteration (fits well in L2 cache - 1.9% utilization)
- **Global memory**: 0.137-0.221 GB per operation
- **Access patterns**: Mixed (some coalesced, some scattered)
- **Cache behavior**: Most operations >50% of L2 cache, likely cache misses

### Bottleneck Analysis

**Potential Bottlenecks:**
1. **HBM2 Latency**: ~194ns per access, multiplied by cache miss rate
2. **Memory Bandwidth**: Current utilization appears low (0.0% reported, likely measurement issue)
3. **Cache Misses**: Large working sets (>4MB) cause frequent L2 misses
4. **Scattered Access**: Some operations have non-coalesced access patterns

### Prefetching Recommendations

**For HBM2 (~300-400 cycle latency):**
1. **Prefetch Distance**: Need to prefetch 2-3 iterations ahead to hide latency
   - Current: 1 iteration (nrows*nwarps)
   - Recommended: 2-3 iterations for better latency hiding

2. **Prefetch Timing**: Prefetch should happen early in iteration
   - Current: Prefetch at start of iteration
   - Consider: Prefetch at end of previous iteration

3. **Selective Prefetching**: Only prefetch when beneficial
   - Skip prefetching for small working sets (already in cache)
   - Focus on large operations (>4MB) that cause cache misses

4. **Prefetch Granularity**: Prefetch entire cache lines
   - HBM2 cache line: 64 bytes
   - Prefetch aligned to cache line boundaries

## Testing Prefetching

**Enable Experimental Prefetching:**
```bash
GGML_HIP_PREFETCH_ENABLE=1 GGML_HIP_LOG_PERFORMANCE=1 llama-bench -m model.gguf
```

**Monitor:**
- Performance metrics (pp512, tg128)
- HBM2 bottleneck analysis output
- Cache utilization metrics
- Memory latency warnings

**Expected Behavior:**
- If prefetching helps: Reduced memory latency warnings, improved throughput
- If prefetching hurts: Performance regression, increased cache pressure warnings

## Next Steps

1. **Profile with ROCm Tools**: Use `rocprof` for accurate kernel timing
   ```bash
   rocprof --stats ./llama-bench -m model.gguf
   ```

2. **Tune Prefetch Distance**: Experiment with 2-3 iteration prefetch distance

3. **Cache-Aware Prefetching**: Only prefetch when working set > L2 cache size

4. **Memory Access Pattern Optimization**: Improve coalescing for scattered accesses

5. **Consider Alternative Strategies**:
   - Software pipelining (double-buffering)
   - Explicit cache management
   - Memory access reordering

## References

- Vega 7nm Shader ISA Reference Guide
- HBM2 latency characteristics: ~300-400 cycles
- gfx906 L2 cache: 4MB
- HBM2 bandwidth: ~1TB/s theoretical peak

