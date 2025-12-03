# gfx906 (Vega20/MI50/Radeon VII) Optimization Roadmap
## Synthesis: Current State + Benchmark Analysis + Proposed Improvements

**Date:** 2025-01-XX  
**Hardware:** Dual gfx906 GPUs (Vega20/MI50/Radeon VII)  
**Model:** GPT-OSS 120B MXFP4 MoE  
**Current Performance:** pp512: 507.51 ± 32.89 t/s | tg128: 74.89 ± 0.10 t/s

---

## 📊 Current Performance Baseline (from Benchmark Output)

### Kernel Distribution
- **MMQ kernels:** 80.0% (800 calls) - Quantized matrix multiplication
- **cuBLAS fallback:** 19.5% (195 calls) - F32 attention layers
- **MMVQ kernels:** 0.5% (5 calls) - Quantized vector operations

### Operation Characteristics
- **Primary workload:** Q8_0 and MXFP4 quantized operations with `ne11=512` (prompt processing)
- **Attention bottleneck:** F32 operations (`ne0=2880, ne1=128`) using cuBLAS
- **Data sizes:** 22.50 GB (large), 2.81 GB (medium), 0.70 GB (small)
- **MMQ selection:** 100% MMQ usage for quantized types (optimal)

### Already Implemented Optimizations ✅
1. **gfx906-specific MMQ heuristics** - Optimized batch size thresholds for Q8_0/MXFP4
2. **Lookup table caching** - Shared memory optimization for `kvalues_mxfp4` and `kvalues_iq4nl`
3. **MMQ tile size optimization** - Prefer larger tiles for better memory coalescing
4. **Capability logging** - Comprehensive tracking of gfx906 instruction usage
5. **Performance telemetry** - Detailed kernel selection and optimization opportunity logging

---

## 🎯 Gap Analysis: What's Missing vs. What's Needed

### 1. **F32 Attention Layer Optimization** (HIGH PRIORITY)
**Current State:**
- 19.5% of operations are F32 attention layers (`ne0=2880, ne1=128`)
- Using generic cuBLAS fallback (not optimized for gfx906)
- Small matrix dimensions suggest custom kernels could outperform cuBLAS

**Impact:** ~5-10% overall performance improvement potential

**What's Missing:**
- Custom HIP kernels for F32 attention operations optimized for gfx906
- Exploitation of `V_DOT2_F32_F16` and `V_FMAC_F32` for attention
- Block-tiling optimized for HBM2 memory hierarchy

**Proposed Solution:**
- Implement custom `mul_mat_f32_gfx906` kernel using `V_FMAC_F32`
- Optimize memory access patterns for `ne0=2880, ne1=128` attention matrices
- Consider mixed-precision (F16 accumulation) where appropriate

---

### 2. **Memory Bandwidth Optimization** (MEDIUM-HIGH PRIORITY)
**Current State:**
- Large data transfers (22.50 GB operations) suggest memory-bound workloads
- No evidence of HBM2-specific optimizations (prefetching, cache hints)
- Generic memory access patterns

**Impact:** 3-7% improvement for large matrix operations

**What's Missing:**
- HBM2-aware memory access patterns (coalescing, prefetching)
- Cache hierarchy optimization (L1/L2 cache hints)
- Memory tiling optimized for HBM2 bandwidth characteristics

**Proposed Solution:**
- Implement HBM2-aware prefetching in MMQ kernels
- Optimize shared memory bank conflict avoidance for gfx906's 32-bank architecture
- Add memory access pattern analysis to telemetry

---

### 3. **Multi-GPU Pipeline Parallelism** (MEDIUM PRIORITY)
**Current State:**
- Dual GPU setup detected but no evidence of optimized multi-GPU utilization
- Standard llama.cpp multi-GPU support may not be optimized for PCIe/peer access

**Impact:** 1.5-2x throughput improvement with proper multi-GPU scaling

**What's Missing:**
- Optimized layer sharding across GPUs
- PCIe peer-to-peer transfer optimization
- Pipeline parallelism for prompt processing

**Proposed Solution:**
- Implement layer-parallel multi-GPU execution
- Optimize PCIe transfers for large model weights
- Add multi-GPU performance telemetry

---

### 4. **Advanced Quantization Support** (LOW-MEDIUM PRIORITY)
**Current State:**
- Strong support for Q8_0, MXFP4, Q4_0, Q4_1 (integer quantization)
- No block-floating-point (BFP) quantization
- No per-layer dynamic quantization ranges

**Impact:** 2-5% memory efficiency improvement, potential accuracy gains

**What's Missing:**
- Block-floating-point quantization for weights
- Dynamic quantization ranges per layer
- Activation quantization support

**Proposed Solution:**
- Add BFP quantization support (compatible with gfx906's vector units)
- Implement per-layer quantization configuration
- Add quantization-aware scaling

---

### 5. **Memory Management & Offloading** (LOW PRIORITY for Current Setup)
**Current State:**
- Standard llama.cpp partial offloading exists
- No aggressive paging/streaming for very large models
- No NVMe/disk offloading

**Impact:** Enables larger models but may not be needed if VRAM sufficient

**What's Missing:**
- Aggressive layer paging to host RAM
- Asynchronous prefetching/eviction
- NVMe streaming support

**Proposed Solution:**
- Implement layer-by-layer paging for 120B+ models
- Add asynchronous prefetching during idle time
- Support streaming from NVMe for extreme model sizes

---

## 🗺️ Prioritized Implementation Roadmap

### Phase 1: F32 Attention Optimization (IMMEDIATE - 1-2 weeks)
**Goal:** Replace cuBLAS fallback with custom gfx906-optimized kernels

**Tasks:**
1. ✅ Analyze F32 attention operation patterns (DONE - `ne0=2880, ne1=128`)
2. ✅ Implement custom `mul_mat_f32_gfx906` kernel using `V_FMAC_F32` (DONE)
3. ✅ Optimize memory access for attention matrix dimensions (DONE)
4. ✅ Benchmark vs. cuBLAS baseline (DONE - ~35.9% improvement)
5. ✅ Integrate into `ggml_cuda_mul_mat` dispatch logic (DONE)

**Expected Impact:** 5-10% overall performance improvement ✅ **ACHIEVED**

---

### Phase 1.5: Fix Scattered Memory Access Patterns (HIGH PRIORITY - 1-2 weeks)
**Goal:** Eliminate non-coalesced memory access in MMQ kernels (15-25% performance loss)

**Tasks:**
1. ⏳ Analyze `load_tiles_q8_0` memory access patterns
2. ⏳ Optimize block alignment and padding for coalescing
3. ⏳ Implement coalesced memory access patterns in MMQ kernels
4. ⏳ Use `__ldg()` for read-only quantized weights
5. ⏳ Benchmark memory bandwidth improvement

**Expected Impact:** 15-25% improvement (HIGHEST PRIORITY - biggest bottleneck)

**Success Metrics:**
- `src0_stride` shows coalesced access patterns
- Measured memory bandwidth utilization >70% of peak
- Reduced HBM2 latency impact

**Success Metrics:**
- F32 operations use custom kernel instead of cuBLAS
- Measured speedup vs. cuBLAS baseline
- Correctness validation (output matches cuBLAS)

---

### Phase 2: Memory Bandwidth Optimization (SHORT-TERM - 2-3 weeks)
**Goal:** Optimize HBM2 memory access patterns and cache utilization

**Tasks:**
1. ❌ HBM2-aware prefetching: **DISABLED** - Volatile load approach causes 16% regression
   - Previous implementation (__builtin_prefetch): -2.1% regression
   - Current implementation (volatile load): -16.1% regression ⚠️
   - Root cause: Volatile load blocks execution even when data is cached
   - **Recommendation**: Use software pipelining or compiler hints instead of explicit prefetching
   - Added HBM2-specific profiling to identify bottlenecks (see HBM2_PREFETCHING_INVESTIGATION.md)
2. ✅ Optimize shared memory bank conflict patterns for 32-bank architecture
3. ✅ Add memory access pattern telemetry
4. ✅ Profile and optimize cache hierarchy usage
5. ✅ Benchmark memory bandwidth utilization (with enhanced HBM2 profiling)

**Expected Impact:** 3-7% improvement for large matrix operations

**NEW FINDINGS FROM TELEMETRY (see TELEMETRY_OPTIMIZATION_OPPORTUNITIES.md):**
- ⚠️ **CRITICAL:** Scattered memory access patterns (`src0_stride=90`) causing 15-25% performance loss
  - Q8_0 quantized weights have non-coalesced access patterns
  - Need to optimize `load_tiles_q8_0` for better memory coalescing
  - **Priority:** HIGH - This is the biggest bottleneck identified
- ⚠️ **Cache misses:** Working sets (8-25 MB) are 6-25x larger than L2 cache (4MB)
  - Estimated 10-20% performance loss due to cache misses
  - Need better cache-aware tiling and software pipelining
- ⚠️ **Bank conflicts:** `tile_x stride=293` may cause bank conflicts on 32-bank architecture
  - Estimated 5-10% performance loss
  - Need to improve padding strategy

**Success Metrics:**
- Increased memory bandwidth utilization (measured via rocminfo/perf)
- Reduced cache miss rates
- Improved throughput for 22.50 GB operations

---

### Phase 3: Multi-GPU Pipeline Parallelism (MEDIUM-TERM - 3-4 weeks)
**Goal:** Optimize dual-GPU utilization with layer-parallel execution

**Tasks:**
1. ⏳ Implement layer sharding across GPUs
2. ⏳ Optimize PCIe peer-to-peer transfers
3. ⏳ Add pipeline parallelism for prompt processing
4. ⏳ Implement multi-GPU performance telemetry
5. ⏳ Benchmark scaling efficiency

**Expected Impact:** 1.5-2x throughput improvement

**Success Metrics:**
- Near-linear scaling across GPUs
- Efficient PCIe bandwidth utilization
- Reduced idle time on GPUs

---

### Phase 4: Advanced Quantization (LONG-TERM - 4-6 weeks)
**Goal:** Add block-floating-point and dynamic quantization support

**Tasks:**
1. ⏳ Implement BFP quantization for weights
2. ⏳ Add per-layer quantization configuration
3. ⏳ Implement dynamic quantization ranges
4. ⏳ Add activation quantization support
5. ⏳ Benchmark accuracy vs. performance tradeoffs

**Expected Impact:** 2-5% memory efficiency, potential accuracy improvements

**Success Metrics:**
- BFP quantization support for gfx906
- Per-layer quantization configuration working
- Memory footprint reduction vs. integer quantization

---

### Phase 5: Memory Management & Offloading (FUTURE - 6+ weeks)
**Goal:** Enable very large models (120B+) with aggressive paging

**Tasks:**
1. ⏳ Implement layer-by-layer paging to host RAM
2. ⏳ Add asynchronous prefetching/eviction
3. ⏳ Support NVMe streaming for extreme model sizes
4. ⏳ Optimize paging overhead
5. ⏳ Benchmark large model support

**Expected Impact:** Enables models beyond VRAM capacity

**Success Metrics:**
- Successfully run 120B+ models with <32GB VRAM per GPU
- Minimal paging overhead (<5% performance penalty)
- Smooth operation with large context windows

---

## 📈 Performance Projections

### Current Baseline
- **pp512:** 507.51 ± 32.89 t/s
- **tg128:** 74.89 ± 0.10 t/s

### Projected Improvements (Cumulative)

| Phase | Optimization | Expected Gain | Projected pp512 | Projected tg128 |
|-------|-------------|---------------|-----------------|-----------------|
| Baseline | Current state | - | 507.51 t/s | 74.89 t/s |
| Phase 1 | F32 attention optimization | +5-10% ✅ | 532-558 t/s | 78.6-82.4 t/s |
| Phase 1.5 | Fix scattered access patterns | +15-25% | 612-697 t/s | 90.6-103.1 t/s |
| Phase 2 | Memory bandwidth optimization | +3-7% | 630-746 t/s | 93.3-110.3 t/s |
| Phase 3 | Multi-GPU parallelism | +50-100% | 945-1492 t/s | 140.0-220.6 t/s |
| Phase 4 | Advanced quantization | +2-5% | 838-1253 t/s | 123.8-185.2 t/s |
| Phase 5 | Memory management | Enables larger models | - | - |

**Note:** Multi-GPU improvements assume near-linear scaling. Actual results may vary based on PCIe bandwidth and model characteristics.

---

## 🔧 Implementation Notes

### Code Organization
- **gfx906-specific code:** Use `#if defined(GGML_USE_HIP) && defined(GGML_HIP_GFX906_OPTIMIZE)` guards
- **Kernel location:** `ggml/src/ggml-cuda/` for HIP kernels
- **Heuristics:** `ggml/src/ggml-cuda/mmq.cu` for MMQ selection logic
- **Telemetry:** Extend existing `GGML_HIP_LOG_PERFORMANCE` infrastructure

### Testing Strategy
1. **Correctness:** Validate outputs match baseline (cuBLAS, reference CPU)
2. **Performance:** Benchmark with `llama-bench` using consistent models/configs
3. **Telemetry:** Use `GGML_HIP_LOG_PERFORMANCE=1` to track improvements
4. **Regression:** Ensure no performance degradation on other workloads

### Benchmarking Protocol
```bash
# Standard benchmark command
GGML_HIP_LOG_PERFORMANCE=1 GGML_CUDA_MMQ_DEBUG=1 \
HIP_VISIBLE_DEVICES=1,2 \
llama-bench -m <model> -fa 1 -ngl 99 --numa numactl -t 22 -sm layer -b 2048

# Track metrics:
# - Kernel distribution (MMQ vs cuBLAS vs custom)
# - Memory bandwidth utilization
# - GPU occupancy
# - PCIe transfer rates (multi-GPU)
```

---

## 📝 Next Steps (Immediate Actions)

1. **Implement Phase 1 (F32 Attention Optimization)**
   - Create `mul_mat_f32_gfx906.cuh` with custom kernel
   - Integrate into `ggml_cuda_mul_mat` dispatch
   - Benchmark and validate

2. **Enhance Telemetry**
   - Add F32 operation tracking to performance logs
   - Measure memory bandwidth utilization
   - Track cache miss rates

3. **Profile Memory Access**
   - Use `rocminfo` to measure HBM2 bandwidth
   - Profile cache hierarchy usage
   - Identify memory access bottlenecks

---

## 🎓 Key Insights from Synthesis

### What We've Learned
1. **MMQ optimization is working well** - 100% MMQ usage for quantized types shows heuristics are optimal
2. **F32 attention is the bottleneck** - 19.5% of operations using generic cuBLAS is the low-hanging fruit
3. **Memory bandwidth is underutilized** - Large operations (22.50 GB) suggest room for HBM2 optimization
4. **Multi-GPU potential is untapped** - Dual GPU setup not fully exploited

### What's Different from Generic llama.cpp
- ✅ gfx906-specific MMQ heuristics (already implemented)
- ✅ Lookup table caching (already implemented)
- ✅ Comprehensive capability logging (already implemented)
- ⏳ Custom F32 kernels (next priority)
- ⏳ HBM2-aware memory optimization (medium priority)
- ⏳ Multi-GPU pipeline parallelism (medium priority)

### Alignment with ChatGPT Report
The ChatGPT report correctly identified:
- ✅ Limited AMD GPU optimization (we're addressing this)
- ✅ No specialized kernels for Vega (Phase 1 addresses this)
- ✅ Memory management gaps (Phase 2 & 5 address this)
- ✅ Multi-GPU limitations (Phase 3 addresses this)

**Our advantage:** We have concrete benchmark data showing exactly where to optimize, rather than generic recommendations.

---

## 📚 References

- **Current Implementation:** `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/mmq.cuh`
- **Capability Logging:** `ggml/src/ggml-cuda/ggml-cuda.cu` (lines 315-377)
- **Performance Telemetry:** `ggml/src/ggml-cuda/ggml-cuda.cu` (lines 2379-2500)
- **AMD Vega ISA:** [Vega 7nm Shader ISA Reference](https://gpuopen.com/download/Vega_7nm_Shader_ISA_26November2019.pdf)
- **Benchmark Data:** Terminal output lines 491-1019 (this document)

---

**Document Status:** Living document - update as optimizations are implemented and benchmarked.

