# llama.cpp - GPU Optimization Fork

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**This is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) focused on GPU architecture-specific optimizations.**

For general documentation, usage instructions, installation guides, model support, and all other llama.cpp features, please refer to the **[main llama.cpp repository](https://github.com/ggml-org/llama.cpp)** and its **[documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs)**.

---

## Optimization Goals

This fork focuses on maximizing inference performance for specific GPU architectures through:

- **Custom kernel implementations** optimized for target architectures
- **Memory bandwidth optimizations** (HBM2 prefetching, bank conflict reduction)
- **Multi-GPU pipeline parallelism** for scaling across devices
- **Advanced quantization optimizations** for specific architectures

---

## Current Progress: gfx906 (Vega20/MI50/Radeon VII)

### Performance Improvements

**Benchmark Results (gpt-oss 120B MXFP4 MoE, 2x gfx906 GPUs):**

| Version | pp512 (tokens/s) | tg128 (tokens/s) | Improvement |
|---------|------------------|-------------------|-------------|
| **Original** | 488.06 ± 2.15 | 72.46 ± 0.20 | Baseline |
| **Optimized** | **663.18 ± 15.28** | **75.23 ± 0.20** | **+35.9%** |

### Completed Optimizations

✅ **Phase 1: F32 Attention Kernel Optimization**
- Custom `mul_mat_f32_gfx906` kernel using `V_FMAC_F32` instruction
- Optimized for attention operations (M=2880, K=128, N≤2048)
- **Result**: ~43% improvement on F32 attention operations

✅ **Phase 2.2: Shared Memory Bank Conflict Optimization**
- Optimized shared memory access patterns for 32-bank architecture
- Added proper padding and alignment to avoid bank conflicts
- Documented bank conflict avoidance strategies

### In Progress / Planned

🔄 **Phase 2.3**: Memory access pattern telemetry  
📋 **Phase 2.4**: Cache hierarchy optimization  
📋 **Phase 3**: Multi-GPU pipeline parallelism  
📋 **Phase 4**: Advanced quantization optimizations  

### Next Target Architecture

- **T4 (Turing)**: Planned for next optimization cycle

---

## Technical Details

See [GFX906_OPTIMIZATION_ROADMAP.md](GFX906_OPTIMIZATION_ROADMAP.md) for detailed technical information about the optimization roadmap, implementation details, and performance analysis.

---

## Quick Start

This fork uses the same build system and commands as the main llama.cpp repository. See the [main repository's build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for installation instructions.

**Example usage:**
```sh
# Build with HIP support for AMD GPUs
cmake -DGGML_HIPBLAS=ON ..

# Run with performance logging
GGML_HIP_LOG_PERFORMANCE=1 llama-bench -m model.gguf
```

---

## Contributing

Contributions focused on GPU optimizations are welcome! Please ensure your changes:
- Include performance benchmarks showing improvement
- Are well-documented with technical rationale
- Follow the existing code style and patterns

For general llama.cpp contributions, please contribute to the [main repository](https://github.com/ggml-org/llama.cpp).

