#pragma once

#include "common.cuh"
#include "ggml.h"

// Custom F32 matrix multiplication kernel optimized for gfx906 (Vega20/MI50/Radeon VII)
// Specifically optimized for attention operations: ne0=2880, ne1=128, ne11=512
// Uses V_FMAC_F32 instruction for efficient FMA operations

// Optimized kernel for attention operations: M=2880, N=128, K=2880
// Tile sizes chosen for gfx906: TILE_M=64, TILE_N=32, TILE_K=64
// This balances shared memory usage (~16KB) with occupancy
__launch_bounds__(256, 1) // 4 warps (64 threads each) for gfx906
static __global__ void mul_mat_f32_gfx906_attention(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M,      // 2880 (ne0)
    const int N,      // 128 (ne1)
    const int K,      // 2880 (ne01)
    const int64_t stride_A,
    const int64_t stride_B,
    const int64_t stride_C) {
    
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 64;
    constexpr int warp_size = 64;
    constexpr int nwarps = 4;
    
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int block_row = blockIdx.x * TILE_M;
    const int block_col = blockIdx.y * TILE_N;
    
    // Shared memory tiles (padded to avoid bank conflicts on 32-bank architecture)
    __shared__ float tile_A[TILE_M][TILE_K + 1];
    __shared__ float tile_B[TILE_K][TILE_N + 1];
    
    // Accumulator: each warp handles TILE_M/nwarps = 16 rows
    float acc[16] = {0.0f};
    
    const int warp_row_start = warp_id * 16; // TILE_M / nwarps = 64/4 = 16
    
    // Main computation loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative loading of tile_A (transposed): A[k][row] -> tile_A[k][row]
        // We need to load A transposed, so we access A[k_global * stride_A + row]
        for (int load_idx = threadIdx.x; load_idx < TILE_K * TILE_M; load_idx += blockDim.x) {
            const int k = load_idx / TILE_M;
            const int i = load_idx % TILE_M;
            const int k_global = k_tile + k;
            const int row = block_row + i;
            if (k_global < K && row < M) {
                // Load A transposed: A[k][row] = A[k_global * stride_A + row]
                tile_A[k][i] = A[k_global * stride_A + row];
            } else {
                tile_A[k][i] = 0.0f;
            }
        }
        
        // Cooperative loading of tile_B
        for (int load_idx = threadIdx.x; load_idx < TILE_K * TILE_N; load_idx += blockDim.x) {
            const int k = load_idx / TILE_N;
            const int j = load_idx % TILE_N;
            const int k_global = k_tile + k;
            const int col = block_col + j;
            if (k_global < K && col < N) {
                tile_B[k][j] = B[k_global * stride_B + col];
            } else {
                tile_B[k][j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using V_FMAC_F32: C = A^T * B
        // Each warp computes 16 rows of C, each thread computes 1 column
        // For A^T: we access A[k][row] instead of A[row][k]
        for (int k = 0; k < TILE_K && (k_tile + k) < K; ++k) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int row = warp_row_start + i;
                // Access A transposed: A[k][row] instead of A[row][k]
                const float a_val = tile_A[k][row]; // Note: swapped indices for transpose
                // Each lane handles one column of B
                const float b_val = (lane_id < TILE_N) ? tile_B[k][lane_id] : 0.0f;
                ggml_cuda_mad(acc[i], a_val, b_val);
            }
        }
        
        __syncthreads();
    }
    
    // Write results: each warp writes 16 rows, each thread writes 1 column
    for (int i = 0; i < 16; ++i) {
        const int row = block_row + warp_row_start + i;
        if (row < M && lane_id < TILE_N) {
            const int col = block_col + lane_id;
            if (col < N) {
                C[row * stride_C + col] = acc[i];
            }
        }
    }
}

// Host function to launch the kernel
// Computes C = A^T * B where:
//   A is M x K (src0, stored row-major, accessed transposed)
//   B is K x N (src1, stored row-major)
//   C is M x N (dst, stored row-major)
static void mul_mat_f32_gfx906(
    cudaStream_t stream,
    const float * A,      // src0, M x K matrix (ne00 x ne01)
    const float * B,      // src1, K x N matrix (ne10 x ne11)
    float * C,            // dst, M x N matrix (ne0 x ne1)
    const int M,          // ne00 (rows of A, rows of C)
    const int N,          // ne11 (cols of B, cols of C)
    const int K,          // ne01 = ne10 (cols of A, rows of B)
    const int64_t stride_A, // stride for A (nb01/sizeof(float))
    const int64_t stride_B, // stride for B (s11)
    const int64_t stride_C) { // stride for C (ne0)
    
    // For attention operations (M=2880, K=128, N<=2048), use specialized kernel
    // Supports batch sizes: 128 (token gen), 512 (prompt), 1024, 2048 (large prompts)
    // Kernel uses dynamic grid dimensions, so it scales to any N <= 2048
    if (M == 2880 && K == 128 && N <= 2048) {
        constexpr int TILE_M = 64;
        constexpr int TILE_N = 32;
        dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
        dim3 block(256); // 4 warps
        
        mul_mat_f32_gfx906_attention<<<grid, block, 0, stream>>>(
            A, B, C, M, N, K, stride_A, stride_B, stride_C);
    }
    // For now, only support the attention case - can extend later
}

