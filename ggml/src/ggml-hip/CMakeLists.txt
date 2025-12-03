if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

# Workaround old compilers
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --gpu-max-threads-per-block=1024")

# Detect if gfx906 is in the target architectures
set(HAS_GFX906 FALSE)
if(CMAKE_HIP_ARCHITECTURES)
    foreach(arch ${CMAKE_HIP_ARCHITECTURES})
        if(arch MATCHES "gfx906")
            set(HAS_GFX906 TRUE)
            break()
        endif()
    endforeach()
elseif(GPU_TARGETS)
    foreach(arch ${GPU_TARGETS})
        if(arch MATCHES "gfx906")
            set(HAS_GFX906 TRUE)
            break()
        endif()
    endforeach()
endif()

# Option to enable gfx906-specific optimizations
option(GGML_HIP_GFX906_OPTIMIZE "Enable gfx906 (Vega20/MI50/Radeon VII) specific optimizations" ${HAS_GFX906})

if(GGML_HIP_GFX906_OPTIMIZE)
    message(STATUS "gfx906-specific optimizations enabled")
    
    # gfx906-specific optimization flags for HIP
    # -O3: Maximum optimization
    # -ffast-math: Fast math operations (safe for LLM inference, preserves NaN/Inf handling)
    # -mllvm -amdgpu-early-inline-all=true: Enable early inlining for better register usage
    # -mllvm -amdgpu-function-calls=false: Disable function calls in kernels for better performance
    # -mllvm -amdgpu-waves-per-eu=4: Optimize for wavefront scheduling (gfx906 has 4 CUs per SE)
    
    set(GFX906_HIP_FLAGS "-O3 -ffast-math")
    
    # Additional LLVM optimizations for gfx906
    # These can be tuned based on profiling results
    option(GGML_HIP_GFX906_AGGRESSIVE_OPT "Enable aggressive gfx906 optimizations (may increase compile time)" OFF)
    if(GGML_HIP_GFX906_AGGRESSIVE_OPT)
        set(GFX906_HIP_FLAGS "${GFX906_HIP_FLAGS} -mllvm -amdgpu-early-inline-all=true")
        message(STATUS "Aggressive gfx906 optimizations enabled")
    endif()
    
    # Add gfx906-specific flags to CMAKE_HIP_FLAGS
    # Note: These flags apply to all HIP compilations when gfx906 is targeted
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${GFX906_HIP_FLAGS}")
    
    # Add compile definition to enable gfx906-specific code paths
    add_compile_definitions(GGML_HIP_GFX906_OPTIMIZE)
endif()

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
