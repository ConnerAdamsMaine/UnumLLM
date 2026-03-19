#include <hip/hip_runtime.h>
#include <stdint.h>

enum OneBitHipWeightFormat : uint32_t {
    ONEBIT_HIP_WEIGHT_TERNARY = 0,
    ONEBIT_HIP_WEIGHT_BINARY = 1,
};

enum OneBitHipScaleLayout : uint32_t {
    ONEBIT_HIP_SCALE_PER_TENSOR = 0,
    ONEBIT_HIP_SCALE_PER_ROW = 1,
    ONEBIT_HIP_SCALE_PER_GROUP = 2,
};

struct OneBitHipPackedMatrixLayout {
    uint32_t abi_version;
    uint32_t weight_format;
    uint32_t scale_layout;
    uint32_t bits_per_weight;
    uint64_t rows;
    uint64_t cols;
    uint64_t packed_word_count;
    const uint64_t* packed_words;
    uint64_t scale_count;
    const float* scales;
    uint64_t group_size;
    uint64_t equalizer_seed;
    uint32_t has_equalizer_seed;
};

static __device__ __forceinline__ uint64_t splitmix64_device(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

static __device__ __forceinline__ float decode_binary_weight(
    const OneBitHipPackedMatrixLayout& packed,
    uint64_t flat_index
) {
    const uint64_t word = packed.packed_words[flat_index >> 6];
    const uint64_t bit = (word >> (flat_index & 63ull)) & 1ull;
    const bool base_sign =
        packed.has_equalizer_seed != 0 &&
        ((splitmix64_device(packed.equalizer_seed ^ flat_index) & 1ull) != 0ull);
    return ((bit != 0ull) ^ base_sign) ? 1.0f : -1.0f;
}

static __device__ __forceinline__ float decode_ternary_weight(
    const OneBitHipPackedMatrixLayout& packed,
    uint64_t flat_index
) {
    const uint64_t word = packed.packed_words[flat_index >> 5];
    const uint8_t bits = (word >> ((flat_index & 31ull) * 2ull)) & 0x3u;
    if (bits == 1u) {
        return 1.0f;
    }
    if (bits == 2u) {
        return -1.0f;
    }
    return 0.0f;
}

static __device__ __forceinline__ float scale_for_index(
    const OneBitHipPackedMatrixLayout& packed,
    uint64_t row,
    uint64_t flat_index
) {
    switch (packed.scale_layout) {
        case ONEBIT_HIP_SCALE_PER_TENSOR:
            return packed.scales[0];
        case ONEBIT_HIP_SCALE_PER_ROW:
            return packed.scales[row];
        case ONEBIT_HIP_SCALE_PER_GROUP:
            return packed.scales[flat_index / packed.group_size];
        default:
            return 0.0f;
    }
}

__global__ void onebitllm_packed_dense_left_transposed_kernel(
    OneBitHipPackedMatrixLayout packed,
    const float* lhs,
    uint64_t lhs_rows,
    float* out
) {
    const uint64_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lhs_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (lhs_row >= lhs_rows || out_col >= packed.rows) {
        return;
    }

    float acc = 0.0f;
    for (uint64_t col = 0; col < packed.cols; ++col) {
        const uint64_t flat_index = out_col * packed.cols + col;
        const float scale = scale_for_index(packed, out_col, flat_index);
        const float weight = packed.weight_format == ONEBIT_HIP_WEIGHT_BINARY
            ? decode_binary_weight(packed, flat_index)
            : decode_ternary_weight(packed, flat_index);
        acc += lhs[lhs_row * packed.cols + col] * weight * scale;
    }

    out[lhs_row * packed.rows + out_col] = acc;
}

extern "C" int onebitllm_hip_packed_matmul_dense_left_transposed_v1(
    const OneBitHipPackedMatrixLayout* packed,
    const float* lhs,
    uint64_t lhs_rows,
    uint64_t lhs_cols,
    float* out
) {
    if (packed == nullptr || lhs == nullptr || out == nullptr) {
        return 1;
    }
    if (packed->abi_version != 1u) {
        return 2;
    }
    if (packed->rows == 0 || packed->cols == 0 || lhs_rows == 0) {
        return 3;
    }
    if (lhs_cols != packed->cols) {
        return 4;
    }
    if (packed->scale_count == 0 || packed->scales == nullptr || packed->packed_words == nullptr) {
        return 5;
    }
    if (packed->weight_format == ONEBIT_HIP_WEIGHT_BINARY && packed->bits_per_weight != 1u) {
        return 6;
    }
    if (packed->weight_format == ONEBIT_HIP_WEIGHT_TERNARY && packed->bits_per_weight != 2u) {
        return 7;
    }
    if (packed->scale_layout == ONEBIT_HIP_SCALE_PER_ROW && packed->scale_count < packed->rows) {
        return 8;
    }
    if (packed->scale_layout == ONEBIT_HIP_SCALE_PER_GROUP && packed->group_size == 0) {
        return 9;
    }

    const size_t packed_bytes = static_cast<size_t>(packed->packed_word_count * sizeof(uint64_t));
    const size_t scales_bytes = static_cast<size_t>(packed->scale_count * sizeof(float));
    const size_t lhs_bytes = static_cast<size_t>(lhs_rows * lhs_cols * sizeof(float));
    const size_t out_bytes = static_cast<size_t>(lhs_rows * packed->rows * sizeof(float));

    uint64_t* d_packed_words = nullptr;
    float* d_scales = nullptr;
    float* d_lhs = nullptr;
    float* d_out = nullptr;

    hipError_t status = hipSuccess;
    status = hipMalloc(reinterpret_cast<void**>(&d_packed_words), packed_bytes);
    if (status != hipSuccess) {
        return 10;
    }
    status = hipMalloc(reinterpret_cast<void**>(&d_scales), scales_bytes);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        return 11;
    }
    status = hipMalloc(reinterpret_cast<void**>(&d_lhs), lhs_bytes);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        return 12;
    }
    status = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        return 13;
    }

    status = hipMemcpy(d_packed_words, packed->packed_words, packed_bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        hipFree(d_out);
        return 14;
    }
    status = hipMemcpy(d_scales, packed->scales, scales_bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        hipFree(d_out);
        return 15;
    }
    status = hipMemcpy(d_lhs, lhs, lhs_bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        hipFree(d_out);
        return 16;
    }

    OneBitHipPackedMatrixLayout device_layout = *packed;
    device_layout.packed_words = d_packed_words;
    device_layout.scales = d_scales;

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((packed->rows + block.x - 1) / block.x),
        static_cast<unsigned int>((lhs_rows + block.y - 1) / block.y)
    );
    hipLaunchKernelGGL(
        onebitllm_packed_dense_left_transposed_kernel,
        grid,
        block,
        0,
        0,
        device_layout,
        d_lhs,
        lhs_rows,
        d_out
    );

    status = hipGetLastError();
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        hipFree(d_out);
        return 17;
    }
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        hipFree(d_packed_words);
        hipFree(d_scales);
        hipFree(d_lhs);
        hipFree(d_out);
        return 18;
    }

    status = hipMemcpy(out, d_out, out_bytes, hipMemcpyDeviceToHost);
    hipFree(d_packed_words);
    hipFree(d_scales);
    hipFree(d_lhs);
    hipFree(d_out);
    if (status != hipSuccess) {
        return 19;
    }
    return 0;
}
