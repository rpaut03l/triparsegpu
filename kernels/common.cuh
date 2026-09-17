#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ bool is_delim(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\t';
}
