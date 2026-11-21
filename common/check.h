#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                           \
    do                                        \
    {                                         \
        cudaError_t err = (call);             \
        if (err != cudaSuccess)               \
        {                                     \
            fprintf(stderr, "%s:%d: %s\n",    \
                    __FILE__, __LINE__,       \
                    cudaGetErrorString(err)); \
            std::exit(1);                     \
        }                                     \
    } while (0)
