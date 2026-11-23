#include "../common/check.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Integer sum so the result is exact and independent of summation order.
using acc_t = unsigned long long;

// Computes map (reduce (+) 0) over the rows of a row-major matrix.
// The map is over blocks (one block per row), the reduce is within a block.
template <int BLOCK_SIZE>
__global__ void map_reduce_kernel(
    const unsigned int *xss, acc_t *ys, size_t cols)
{
    __shared__ acc_t partial[BLOCK_SIZE];

    size_t row = blockIdx.x;
    const unsigned int *xs = xss + row * cols;

    // Block-stride loop: each thread sums many elements of the row.
    acc_t sum = 0;
    for (size_t j = threadIdx.x; j < cols; j += BLOCK_SIZE)
    {
        sum += xs[j];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        ys[row] = partial[0];
    }
}

int main()
{
    constexpr size_t ROWS = 1 << 14;
    constexpr size_t COLS = 1 << 12;
    constexpr size_t N = ROWS * COLS;
    constexpr int NUM_GPUS = 2;
    constexpr int BLOCK_SIZE = 256;

    int count;
    CHECK(cudaGetDeviceCount(&count));

    if (count < NUM_GPUS)
    {
        fprintf(stderr, "Need at least 2 GPUs\n");
        return 1;
    }

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        int concurrent = 0;
        CHECK(cudaDeviceGetAttribute(
            &concurrent, cudaDevAttrConcurrentManagedAccess, gpu));
        if (!concurrent)
        {
            fprintf(stderr, "GPU %d does not support managed-memory "
                            "prefetching\n",
                    gpu);
            return 1;
        }
    }

    unsigned int *xss;
    acc_t *ys;
    CHECK(cudaMallocManaged(&xss, N * sizeof(unsigned int)));
    CHECK(cudaMallocManaged(&ys, ROWS * sizeof(acc_t)));

    for (size_t i = 0; i < N; i++)
    {
        xss[i] = static_cast<unsigned int>(i % 1000);
    }

    cudaStream_t streams[NUM_GPUS] = {};
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamCreate(&streams[gpu]));
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        // Split the rows (the map dimension) between the GPUs.
        size_t row_start = ROWS * gpu / NUM_GPUS;
        size_t row_end = ROWS * (gpu + 1) / NUM_GPUS;
        size_t rows = row_end - row_start;
        size_t in_bytes = rows * COLS * sizeof(unsigned int);
        size_t out_bytes = rows * sizeof(acc_t);

        // Bulk-migrate this slice instead of faulting it in page by page.
        CHECK(cudaMemPrefetchAsync(
            xss + row_start * COLS, in_bytes, gpu, streams[gpu]));
        CHECK(cudaMemPrefetchAsync(
            ys + row_start, out_bytes, gpu, streams[gpu]));

        map_reduce_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE, 0,
                                        streams[gpu]>>>(
            xss + row_start * COLS, ys + row_start, COLS);
        CHECK(cudaGetLastError());

        // Send results back before the CPU reads them.
        CHECK(cudaMemPrefetchAsync(
            ys + row_start, out_bytes, cudaCpuDeviceId, streams[gpu]));
    }

    // Wait for both GPUs to finish.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamSynchronize(streams[gpu]));
    }

    auto t1 = std::chrono::steady_clock::now();

    // Validate the result. This touches xss on the CPU, migrating it back.
    for (size_t row = 0; row < ROWS; row++)
    {
        acc_t expected = 0;
        for (size_t j = 0; j < COLS; j++)
        {
            expected += xss[row * COLS + j];
        }

        if (ys[row] != expected)
        {
            fprintf(stderr, "Mismatch at row %zu: got %llu, expected %llu\n",
                    row, ys[row], expected);
            return 1;
        }
    }

    using ms = std::chrono::duration<double, std::milli>;
    printf("Two-GPU unified-memory map-reduce succeeded!\n");
    printf("  GPU phase:      %.3f ms\n", ms(t1 - t0).count());

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamDestroy(streams[gpu]));
    }

    CHECK(cudaFree(xss));
    CHECK(cudaFree(ys));
}
