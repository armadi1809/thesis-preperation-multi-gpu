#include "../common/check.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Integer sum so the result is exact and independent of summation order.
using acc_t = unsigned long long;

template <int BLOCK_SIZE>
__global__ void reduce_kernel(
    const unsigned int *xs, acc_t *result, size_t n)
{
    __shared__ acc_t partial[BLOCK_SIZE];

    // Grid-stride loop: each thread sums many elements.
    acc_t sum = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x)
    {
        sum += xs[i];
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

    // One atomic per block into this GPU's result.
    if (threadIdx.x == 0)
    {
        atomicAdd(result, partial[0]);
    }
}

int main()
{
    constexpr size_t N = 1 << 26;
    constexpr int NUM_GPUS = 2;
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCKS_PER_SM = 4;

    int count;
    CHECK(cudaGetDeviceCount(&count));
    printf("Found %d GPU(s)\n", count);

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

    unsigned int *xs;
    CHECK(cudaMallocManaged(&xs, N * sizeof(unsigned int)));

    // Separate allocations per GPU so the two results never share a page,
    // which would otherwise bounce between the GPUs.
    acc_t *results[NUM_GPUS] = {};
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaMallocManaged(&results[gpu], sizeof(acc_t)));
    }

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = static_cast<unsigned int>(i % 1000);
    }

    cudaStream_t streams[NUM_GPUS] = {};
    int grid_sizes[NUM_GPUS] = {};
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamCreate(&streams[gpu]));

        int sms = 0;
        CHECK(cudaDeviceGetAttribute(
            &sms, cudaDevAttrMultiProcessorCount, gpu));
        grid_sizes[gpu] = sms * BLOCKS_PER_SM;
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        size_t start = N * gpu / NUM_GPUS;
        size_t end = N * (gpu + 1) / NUM_GPUS;
        size_t size = end - start;
        size_t bytes = size * sizeof(unsigned int);

        // Bulk-migrate this slice instead of faulting it in page by page.
        CHECK(cudaMemPrefetchAsync(xs + start, bytes, gpu, streams[gpu]));
        CHECK(cudaMemPrefetchAsync(
            results[gpu], sizeof(acc_t), gpu, streams[gpu]));

        CHECK(cudaMemsetAsync(
            results[gpu], 0, sizeof(acc_t), streams[gpu]));

        reduce_kernel<BLOCK_SIZE><<<grid_sizes[gpu], BLOCK_SIZE, 0,
                                    streams[gpu]>>>(
            xs + start, results[gpu], size);
        CHECK(cudaGetLastError());

        // Send the partial result back before the CPU reads it.
        CHECK(cudaMemPrefetchAsync(
            results[gpu], sizeof(acc_t), cudaCpuDeviceId, streams[gpu]));
    }

    // Wait for both GPUs to finish.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamSynchronize(streams[gpu]));
    }

    // Combine the per-GPU partial results on the host.
    acc_t total = 0;
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        total += *results[gpu];
    }

    auto t1 = std::chrono::steady_clock::now();

    // Validate the result. This touches xs on the CPU, migrating it back.
    acc_t expected = 0;
    for (size_t i = 0; i < N; i++)
    {
        expected += xs[i];
    }

    if (total != expected)
    {
        fprintf(stderr, "Mismatch: got %llu, expected %llu\n",
                total, expected);
        return 1;
    }

    using ms = std::chrono::duration<double, std::milli>;
    printf("Two-GPU unified-memory reduce succeeded! (sum = %llu)\n", total);
    printf("  GPU phase:      %.3f ms\n", ms(t1 - t0).count());

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamDestroy(streams[gpu]));
        CHECK(cudaFree(results[gpu]));
    }

    CHECK(cudaFree(xs));
}
