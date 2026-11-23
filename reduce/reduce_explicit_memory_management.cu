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

    // Pinned host memory for asynchronous transfers.
    // This allows us to use asynchronous memory copying
    // so that the two GPUs can run in parallel.
    unsigned int *xs;
    acc_t *partials;
    CHECK(cudaMallocHost(&xs, N * sizeof(unsigned int)));
    CHECK(cudaMallocHost(&partials, NUM_GPUS * sizeof(acc_t)));

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = static_cast<unsigned int>(i % 1000);
    }

    unsigned int *d_xs[NUM_GPUS] = {};
    acc_t *d_result[NUM_GPUS] = {};
    cudaStream_t streams[NUM_GPUS] = {};

    size_t starts[NUM_GPUS] = {};
    size_t sizes[NUM_GPUS] = {};
    int grid_sizes[NUM_GPUS] = {};

    // Set up streams and device buffers outside the timed region.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        starts[gpu] = N * gpu / NUM_GPUS;
        size_t end = N * (gpu + 1) / NUM_GPUS;
        sizes[gpu] = end - starts[gpu];

        size_t bytes = sizes[gpu] * sizeof(unsigned int);

        CHECK(cudaStreamCreate(&streams[gpu]));
        CHECK(cudaMalloc(&d_xs[gpu], bytes));
        CHECK(cudaMalloc(&d_result[gpu], sizeof(acc_t)));

        int sms = 0;
        CHECK(cudaDeviceGetAttribute(
            &sms, cudaDevAttrMultiProcessorCount, gpu));
        grid_sizes[gpu] = sms * BLOCKS_PER_SM;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Submit work to both GPUs.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        size_t bytes = sizes[gpu] * sizeof(unsigned int);

        CHECK(cudaMemcpyAsync(
            d_xs[gpu],
            xs + starts[gpu],
            bytes,
            cudaMemcpyHostToDevice,
            streams[gpu]));

        CHECK(cudaMemsetAsync(
            d_result[gpu], 0, sizeof(acc_t), streams[gpu]));

        reduce_kernel<BLOCK_SIZE><<<grid_sizes[gpu], BLOCK_SIZE, 0,
                                    streams[gpu]>>>(
            d_xs[gpu], d_result[gpu], sizes[gpu]);
        CHECK(cudaGetLastError());

        // Only a single value per GPU has to come back.
        CHECK(cudaMemcpyAsync(
            &partials[gpu],
            d_result[gpu],
            sizeof(acc_t),
            cudaMemcpyDeviceToHost,
            streams[gpu]));
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
        total += partials[gpu];
    }

    auto t1 = std::chrono::steady_clock::now();

    // Validate the result.
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
    printf("Two-GPU explicit-memory reduce succeeded! (sum = %llu)\n", total);
    printf("  GPU phase:      %.3f ms\n", ms(t1 - t0).count());

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaFree(d_xs[gpu]));
        CHECK(cudaFree(d_result[gpu]));
        CHECK(cudaStreamDestroy(streams[gpu]));
    }

    CHECK(cudaFreeHost(xs));
    CHECK(cudaFreeHost(partials));
}
