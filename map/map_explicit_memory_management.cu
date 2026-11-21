
#include "../common/check.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void map_kernel(
    const float *xs, float *ys, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        ys[i] = 2.0f * xs[i];
    }
}

int main()
{
    constexpr size_t N = 1 << 26;
    constexpr int NUM_GPUS = 2;
    constexpr int BLOCK_SIZE = 256;

    int count;
    CHECK(cudaGetDeviceCount(&count));

    if (count < NUM_GPUS)
    {
        fprintf(stderr, "Need at least 2 GPUs\n");
        return 1;
    }

    // Pinned host memory for asynchronous transfers.
    // This allows us to use asynchronous memory copying
    // so that the two GPUs can run in parallel.
    float *xs, *ys;
    CHECK(cudaMallocHost(&xs, N * sizeof(float)));
    CHECK(cudaMallocHost(&ys, N * sizeof(float)));

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = static_cast<float>(i);
    }

    float *d_xs[NUM_GPUS] = {};
    float *d_ys[NUM_GPUS] = {};
    cudaStream_t streams[NUM_GPUS] = {};

    size_t starts[NUM_GPUS] = {};
    size_t sizes[NUM_GPUS] = {};

    // Set up streams and device buffers outside the timed region.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        starts[gpu] = N * gpu / NUM_GPUS;
        size_t end = N * (gpu + 1) / NUM_GPUS;
        sizes[gpu] = end - starts[gpu];

        size_t bytes = sizes[gpu] * sizeof(float);

        CHECK(cudaStreamCreate(&streams[gpu]));
        CHECK(cudaMalloc(&d_xs[gpu], bytes));
        CHECK(cudaMalloc(&d_ys[gpu], bytes));
    }

    auto t0 = std::chrono::steady_clock::now();

    // Submit work to both GPUs.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));

        size_t bytes = sizes[gpu] * sizeof(float);

        CHECK(cudaMemcpyAsync(
            d_xs[gpu],
            xs + starts[gpu],
            bytes,
            cudaMemcpyHostToDevice,
            streams[gpu]));

        int blocks = (sizes[gpu] + BLOCK_SIZE - 1) / BLOCK_SIZE;

        map_kernel<<<blocks, BLOCK_SIZE, 0,
                     streams[gpu]>>>(
            d_xs[gpu], d_ys[gpu], sizes[gpu]);
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpyAsync(
            ys + starts[gpu],
            d_ys[gpu],
            bytes,
            cudaMemcpyDeviceToHost,
            streams[gpu]));
    }

    // Wait for both GPUs to finish.
    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamSynchronize(streams[gpu]));
    }

    auto t1 = std::chrono::steady_clock::now();

    // Validate the result.
    for (size_t i = 0; i < N; i++)
    {
        if (ys[i] != 2.0f * xs[i])
        {
            fprintf(stderr, "Mismatch at %zu\n", i);
            return 1;
        }
    }

    auto t2 = std::chrono::steady_clock::now();

    using ms = std::chrono::duration<double, std::milli>;
    printf("Two-GPU explicit-memory map succeeded!\n");
    printf("  GPU phase:      %.3f ms\n", ms(t1 - t0).count());
    printf("  CPU validation: %.3f ms\n", ms(t2 - t1).count());

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaFree(d_xs[gpu]));
        CHECK(cudaFree(d_ys[gpu]));
        CHECK(cudaStreamDestroy(streams[gpu]));
    }

    CHECK(cudaFreeHost(xs));
    CHECK(cudaFreeHost(ys));
}
