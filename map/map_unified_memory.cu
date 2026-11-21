#include "../common/check.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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

    float *xs, *ys;
    CHECK(cudaMallocManaged(&xs, N * sizeof(float)));
    CHECK(cudaMallocManaged(&ys, N * sizeof(float)));

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = static_cast<float>(i);
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

        size_t start = N * gpu / NUM_GPUS;
        size_t end = N * (gpu + 1) / NUM_GPUS;
        size_t size = end - start;
        size_t bytes = size * sizeof(float);

        // Bulk-migrate this slice instead of faulting it in page by page.
        CHECK(cudaMemPrefetchAsync(xs + start, bytes, gpu, streams[gpu]));
        CHECK(cudaMemPrefetchAsync(ys + start, bytes, gpu, streams[gpu]));

        int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        map_kernel<<<blocks, BLOCK_SIZE, 0,
                     streams[gpu]>>>(
            xs + start, ys + start, size);
        CHECK(cudaGetLastError());

        // Send results back before the CPU reads them.
        CHECK(cudaMemPrefetchAsync(
            ys + start, bytes, cudaCpuDeviceId, streams[gpu]));
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
    printf("Two-GPU unified-memory map succeeded!\n");
    printf("  GPU phase:      %.3f ms\n", ms(t1 - t0).count());
    printf("  CPU validation: %.3f ms\n", ms(t2 - t1).count());

    for (int gpu = 0; gpu < NUM_GPUS; gpu++)
    {
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamDestroy(streams[gpu]));
    }

    CHECK(cudaFree(xs));
    CHECK(cudaFree(ys));
}
