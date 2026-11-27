#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#ifndef BUFFERED
#define BUFFERED 0
#endif

void check(CUresult err)
{
    if (err != CUDA_SUCCESS)
    {
        const char *msg;
        cuGetErrorString(err, &msg);
        std::cerr << msg << std::endl;
        std::exit(1);
    }
}

int main()
{
    int n = 8;
    int k = 2;

    constexpr int num_gpus = 2;
    int count = n / num_gpus;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    CUdevice devices[num_gpus];
    CUcontect contexts[num_gpus];
    CUmodule modules[num_gpus];
    CUfunction kernels[num_gpus];

    CUdeviceptr buffers[num_gpus];

    check(cuInit(0));

    for (int i = 0; i < num_gpus; i++)
    {
        check(cuDeviceGet(&devices[g], g));

        check(cuDevicePrimaryCtxRetain(
            &contexts[g], devices[g]));

        check(cuCtxSetCurrent(contexts[g]));

        check(cuModuleLoad(
            &modules[g], "stencil.ptx"));

        check(cuModuleGetFunction(
            &kernels[g],
            modules[g],
            BUFFERED ? "stencil_buffered"
                     : "stencil_direct"));

#if BUFFERED
        check(cuMemAlloc(&buffers[g], count * sizeof(float)));
#endif
        check(cu)
    }

    check(cuCtxSetCurrent(contexts[0]));

    CUdeviceptr xs, ys;

    check(cuMemAllocManaged(
        &xs, n * sizeof(float), CU_MEM_ATTACH_GLOBAL));

    check(cuMemAllocManaged(
        &ys, n * sizeof(float), CU_MEM_ATTACH_GLOBAL));

    // Initialize input on the CPU.
    float *input = reinterpret_cast<float *>(xs);

    for (int i = 0; i < n; i++)
        input[i] = 3.0f * i;

    auto start = std::chrono::steady_clock::now();
    // Stencil iterations.
    for (int j = 0; j < k; j++)
    {

        for (int g = 0; g < num_gpus; g++)
        {
            check(cuCtxSetCurrent(contexts[g]));

            int offset = g * count;

#if BUFFERED
            // Copy the current slice into local memory.
            check(cuMemcpyAsync(
                buffers[g],
                xs + offset * sizeof(float),
                count * sizeof(float),
                nullptr));

            void *args[] = {
                &xs, &buffers[g], &ys,
                &n, &offset, &count};
#else
            void *args[] = {
                &xs, &ys, &n, &offset, &count};
#endif

            check(cuLaunchKernel(
                kernels[g],
                blocks, 1, 1,
                threads, 1, 1,
                0, nullptr,
                args, nullptr));
        }

        // Wait for both GPUs.
        for (int g = 0; g < num_gpus; g++)
        {
            check(cuCtxSetCurrent(contexts[g]));
            check(cuCtxSynchronize());
        }

        std::swap(xs, ys);
    }

    auto end = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(
                    end - start)
                    .count();

    std::cout << "Execution time: " << ms << " ms\n";

    // Print result.
    float *result = reinterpret_cast<float *>(xs);

    for (int i = 0; i < std::min(n, 16); i++)
        std::cout << result[i] << " ";

    std::cout << "\n";

    // Free Unified Memory.
    check(cuCtxSetCurrent(contexts[0]));
    check(cuMemFree(xs));
    check(cuMemFree(ys));

    // Cleanup.
    for (int g = 0; g < num_gpus; g++)
    {
        check(cuCtxSetCurrent(contexts[g]));

#if BUFFERED
        check(cuMemFree(buffers[g]));
#endif

        check(cuModuleUnload(modules[g]));
        check(cuCtxSetCurrent(nullptr));
        check(cuDevicePrimaryCtxRelease(devices[g]));
    }
}