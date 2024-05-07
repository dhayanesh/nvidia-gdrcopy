#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

void init_cuda() {
    cudaSetDevice(0);
}

void close_cuda() {
    cudaDeviceReset();
}

void* gpu_to_cpu_memcpy(void *gpu_ptr, void *host_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return host_ptr;
}
