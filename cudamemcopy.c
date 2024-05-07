#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void gpu_to_cpu_transfer(void *gpu_ptr, void *host_ptr, size_t size) {
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMemcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Transfer took %f milliseconds.\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    size_t size = 8 * 8 * sizeof(float);
    printf("Size: %zu bytes\n", size);

    float *h_a, *d_a;
    h_a = (float *)malloc(size);
    cudaMalloc((void **)&d_a, size);

    cudaMemset(d_a, 0, size);

    gpu_to_cpu_transfer(d_a, h_a, size);
    gpu_to_cpu_transfer(d_a, h_a, size);
    gpu_to_cpu_transfer(d_a, h_a, size);
    gpu_to_cpu_transfer(d_a, h_a, size);
    gpu_to_cpu_transfer(d_a, h_a, size);

    cudaFree(d_a);
    free(h_a);

    return 0;
}
