#include <cuda_runtime_api.h>
#include <gdrapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

gdr_t init_gdr() {
    return gdr_open();
}

void close_gdr(gdr_t g) {
    gdr_close(g);
}

void gpu_to_cpu_transfer(gdr_t g, void *gpu_ptr, void *host_ptr, size_t size) {
    gdr_mh_t mh;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    if (gdr_pin_buffer(g, (unsigned long)gpu_ptr, size, 0, 0, &mh) != 0) {
        fprintf(stderr, "Error in pinning buffer\n");
        return;
    }

    void *mapped_ptr = NULL;
    if (gdr_map(g, mh, &mapped_ptr, size) != 0) {
        fprintf(stderr, "Error in mapping buffer\n");
        gdr_unpin_buffer(g, mh);
        return;
    }

    gdr_copy_from_mapping(mh, host_ptr, mapped_ptr, size);

    gdr_unmap(g, mh, mapped_ptr, size);
    gdr_unpin_buffer(g, mh);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("All operations took %f milliseconds.\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    size_t size = 8 * 8 * sizeof(float);
    printf("Size: %zu bytes\n", size);

    float *h_a, *d_a;
    //h_a = (float *)malloc(size);
    if (posix_memalign((void **)&h_a, sysconf(_SC_PAGESIZE), size) != 0) {
        fprintf(stderr, "Error in posix_memalign\n");
        return 1;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMemset(d_a, 0, size);

    gdr_t g = init_gdr();
    
    gpu_to_cpu_transfer(g, d_a, h_a, size);

    // printf("Data transferred to CPU:\n");
    // for (int i = 0; i < 10; ++i) {
    //    printf("%f ", h_a[i]);
    // }
    // printf("\n");

    close_gdr(g);
    cudaFree(d_a);
    free(h_a);

    return 0;
}
