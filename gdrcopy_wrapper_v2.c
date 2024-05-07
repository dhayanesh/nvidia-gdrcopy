#include <cuda_runtime_api.h>
#include <gdrapi.h>
#include <stdlib.h>

gdr_t init_gdr() {
    return gdr_open();
}

void close_gdr(gdr_t g) {
    gdr_close(g);
}

void* gpu_to_cpu_transfer(gdr_t g, void *gpu_ptr, void* host_ptr, size_t size) {
    gdr_mh_t mh;
    if (gdr_pin_buffer(g, (unsigned long)gpu_ptr, size, 0, 0, &mh) != 0) {
        return NULL;
    }

    void *mapped_ptr = NULL;

    gdr_map(g, mh, &mapped_ptr, size);
    if (!mapped_ptr) {
        gdr_unpin_buffer(g, mh);
        return NULL;
    }

    gdr_copy_from_mapping(mh, host_ptr, mapped_ptr, size);

    gdr_unmap(g, mh, mapped_ptr, size);

    gdr_unpin_buffer(g, mh);

    return host_ptr;
}