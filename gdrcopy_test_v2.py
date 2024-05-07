import torch
import numpy as np
import ctypes
import os
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(current_dir, 'libgdrcopy_wrapper_v2.so')
lib = ctypes.CDLL(lib_path)

init_gdr = lib.init_gdr
init_gdr.restype = ctypes.c_void_p

close_gdr = lib.close_gdr
close_gdr.argtypes = [ctypes.c_void_p]

gpu_to_cpu_transfer = lib.gpu_to_cpu_transfer
gpu_to_cpu_transfer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
gpu_to_cpu_transfer.restype = ctypes.c_void_p

def transfer_gpu_to_cpu(gdr_handle, tensor_gpu, tensor_cpu, size):
    tensor_gpu = tensor_gpu.contiguous()
    if not gpu_to_cpu_transfer(gdr_handle, ctypes.c_void_p(tensor_gpu.data_ptr()), ctypes.c_void_p(tensor_cpu.data_ptr()), size):
        raise RuntimeError("GPU to CPU transfer failed")

def main():
    gdr_handle = init_gdr()

    tensor_gpu = torch.rand((1024 * 4, 1024 * 4), device='cuda', dtype=torch.float64)
    tensor_cpu = torch.empty(tensor_gpu.shape, dtype=torch.float64)
    
    start_time = time.time()

    size = tensor_gpu.numel() * tensor_gpu.element_size()
    transfer_gpu_to_cpu(gdr_handle, tensor_gpu, tensor_cpu, size)

    end_time = time.time()

    print("Tensor size in kb: ", size / 1024)
    print(f"Total time gdrCopy: {end_time - start_time:.10f} seconds")
    try:
        #print("Copied from GPU:", tensor_cpu)
        print("Device: ", tensor_cpu.device)
    except Exception as e:
        print("Error:", e)

    close_gdr(gdr_handle)

if __name__ == "__main__":
    main()
