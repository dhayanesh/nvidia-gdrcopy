import torch
import ctypes
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'cuda_wrapper.so')
lib = ctypes.CDLL(lib_path)

init_cuda = lib.init_cuda
close_cuda = lib.close_cuda

gpu_to_cpu_memcpy = lib.gpu_to_cpu_memcpy
gpu_to_cpu_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
gpu_to_cpu_memcpy.restype = ctypes.c_void_p

def transfer_gpu_to_cpu(tensor_gpu, tensor_cpu):
    size = tensor_gpu.numel() * tensor_gpu.element_size()
    result = gpu_to_cpu_memcpy(ctypes.c_void_p(tensor_gpu.data_ptr()), ctypes.c_void_p(tensor_cpu.data_ptr()), size)
    if result is None:
        raise RuntimeError("GPU to CPU transfer failed")

def main():
    init_cuda()

    tensor_gpu = torch.rand((1024, 1024), device='cuda', dtype=torch.float64)

    start_time = time.time()
    tensor_cpu = torch.empty_like(tensor_gpu, device='cpu')
    transfer_gpu_to_cpu(tensor_gpu, tensor_cpu)
    end_time = time.time()

    print("Tensor size in kb: ", tensor_gpu.nelement() * tensor_gpu.element_size() / 1024)
    print(f"Total time cudaMemcpy: {end_time - start_time:.10f} seconds")

    print("Device: ", tensor_cpu.device) 

    close_cuda()

if __name__ == "__main__":
    main()
