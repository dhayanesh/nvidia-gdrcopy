Implementations of gdrcopy and cudamemcpy in C and gdrcopy wrapper for python for data movement from device to host (tensors) which is not available online. This is done for runtime comparisons of data movement using gdrcopy vs cudamemcpy for specfic data size and hardware.

Note: Verfiy required libraries are install before testing.

My setup:

## GPU Specifications

- **NVIDIA RTX A6000** (x4 GPUs)
  - Memory: 49 GB (per GPU)
  - CUDA Version: 12.2
  - Driver Version: 535.161.08
