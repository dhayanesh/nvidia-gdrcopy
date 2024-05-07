Implementations of gdrcopy and cudamemcpy in C and gdrcopy wrapper for python for data movement from device to host (tensors) which is not available online. This is done for runtime comparisons of data movement using gdrcopy vs cudamemcpy for specfic data size and hardware.

Note: Verfiy required libraries are install before testing.

My setup:
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               Off | 00000000:01:00.0 Off |                  Off |
| 30%   47C    P2             259W / 300W |  45622MiB / 49140MiB |     90%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000               Off | 00000000:2C:00.0 Off |                  Off |
| 30%   54C    P2             254W / 300W |  45578MiB / 49140MiB |     18%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA RTX A6000               Off | 00000000:41:00.0 Off |                  Off |
| 30%   47C    P2             244W / 300W |  45578MiB / 49140MiB |     81%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA RTX A6000               Off | 00000000:61:00.0 Off |                  Off |
| 30%   52C    P2             243W / 300W |  45458MiB / 49140MiB |     29%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
