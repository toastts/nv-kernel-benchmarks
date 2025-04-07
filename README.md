# CUTLASS vs cuBLAS kernel implementation benchmarks
## setup
- reserve hardware `interactive -c 4 -G a100:1 -t 120 --mem=32G`
- load modules
  - `ml cuda-12.6.1-gcc-12.1.0`
  - `ml mamba/latest`
- set nvidia compiler env var `export CUDACXX=${CUDA_HOME}/bin/nvcc`
- run `nvidia-smi` for device info dump, make sure A100 is listed

## building
- `mkdir build && cd build`


## resources
- cublas docs: https://docs.nvidia.com/cuda/cublas/#cublasgemmex
- cutlass quickstart: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/quickstart.md
