# CUTLASS vs cuBLAS kernel implementation benchmarks
## setup on SOL supercomputer
- reserve hardware `interactive -c 4 -G a100:1 -t 120 --mem=32G`
- load modules
  - `ml cuda-12.6.1-gcc-12.1.0`
  - `ml mamba/latest`
- set nvidia compiler env var `export CUDACXX=${CUDA_HOME}/bin/nvcc`
- run `nvidia-smi` for device info dump, make sure A100 is listed

## setup locally
- install the cuda toolkit
- figure out your $CUDAPATH (/opt/cuda on many systems)
- check CMakeLists.txt and look at the line `set(CMAKE_CUDA_ARCHITECTURES 89)`, make sure your GPU compute compatability matches [here](https://developer.nvidia.com/cuda-gpus)
  - this is 8.9 formatted as 89, so 8.0 would be 80 for reference


## building
- `mkdir build`
- `cmake -B build/ -S .`
- now run `./build/gemm [0-5]`
  - 0 is cuBLAS
  - 1 is simple gemm
  - 2 is global mem coalescing
  - 3 is 1D blocktiling
  - 4 is 2D blocktiling
  - 5 is vectorized mem accesses


## resources
- cublas docs: https://docs.nvidia.com/cuda/cublas/#cublasgemmex
- cutlass quickstart: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/quickstart.md
- great matrix mult primer/explanation: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication
- https://github.com/Bruce-Lee-LY/cuda_hgemm
- https://github.com/xlite-dev/LeetCUDA
- https://github.com/lessw2020/cutlass-kernels
