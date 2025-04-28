#include "kernels.cuh"
#include "kernels/1D_blocktiling.cuh"
#include "runner.cuh"
#include "util.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order -> change A & B, since (B^T*A^T)^T = (A*B)
  // |-> runs cuBLAS in full fp32 mode
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_simple_gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  simple<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_global_mem_gemm(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  global_mem<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_1D_blocktiling_gemm(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  blocktile_1D<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_2D_blocktiling_gemm(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  //NOTE: no bounds checking on the kernel here!! don't pass small sizes <128,
  // we can't tile that properly :3
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  blocktile_2D<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    run_cublas(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_simple_gemm(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_global_mem_gemm(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_1D_blocktiling_gemm(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    run_2D_blocktiling_gemm(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
