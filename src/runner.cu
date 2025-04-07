#include "kernels.cuh"
#include "kernels/cublas.cuh"
#include "kernels/global_mem.cuh"
#include "runner.cuh"
#include "util.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

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
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  cublasFP32(handle, M, N, K, &alpha, A, B, &beta, C);
}

void run_simple_gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  simple_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_global_mem_gemm(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  global_mem_gemm<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
