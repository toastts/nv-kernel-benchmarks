#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void blocktile_1D(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // 2d blocks of bx*by threads, each covers bm x bn of C
  uint br = blockIdx.y;
  uint bc = blockIdx.x;

  uint tc = threadIdx.x % BN;
  uint tr = threadIdx.x / BN;

  __shared__ float smA[BM * BK];
  __shared__ float smB[BK * BN];

  // move ptrs to tile start
  const float *aPtr = A + br * BM * K;
  const float *bPtr = B + bc * BN;
  float *cPtr = C + br * BM * N + bc * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);

  uint ica = threadIdx.x % BK;
  uint ira = threadIdx.x / BK;
  uint icb = threadIdx.x % BN;
  uint irb = threadIdx.x / BN;

  float reg[TM] = {0.0f};

  for (uint k0 = 0; k0 < K; k0 += BK) {
    // load tile into smem
    smA[ira * BK + ica] = aPtr[ira * K + ica];
    smB[irb * BN + icb] = bPtr[irb * N + icb];
    __syncthreads();

    aPtr += BK;
    bPtr += BK * N;

    for (uint k1 = 0; k1 < BK; ++k1) {
      float tmp = smB[k1 * BN + tc];
      for (uint m1 = 0; m1 < TM; ++m1) {
        reg[m1] += smA[(tr * TM + m1) * BK + k1] * tmp;
      }
    }
    __syncthreads();
  }

  // write back with scaling
  for (uint m2 = 0; m2 < TM; ++m2) {
    uint off = (tr * TM + m2) * N + tc;
    cPtr[off] = alpha * reg[m2] + beta * cPtr[off];
  }
}
