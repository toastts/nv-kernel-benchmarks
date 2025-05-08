#pragma once
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// dynamic‐shared holds BM*BK + BK*BN floats
extern __shared__ float _smem[];
#define smA (_smem)
#define smB (_smem + BM * BK)

extern "C" __global__ void blocktile2d_generic(int M, int N, int K, float alpha,
                                               const float *A, const float *B,
                                               float beta, float *C, int BM,
                                               int BN, int BK, int TM, int TN) {
  // tile coords
  int br = blockIdx.y, bc = blockIdx.x;
  int threads = (BM * BN) / (TM * TN);
  assert(blockDim.x == threads);

  // thread‐local sub‐tile indices
  int tc = threadIdx.x % (BN / TN);
  int tr = threadIdx.x / (BN / TN);

  // pointers to the corner of this BM×BN block
  const float *aPtr = A + br * BM * K;
  const float *bPtr = B + bc * BN;
  float *cPtr = C + br * BM * N + bc * BN;

  // how many loads per thread
  int strideA = threads / BK;
  int strideB = threads / BN;

  // accumulators
  float regTile[/*TM*TN*/ 256]; // static max bounds, you know TM,TN ≤16
  float regM[16], regN[16];
#pragma unroll
  for (int i = 0; i < TM * TN; ++i)
    regTile[i] = 0;

  // loop over K‐stripes
  for (int k0 = 0; k0 < K; k0 += BK) {
// load A‐tile
#pragma unroll
    for (int off = 0; off < BM; off += strideA)
      smA[(threadIdx.x / BK + off) * BK + (threadIdx.x % BK)] =
          aPtr[(threadIdx.x / BK + off) * K + (threadIdx.x % BK)];

// load B‐tile
#pragma unroll
    for (int off = 0; off < BK; off += strideB)
      smB[(threadIdx.x / BN + off) * BN + (threadIdx.x % BN)] =
          bPtr[(threadIdx.x / BN + off) * N + (threadIdx.x % BN)];

    __syncthreads();

    aPtr += BK;
    bPtr += BK * N;
// do TM×TN multiply
#pragma unroll
    for (int k1 = 0; k1 < BK; ++k1) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
        regM[i] = smA[(tr * TM + i) * BK + k1];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        regN[j] = smB[k1 * BN + tc * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          regTile[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

// writeback
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int off = (tr * TM + i) * N + tc * TN + j;
      cPtr[off] = alpha * regTile[i * TN + j] + beta * cPtr[off];
    }
}
