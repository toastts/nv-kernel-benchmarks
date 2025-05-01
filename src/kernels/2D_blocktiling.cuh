#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN),
                             1) void blocktile_2D(int M, int N, int K,
                                                  float alpha, const float *A,
                                                  const float *B, float beta,
                                                  float *C) {
  // 2d tiles of bm x bn, each thread does tm x tn
  uint br = blockIdx.y;
  uint bc = blockIdx.x;

  uint total = BM * BN;
  uint threads = total / (TM * TN);
  assert(threads == blockDim.x);

  uint tc = threadIdx.x % (BN / TN);
  uint tr = threadIdx.x / (BN / TN);

  __shared__ float smA[BM * BK];
  __shared__ float smB[BK * BN];

  // ptrs to start of tile
  const float *aPtr = A + br * BM * K;
  const float *bPtr = B + bc * BN;
  float *cPtr = C + br * BM * N + bc * BN;

  // calc load offsets
  uint ira = threadIdx.x / BK;
  uint ica = threadIdx.x % BK;
  uint strideA = threads / BK;

  uint irb = threadIdx.x / BN;
  uint icb = threadIdx.x % BN;
  uint strideB = threads / BN;

  float regTile[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint k0 = 0; k0 < K; k0 += BK) {
    // load tile into shared mem
    for (uint off = 0; off < BM; off += strideA)
      smA[(ira + off) * BK + ica] = aPtr[(ira + off) * K + ica];

    for (uint off = 0; off < BK; off += strideB)
      smB[(irb + off) * BN + icb] = bPtr[(irb + off) * N + icb];

    __syncthreads();

    aPtr += BK;
    bPtr += BK * N;

    for (uint k1 = 0; k1 < BK; ++k1) {
      // load registers per thread
      for (uint i = 0; i < TM; ++i)
        regM[i] = smA[(tr * TM + i) * BK + k1];
      for (uint j = 0; j < TN; ++j)
        regN[j] = smB[k1 * BN + tc * TN + j];

      // compute tm x tn results
      for (uint m1 = 0; m1 < TM; ++m1)
        for (uint n1 = 0; n1 < TN; ++n1)
          regTile[m1 * TN + n1] += regM[m1] * regN[n1];
    }
    __syncthreads();
  }

  // write back scaled results
  for (uint m2 = 0; m2 < TM; ++m2)
    for (uint n2 = 0; n2 < TN; ++n2) {
      uint off = (tr * TM + m2) * N + tc * TN + n2;
      cPtr[off] = alpha * regTile[m2 * TN + n2] + beta * cPtr[off];
    }
}
