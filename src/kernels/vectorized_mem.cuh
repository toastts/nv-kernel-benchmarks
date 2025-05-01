#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  uint blk_r = blockIdx.y;
  uint blk_c = blockIdx.x;

  uint tcol = threadIdx.x % (BN / TN);
  uint trow = threadIdx.x / (BN / TN);

  __shared__ float shmA[BM * BK];
  __shared__ float shmB[BK * BN];

  // move ptrs to tile start
  float *a_ptr = A + blk_r * BM * K;
  float *b_ptr = B + blk_c * BN;
  float *c_ptr = C + blk_r * BM * N + blk_c * BN;

  // vector load indices (4-float loads)
  uint a_row4 = threadIdx.x / (BK / 4);
  uint a_col4 = threadIdx.x % (BK / 4);
  uint b_row4 = threadIdx.x / (BN / 4);
  uint b_col4 = threadIdx.x % (BN / 4);

  float acc_tile[TM * TN] = {0.0f};
  float regA[TM] = {0.0f};
  float regB[TN] = {0.0f};

  for (uint k0 = 0; k0 < K; k0 += BK) {
    // load and transpose A tile via float4
    float4 vA = reinterpret_cast<float4*>(&a_ptr[a_row4 * K + a_col4 * 4])[0];
    shmA[(a_col4 * 4 + 0) * BM + a_row4] = vA.x;
    shmA[(a_col4 * 4 + 1) * BM + a_row4] = vA.y;
    shmA[(a_col4 * 4 + 2) * BM + a_row4] = vA.z;
    shmA[(a_col4 * 4 + 3) * BM + a_row4] = vA.w;
    
    // load B tile via float4
    float4 vB = reinterpret_cast<float4*>(&b_ptr[b_row4 * N + b_col4 * 4])[0];
    reinterpret_cast<float4*>(&shmB[b_row4 * BN + b_col4 * 4])[0] = vB;
    __syncthreads();

    a_ptr += BK;
    b_ptr += BK * N;

    for (uint k1 = 0; k1 < BK; ++k1) {
      // bring A row into regs
      for (uint i = 0; i < TM; ++i)
        regA[i] = shmA[k1 * BM + trow * TM + i];
      // bring B col chunk into regs
      for (uint j = 0; j < TN; ++j)
        regB[j] = shmB[k1 * BN + tcol * TN + j];
      // outer prod update
      for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
          acc_tile[i * TN + j] += regA[i] * regB[j];
    }
    __syncthreads();
  }

  // write back via float4 updates
  for (uint i = 0; i < TM; ++i) {
    for (uint j = 0; j < TN; j += 4) {
      uint off = (trow * TM + i) * N + tcol * TN + j;
      float4 c4 = reinterpret_cast<float4*>(&c_ptr[off])[0];
      c4.x = alpha * acc_tile[i * TN + j]     + beta * c4.x;
      c4.y = alpha * acc_tile[i * TN + j + 1] + beta * c4.y;
      c4.z = alpha * acc_tile[i * TN + j + 2] + beta * c4.z;
      c4.w = alpha * acc_tile[i * TN + j + 3] + beta * c4.w;
      reinterpret_cast<float4*>(&c_ptr[off])[0] = c4;
    }
  }
}
