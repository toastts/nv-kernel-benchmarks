#pragma once
#include <cuda_runtime.h>

// runtime-parametrized vectorized SGEMM kernel
// launches with: <<<grid, threads, (BM*BK+BN*BK)*sizeof(float)>>>
extern "C" __global__ void vec_mem_generic(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C, int BM, int BN,
                                           int BK, int TM, int TN) {
  // dynamic shared mem consists of two tiles back-to-back
  extern __shared__ float shm[];
  float *shmA = shm;
  float *shmB = shm + BM * BK;

  // tile coords
  int tile_r = blockIdx.y;
  int tile_c = blockIdx.x;

  // global pointers at tile origin
  const float *a_base = A + tile_r * BM * K;
  const float *b_base = B + tile_c * BN;
  float *c_base = C + tile_r * BM * N + tile_c * BN;

  // thread’s column/row within the BM×BN tile
  int t = threadIdx.x;
  int threads = (BM * BN) / (TM * TN);
  int tcol = t % (BN / TN);
  int trow = t / (BN / TN);

  // each thread accumulates a TM×TN sub-tile
  // allocate in registers
  float acc[64]; // max TM*TN <= 64
  for (int i = 0; i < TM * TN; ++i)
    acc[i] = 0.f;

  // loop over K in stripes
  for (int k0 = 0; k0 < K; k0 += BK) {
    // load A-tile (BM×BK) into shmA, B-tile (BK×BN) into shmB
    int idx = threadIdx.x;
    int Asz = BM * BK, Bsz = BK * BN;
    if (idx < Asz)
      shmA[idx] = a_base[idx];
    if (idx < Bsz)
      shmB[idx] = b_base[idx * N / BN];
    // careful: b_base[idx] is B[row * N + col], but you can
    // map idx→(row,col) = (idx/BN, idx%BN):
    //   shmB[row*BN + col] = b_base[row*N + col];

    __syncthreads();

    // advance pointers
    a_base += BK;
    b_base += BK * N;

    // compute outer-product over this stripe
    for (int k1 = 0; k1 < BK; ++k1) {
      // load one A-column into regs
      float a_reg[16]; // max TM<=16
      for (int i = 0; i < TM; ++i)
        a_reg[i] = shmA[k1 * BM + (trow * TM + i)];
      // load one B-row into regs
      float b_reg[16]; // max TN<=16
      for (int j = 0; j < TN; ++j)
        b_reg[j] = shmB[k1 * BN + (tcol * TN + j)];
      // accumulate TM×TN
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          acc[i * TN + j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }

  // write back TM×TN block
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = trow * TM + i;
      int col = tcol * TN + j;
      c_base[row * N + col] =
          alpha * acc[i * TN + j] + beta * c_base[row * N + col];
    }
  }
}
