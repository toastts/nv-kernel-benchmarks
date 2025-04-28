#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

const char *kernel_names[] = {"cuBLAS FP32", "Naive GEMM",
                              "Global Memory Coalescing GEMM"};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "select kernel range from [0,2]" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 4) {
    std::cerr << "Please enter a valid kernel number [0,4]" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %s on device %d.\n", kernel_names[kernel_num],
         deviceIdx);

  // print some device info
  CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    float h2d_time, d2h_time, kernel_time;
    
    // Time H2D transfers
    cudaEvent_t h2d_start, h2d_end;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_end);
    
    cudaEventRecord(h2d_start);
    cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_end);
    cudaEventSynchronize(h2d_end);
    cudaEventElapsedTime(&h2d_time, h2d_start, h2d_end);
    
    // Time kernel execution
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    
    cudaEventRecord(kernel_start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    
    // Calculate metrics
    long total_bytes = (m * k + k * n + m * n) * sizeof(float);
    float gb_per_sec = (total_bytes * repeat_times) / (1e9 * kernel_time/1000.0);
    long flops = 2 * m * n * k; // Keep original FLOPS calculation
    float flops_per_byte = (float)flops / total_bytes;
    
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS, M B/W: (%7.1f) GB/s, AI: (%4.1f) FLOPS/B, H2D: (%4.1f%%), size: (%ld).\n",
        kernel_time / (repeat_times * 1000.0),
        (repeat_times * flops * 1e-9) / (kernel_time/1000.0),
        gb_per_sec,
        flops_per_byte,
        (h2d_time/kernel_time) * 100,
        m);

    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
