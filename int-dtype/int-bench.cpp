#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]) {

  cudaSetDevice(3);
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate_v2(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "create cublas handle fail." << std::endl;
    return -1;
  }

  int A = atoi(argv[1]);
  int B = atoi(argv[2]);

  const int N = 1;
  const int K = 64;
  const int M = 1;
  char h_a_arr[N * K];
  char h_b_arr[K * M];
  int h_c_arr[N * M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      h_a_arr[i * K + j] = A;
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < M; ++j) {
      h_b_arr[i * M + j] = B;
    }
  }

  char* d_a_arr = nullptr;
  char* d_b_arr = nullptr;
  char* d_c_arr = nullptr;
  cudaMalloc((void**)&d_a_arr, sizeof(h_a_arr));
  cudaMalloc((void**)&d_b_arr, sizeof(h_b_arr));
  cudaMalloc((void**)&d_c_arr, sizeof(h_c_arr));
  cudaMemcpy(d_a_arr, h_a_arr, sizeof(h_a_arr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_arr, h_b_arr, sizeof(h_b_arr), cudaMemcpyHostToDevice);

  char alpha = 1;
  char beta = 0;
  status = cublasGemmEx(handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      N,
      M,
      K,
      &alpha,
      d_a_arr,
      CUDA_R_8I,
      N,
      d_b_arr,
      CUDA_R_8I,
      K,
      &beta,
      d_c_arr,
      CUDA_R_32I,
      N,
      CUDA_R_8I,
      CUBLAS_GEMM_ALGO1
      );

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "cublas gemm fail, err:" << status << std::endl;
    return status;
  }

  cudaMemcpy(h_c_arr, d_c_arr, sizeof(h_c_arr), cudaMemcpyDeviceToHost);
  std::cout << "cublas gemm sum:" << std::endl;
  for (int i = 0; i < N * N; ++i) {
    printf("%d", h_c_arr[i]);
  }
  printf("\n");

  cudaFree(d_a_arr);
  cudaFree(d_b_arr);
  cudaFree(d_c_arr);
  cublasDestroy_v2(handle);
  return 0
}
