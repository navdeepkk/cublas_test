// RUN: nvcc -lcurand -lcublas -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -expt-relaxed-constexpr
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "cublas_v2.h"
#include "curand.h"
#include "cuda_fp16.h"
#include <math.h>
#include <time.h>
#include <library_types.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include "common.h"

using namespace std;

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = (in[idx]);
  }
}

__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = (in[idx]);
  }
}

void print_matrix(float *A, int nr_rows_A, int nr_cols_A) { 
  for(int i = 0; i < nr_rows_A; i++){
    for(int j = 0; j < nr_cols_A; j++){
      std::cout << A[i * nr_cols_A + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Fill the array with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  // Create a pseudo-random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  
  // Set the seed for the random number generator using the system clock
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  
  // Fill the array with random numbers on the device
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(__half *A, __half *B, __half *C, int m, int k, int n, int iter) {
  const __half alf = 1.0f;
  const __half bet = 0.0f;
  const __half *alpha = &alf;
  const __half *beta = &bet;
  
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasStatus_t cublasStat  = cublasCreate(&handle);
  
  // Set the math mode to allow cuBLAS to use Tensor Cores:
  cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  //cublasStat = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  // n maps to the output dimension.
  // m is the batch size * seq length.
  // k maps to the input dimension.

  // leading dimension of B will be rows in B(host) and it will be accessed as T.
  // leading dimension of A will be rows in A(host) and it will be accessed as N.
  // Leading dimension of C will be cols in C(host) and it will be accesses as N. 

  // A is m * k(non transposed, row major) in host, device will expect it to be
  // in col major and hence will see it as a m * k matrix.
  // B is n * K(trasnposed) in host, device will expect it to be in col major
  // and hence will see it as a k * n matrix. 
  // C should be m * n in host but and the required computation is C = A B' and we are calculating
  // C' = B A', beacause we have A as A' and B' as B in the device because it
  // assumes data to be in col major format, the output will be n * m in device. 

  // A operand in cblas call will acutally be the weight matrix.
  // B operand in cbals call will acutally be the activation matrix.

  // m will be rows A, C.
  // k will be cols A, B.
  // n will be rows B, cols in C.
  // int lda = k, ldb = k, ldc = n;
  int lda = m, ldb = k, ldc = n;
  float matmulTime = 0.0f;	

  //-------------------------------peforming warmup runs-------------------------------------//
  for(int i = 0; i < 1; i++){
    // Do the actual multiplication
    check_cuda_error(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
	  /*number of rows of matrix op(A) and C*/n,
	  /*number of columns of matrix op(B) and C*/m,
	  /*number of columns of op(A) and rows of op(B)*/k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  //-------------------------------------perform actual runs--------------------------------//
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int niter = iter;
  float mintime = 1000.0f;
  for(int i = 0; i < niter; i++){
    //Do the actual multiplication
    cudaEventRecord(start, NULL);
    check_cuda_error(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
	  /*number of rows of matrix op(A) and C*/n,
	  /*number of columns of matrix op(B) and C*/m,
	  /*number of columns of op(A) and rows of op(B)*/k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaEventRecord(stop, NULL);

    //stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    if(msecTotal < mintime)
      mintime = msecTotal;
    matmulTime += msecTotal;
  }

  double flopsPerMatrixMul = 2.0 * (double) m * (double) n * (double) k;
  //double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (matmulTime / niter / 1000.0f);
  double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (matmulTime / niter / 1000.0f);
  std::cout<<m<<", "<<n<<", "<<k<<", "<<teraFlops<<" FLOPs: "<<flopsPerMatrixMul<<std::endl;
  
  // Destroy the handle
  cublasDestroy(handle);
}

int main(int argc, char * argv[]) {
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // A is for the Activations. has dimensions m * k, where m is (seq length * batchsize),
  // k is no of inputs to the layer. 
  // B is for the weights. stored as B' at host. has dimensions n * k. n is the number of outputs,
  // k is the no of inputs to the layer.  
  // C is the output matrix. has dimensions m * n.
  // The required Operations is A B'.
 
  if(argc != 5){
    printf("Min args required are 4, aborting!.\n");
    return 0;
  }

  // set dims according to operation c = a * b'
  char * mc = argv[1];
  char * kc = argv[2];
  char * nc = argv[3];
  char * iter = argv[4];

  int m = atoi(mc);
  int k = atoi(kc);
  int n = atoi(nc);
  int niter = atoi(iter); 

  nr_rows_A = m;				
  nr_cols_A = k;					
  nr_rows_B = k;							
  nr_cols_B = n;					
  nr_rows_C = m;					
  nr_cols_C = n;					

  //nr_rows_A = 1536;
  //nr_cols_A = 1024;
  //nr_rows_B = 1024;
  //nr_cols_B = 1024;
  //nr_rows_C = 1536;
  //nr_cols_C = 1024;

  // Allocate arrays on GPU.
  // array on device of type half.
  // float because curand generates only fp32 numbers.
  // __half arrays for fp16 numbers.
  float *df_A, *df_B;	
  __half *d_A, *d_B, *d_C;

  check_cuda_error(cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(__half)));
  check_cuda_error(cudaMalloc(&df_A,nr_rows_A * nr_cols_A * sizeof(float)));
  GPU_fill_rand(df_A, nr_rows_A, nr_cols_A);	
  convertFp32ToFp16 <<< (nr_rows_A * nr_cols_A+ 255) / 256, 256 >>> (d_A, df_A, nr_rows_A * nr_cols_A);

  check_cuda_error(cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(__half)));
  check_cuda_error(cudaMalloc(&df_B,nr_rows_B * nr_cols_B * sizeof(float)));
  GPU_fill_rand(df_B, nr_rows_B, nr_cols_B);	
  convertFp32ToFp16 <<< (nr_rows_B * nr_cols_B + 255) / 256, 256 >>> (d_B, df_B, nr_rows_B * nr_cols_B);

  check_cuda_error(cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(__half)));

  // m will be rows a.
  // k will be cols a.
  // n will be rows b.
  // call the matmul function.
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_rows_B, niter);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(df_A);
  cudaFree(df_B);
  cudaFree(d_C);

  return 0;
}
