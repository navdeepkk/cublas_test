// nvcc 001 isamax .c -lcublas
#include <iostream>
#include </usr/include/stdio.h>
#include </usr/include/stdlib.h>
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
/*
__global__ void convertFp32ToFp16 (__half *out, float *in, int rows, int cols) {
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){   
				out[i * cols + j] = __float2half(in[i * cols + j]);
			}
		}
}
*/
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
void GPU_fill_rand(float *A, int A_size) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, A_size);
}


int batch_size, num_heads, size_per_head, seq_len, hidden_size;

	
void gpu_blas_mmul(__half *A, __half *B, __half *C) {
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
		cublasStatus_t cublasStat  = cublasCreate(&handle);
		// Set the math mode to allow cuBLAS to use Tensor Cores:
		cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
		//cublasStat = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		

		//leading dimension of B will be cols in B(host) and it will be accessed as T.
		//leading dimension of A will be cols in A(host) and it will be accessed as N.
		//Leading dimension of C will be cols in C(host) and it will be accesses as N. 
			
		//A is m * k in host k * m in device.
		//B is n * K in host k * n in device. 
		//C is m * n in host n * m in device. 

		//m will be rows A, C.
		//k will be cols A, B.
		//n will be rows B, cols in C.
		//-------------------------------------------------------performing warmup runs--------------------------------------------//
		for(int i = 0; i < 1; i++){
			// Do the actual multiplication
		// Second gemm
			cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, size_per_head, seq_len, alpha, B, CUDA_R_16F, seq_len, seq_len * seq_len, A, CUDA_R_16F, seq_len, seq_len * size_per_head, beta, C, CUDA_R_16F, seq_len, seq_len * size_per_head, num_heads * batch_size, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
}
		//-------------------------------------------------------------performing actual runs---------------------------------------//
		cudaDeviceSynchronize();
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		//cudaEventRecord(start, NULL);
		int niter = 1;
		float minTime = 100.0f;
		for(int i = 0; i < niter; i++){
			// Do the actual multiplication
			cudaEventRecord(start, NULL);
			//cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, size_per_head, size_per_head, seq_len, alpha, B, CUDA_R_16F, seq_len, seq_len * size_per_head, A, CUDA_R_16F, seq_len, seq_len * size_per_head, beta, C, CUDA_R_16F, size_per_head, size_per_head * size_per_head, num_heads * batch_size, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
			//secon gemm
			cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, size_per_head, seq_len, alpha, B, CUDA_R_16F, seq_len, seq_len * seq_len, A, CUDA_R_16F, seq_len, seq_len * size_per_head, beta, C, CUDA_R_16F, seq_len, seq_len * size_per_head, num_heads * batch_size, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		
		cudaEventRecord(stop, NULL);

		//stop event to complete
		cudaEventSynchronize(stop);

		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		if(msecTotal < minTime){
				minTime = msecTotal;
		}
}
/*
		cudaEventRecord(stop, NULL);

		//stop event to complete
		cudaEventSynchronize(stop);

		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		cout<<"Total time Taken: "<<msecTotal<<" msec"<<endl;

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal/niter;
		cout<<"Average time taken per matmul: "<<msecPerMatrixMul<<" msec"<<endl;
*/		
		double flopsPerMatrixMul = 2.0 * (double) seq_len * (double)seq_len  * (double)size_per_head * (double) batch_size * (double) num_heads;
		double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (minTime / 1000.0f);
		printf(
				"Performance= %.2f TFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
				teraFlops,
				minTime,
				flopsPerMatrixMul);
/*
		for(int i = 0; i < 20; i++){
			cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_16F, lda, 384 * 384, B, CUDA_R_16F, ldb, 384 * 64, beta, C, CUDA_R_16F, ldc, 384 * 64, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		}
*/
    // Destroy the handle
    cublasDestroy(handle);
}

int main() {
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C, A_size, B_size, C_size;
		batch_size = 4;
		seq_len = 384;
		num_heads = 16;
		hidden_size = 1024;	
		size_per_head = hidden_size / num_heads;
		A_size =  batch_size * num_heads * seq_len * size_per_head;
		B_size =  batch_size * num_heads * seq_len * size_per_head;
		C_size = batch_size * num_heads * size_per_head * size_per_head; 	 
		
   		//second gemm 
		//A_size =  batch_size * num_heads * seq_len * size_per_head;
		//B_size =  batch_size * num_heads * seq_len * seq_len;
		//C_size = batch_size * num_heads * seq_len * size_per_head; 	 
		//A is for the matrix Q. 
		//B is for the matrix K.
		//C is the output matrix. 
		//Matmul will be A B.

		nr_rows_A = seq_len;
		nr_cols_A = size_per_head;
		nr_rows_B = seq_len;
		nr_cols_B =	seq_len;
		nr_rows_C = size_per_head;
		nr_cols_C = seq_len;

    // Allocate 6 arrays on GPU.
		// array on device of type half.
		// float because curand generates only fp32 numbers.
		// __half arrays for fp16 numbers.
		float *df_A, *df_B, *df_C;	
    __half *d_A, *d_B, *d_C;


    check_cuda_error(cudaMalloc(&d_A, A_size * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_A, A_size * sizeof(float)));
		GPU_fill_rand(df_A, A_size);	
		convertFp32ToFp16 <<< (A_size+ 255) / 256, 256 >>> (d_A, df_A, A_size);

 
		check_cuda_error(cudaMalloc(&d_B, B_size * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_B, B_size * sizeof(float)));
		GPU_fill_rand(df_B, B_size);	
		convertFp32ToFp16 <<< (B_size + 255) / 256, 256 >>> (d_B, df_B, B_size);
    
		check_cuda_error(cudaMalloc(&d_C, C_size * sizeof(__half)));
    check_cuda_error(cudaMalloc(&df_C, C_size* sizeof(float)));
 
			
		
  
		//m will be rows a.
		//k will be cols a.
		//n will be rows b.
		//call the matmul function.
	  gpu_blas_mmul(d_A, d_B, d_C);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);


    return 0;
}
