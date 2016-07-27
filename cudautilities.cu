#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "cublasXt.h"
#include <stdio.h> // <-- added for 'printf'

extern "C"
{
    const float TILE_WIDTH = 16.0;
    void zero(double *l_p_array, int a_numElements);
    float * add(float *A, float *B, int numRows, int numCols);
    float * multiply(float *A, float *B, int numRows, int numCols);

}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
            printf("%f ",A[j * nr_rows_A + i]);
         }
        printf("\n");
    }
     printf("\n");
}

//test function for CUDA
__global__ void zero_GPU(double *l_p_array_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf(" %i: Hello World!\n",i);
    l_p_array_gpu[i] = 0;
}



void zero(double *l_p_array, int a_numElements)
{
    double *l_p_array_gpu;
    int size = a_numElements * int(sizeof(double));
    cudaMalloc((void**) &l_p_array_gpu, size);
    cudaMemcpy(l_p_array_gpu, l_p_array, size, cudaMemcpyHostToDevice);

    zero_GPU<<<1,a_numElements>>>(l_p_array_gpu);

    cudaMemcpy(l_p_array,l_p_array_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(l_p_array_gpu);

}

__global__ void add_GPU(float * A, float * B, float * R, int numRows, int numCols)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int index = row * numCols + col;
    //printf(" %f ",A[index]);
    if(row < numRows && col < numCols)
    {
        R[index] = A[index] + B[index];
    }

}

float* add(float *A, float *B, int numRows, int numCols)
{
    float * result = (float *) malloc(numRows * numCols * sizeof(float));

    float * deviceA;
    float * deviceB;
    float * deviceR;

    const int size = numRows * numCols;

    cudaMalloc((void **) &deviceA, size * sizeof(float));
    cudaMalloc((void **) &deviceB, size * sizeof(float));
    cudaMalloc((void **) &deviceR, size * sizeof(float));

    _cudaGetErrorEnum(cudaMemcpy(deviceA, A,size * sizeof(float),cudaMemcpyHostToDevice));
    _cudaGetErrorEnum(cudaMemcpy(deviceB, B,size * sizeof(float),cudaMemcpyHostToDevice));

    dim3 dimGrid((int) ceil(numCols/TILE_WIDTH), (int)ceil(numRows/TILE_WIDTH), 1);
    dim3 dimBlock((int) TILE_WIDTH, (int) TILE_WIDTH, 1);

    add_GPU<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceR, numRows, numCols);
    cudaDeviceSynchronize();

    cudaMemcpy(result, deviceR, size * sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}
__global__ void multiply_GPU(float * A, float * B, float * R, int numRows, int numCols)
{
    __shared__ float subTileA[(int) TILE_WIDTH][(int) TILE_WIDTH];
    __shared__ float subTileB[(int) TILE_WIDTH][(int) TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float pVal = 0.0;

    for(int m = 0; m < ceil(numCols / TILE_WIDTH); m++)
    {
        int Alocal = row * numCols + m*TILE_WIDTH+tx;
        int Blocal = (m*TILE_WIDTH+ty) * numCols + col;

        if( (m*TILE_WIDTH+tx) >= numCols || row >= numRows)
            subTileA[ty][tx] = 0.0;
        else
            subTileA[ty][tx] = A[Alocal];
        if( (m*TILE_WIDTH+ty) >=numRows || col >= numCols)
            subTileB[ty][tx] = 0.0;
        else
            subTileB[ty][tx] = B[Blocal];

        __syncthreads();
        for(int k = 0; k < TILE_WIDTH; k++)
        {
            if((m*TILE_WIDTH+k) < numCols)
            {
                //printf("adding to pVal: %f \n", pVal);
                pVal += subTileA[ty][k] * subTileB[k][tx];
            }
        }
        __syncthreads();
    }
    if((row < numRows && col < numCols))
        R[row*numCols+col] = pVal;

}

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
     const int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     cublasXtHandle_t handle;
     cublasXtCreate(&handle);
     int devices[1] = { 0 };
     cublasXtDeviceSelect(handle, 1, devices);
     cublasStatus_t stat;

     stat = cublasXtSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

     cudaDeviceSynchronize();
     if(stat != CUBLAS_STATUS_SUCCESS)
     {
         printf("Dgemm failed");
     }

     cublasXtDestroy(handle);
}

float * multiply(float *A, float *B, int numRows, int numCols)
{
    float * result = (float *) malloc(numRows * numCols * sizeof(float));

    float * deviceA;
    float * deviceB;
    float * deviceR;
    const int size = numRows * numCols;
//    printf("A[4] = %f",A[4]);
//    printf("B[4] = %f",B[4]);
    //printf("allocating memory on device...\n");

    _cudaGetErrorEnum(cudaMalloc((void **) &deviceA, size * sizeof(float)));
    _cudaGetErrorEnum(cudaMalloc((void **) &deviceB, size * sizeof(float)));
    _cudaGetErrorEnum(cudaMalloc((void **) &deviceR, size * sizeof(float)));

    //printf("copying memory to device\n");

    _cudaGetErrorEnum(cudaMemcpy(deviceA, A,size * sizeof(float),cudaMemcpyHostToDevice));
    _cudaGetErrorEnum(cudaMemcpy(deviceB, B,size * sizeof(float),cudaMemcpyHostToDevice));

//    dim3 dimGrid((int) ceil(numCols/TILE_WIDTH), (int)ceil(numRows/TILE_WIDTH), 1);
//    dim3 dimBlock((int) TILE_WIDTH, (int) TILE_WIDTH, 1);

//    printf("Executing kernel...\n");

//    multiply_GPU<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceR, numRows, numCols);
//    printf(_cudaGetErrorEnum(cudaPeekAtLastError()));
//    printf("\n");

    gpu_blas_mmul(deviceA, deviceB, deviceR, numRows, numCols, numCols);
    cudaDeviceSynchronize();

    //printf("Copying result to host\n");
    _cudaGetErrorEnum(cudaMemcpy(result, deviceR, size * sizeof(float),cudaMemcpyDeviceToHost));

    //print_matrix(result,numRows,numCols);
    //printf("%f",result[45]);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}
