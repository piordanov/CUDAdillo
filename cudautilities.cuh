#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cublas_v2.h>
#include "cublasXt.h"
#include <type_traits>
#include <stdio.h>


#define TILE_WIDTH 16.0

template <typename T>
__global__ void _addGPUKernel(const T * A, const T * B, T * R, int numRows, int numCols)
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

template<typename T>
T* addGPU(T *A, T *B, int numRows, int numCols)
{
    T * result = (T *) malloc(numRows * numCols * sizeof(T));

    T * deviceA;
    T * deviceB;
    T * deviceR;

    const int size = numRows * numCols;

    cudaMalloc((void **) &deviceA, size * sizeof(T));
    cudaMalloc((void **) &deviceB, size * sizeof(T));
    cudaMalloc((void **) &deviceR, size * sizeof(T));

    cudaMemcpy(deviceA, A,size * sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B,size * sizeof(T),cudaMemcpyHostToDevice);

    dim3 dimGrid((int) ceil(numCols/TILE_WIDTH), (int)ceil(numRows/TILE_WIDTH), 1);
    dim3 dimBlock((int) TILE_WIDTH, (int) TILE_WIDTH, 1);

    _addGPUKernel<T><<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceR, numRows, numCols);
    cudaDeviceSynchronize();

    (cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost));

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}
template <typename T>
__global__ void _multGPUKernel(const T * A, const T * B, T * R, int numRows, int numCols)
{
    __shared__ T subTileA[(int) TILE_WIDTH][(int) TILE_WIDTH];
    __shared__ T subTileB[(int) TILE_WIDTH][(int) TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    T pVal = 0.0;

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
template <typename T>
void _gpu_blas_mmul(const T *A, const T *B, T * C, const int m, const int k, const int n)
{ }

template <>
void _gpu_blas_mmul<float>(const float *A, const float *B, float * C, const int m, const int k, const int n)
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

template <>
void _gpu_blas_mmul<double>(const double *A, const double *B, double * C, const int m, const int k, const int n)
{
    const int lda=m,ldb=k,ldc=m;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    int devices[1] = { 0 };
    cublasXtDeviceSelect(handle, 1, devices);
    cublasStatus_t stat;
    stat = cublasXtDgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Dgemm failed");
    }

    cublasXtDestroy(handle);
}



template <typename T>
T * multGPUcuBLAS(T *A, T *B, int numRows, int numCols)
{
    T * result = (T *) malloc(numRows * numCols * sizeof(T));

    T * deviceA;
    T * deviceB;
    T * deviceR;
    const int size = numRows * numCols;

    cudaMalloc((void **) &deviceA, size * sizeof(T));
    cudaMalloc((void **) &deviceB, size * sizeof(T));
    cudaMalloc((void **) &deviceR, size * sizeof(T));

    cudaMemcpy(deviceA, A,size * sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B,size * sizeof(T),cudaMemcpyHostToDevice);

    _gpu_blas_mmul(deviceA, deviceB, deviceR, numRows, numCols, numCols);
    cudaDeviceSynchronize();

    cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}

template <typename T>
T * multGPU(T *A, T *B, int numRows, int numCols)
{
    T * result = (T *) malloc(numRows * numCols * sizeof(T));

    T * deviceA;
    T * deviceB;
    T * deviceR;
    const int size = numRows * numCols;

    cudaMalloc((void **) &deviceA, size * sizeof(T));
    cudaMalloc((void **) &deviceB, size * sizeof(T));
    cudaMalloc((void **) &deviceR, size * sizeof(T));

    cudaMemcpy(deviceA, A,size * sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B,size * sizeof(T),cudaMemcpyHostToDevice);

    dim3 dimGrid((int) ceil(numCols/TILE_WIDTH), (int)ceil(numRows/TILE_WIDTH), 1);
    dim3 dimBlock((int) TILE_WIDTH, (int) TILE_WIDTH, 1);

    _multGPUKernel<T><<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceR, numRows, numCols);

    cudaDeviceSynchronize();

    cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}

