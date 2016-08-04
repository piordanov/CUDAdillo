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
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

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
        printf("Dgemm failed with error code: %s\n", cublasGetErrorString(stat));
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
        printf("Dgemm failed with error code: %s\n", cublasGetErrorString(stat));
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

    cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}

template <typename T>
void _gpu_blas_trans(const T *A, T *B, const int m, const int k)
{ }

template <>
void _gpu_blas_trans<double>(const double *M, double *R, const int m, const int k)
{
    const int lda=m,ldb=k,ldc=k;
    const double alf = 1.;
    const double bet = 0.;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t stat;
    stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, k, &alf, M, lda, &bet, M, ldb, R, ldc);
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Dgeam failed with error code: %s", cublasGetErrorString(stat));
    }
    cublasDestroy(handle);
}
template <>
void _gpu_blas_trans<float>(const float *M, float *R, const int m, const int k)
{
    const int lda=m,ldb=k,ldc=k;
    const float alf = 1.;
    const float bet = 0.;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t stat;
    stat = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, k, &alf, M, lda, &bet, M, ldb, R, ldc);
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Dgeam failed with error code: %s", cublasGetErrorString(stat));
    }
    cublasDestroy(handle);
}


template <typename T>
T * transposeGPUcuBLAS(T* M, int nRows, int nCols)
{
    const int size = nRows * nCols;
    T * result = (T *) malloc(size * sizeof(T));

    T * deviceM;
    T * deviceR;

    cudaMalloc((void **) &deviceM, size * sizeof(T));
    cudaMalloc((void **) &deviceR, size * sizeof(T));

    cudaMemcpy(deviceM, M,size * sizeof(T),cudaMemcpyHostToDevice);

    _gpu_blas_trans(deviceM,deviceR,nRows,nCols);
    cudaDeviceSynchronize();

    cudaMemcpy(result, deviceR, size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(deviceM);
    cudaFree(deviceR);
    return result;
}
template <typename T>
void _gpu_blas_cov(const T * A, const T * B, T * R, const int m, const int k)
{ }

template <>
void _gpu_blas_cov<double>(const double *A, const double *B, double *R, const int m, const int k)
{
    const int lda=m,ldb=k,ldc=m;
    const double alf = 1.;
    const double bet = 1.;
    cublasStatus_t stat;
    cublasXtHandle_t handle;
    stat = cublasXtCreate(&handle);
    int devices[1] = { 0 };
    cublasXtDeviceSelect(handle, 1, devices);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed with error code: %s\n", cublasGetErrorString(stat));
        return;
    }

    stat = cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, k, &alf, A, lda, B, ldb, &bet, R, ldc);
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Dgemm failed with error code: %s\n", cublasGetErrorString(stat));
    }
    cublasXtDestroy(handle);
}
template <>
void _gpu_blas_cov<float>(const float *A, const float *B, float *R, const int m, const int k)
{
    const int lda=m,ldb=k,ldc=m;
    const float alf = 1.;
    const float bet = 1.;
    cublasStatus_t stat;
    cublasXtHandle_t handle;
    stat = cublasXtCreate(&handle);
    int devices[1] = { 0 };
    cublasXtDeviceSelect(handle, 1, devices);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed with error code: %s\n", cublasGetErrorString(stat));
        return;
    }

    stat = cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, k, &alf, A, lda, B, ldb, &bet, R, ldc);
    cudaDeviceSynchronize();
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Dgemm failed with error code: %s\n", cublasGetErrorString(stat));
    }
    cublasXtDestroy(handle);
}


template <typename T>
T * covGPUcuBLAS(T * A, T * B, int nRows, int nCols)
{
    const int size = nRows * nCols;
    T * result = (T *) malloc(size * sizeof(T));

    T * deviceA;
    T * deviceB;
    T * deviceR;

    cudaMalloc((void **) &deviceA, size * sizeof(T));
    cudaMalloc((void **) &deviceB, size * sizeof(T));
    cudaMalloc((void **) &deviceR, size * sizeof(T));

    cudaMemcpy(deviceA, A,size * sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B,size * sizeof(T),cudaMemcpyHostToDevice);

    _gpu_blas_cov(deviceA, deviceB, deviceR, nRows, nCols);
    cudaDeviceSynchronize();

    cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}
