#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define TILE_WIDTH 16.0

template <typename T>
__global__ void _addGPUKernel(T * A, T * B, T * R, int numRows, int numCols)
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

    cudaMemcpy(result, deviceR, size * sizeof(T),cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceR);
    return result;
}
