#include "cudautilities.cuh"

template float * addGPU<float>(float *, float *, int, int);
template double * addGPU<double>(double *, double *, int, int);

template float * multGPUcuBLAS<float>(float *, float *, int, int);
template double * multGPUcuBLAS<double>(double *, double *, int, int);

template float * transposeGPUcuBLAS<float>(float*,int,int);
template double * transposeGPUcuBLAS<double>(double*,int,int);

template float * covGPUcuBLAS<float>(float *, float *, int, int);
template double * covGPUcuBLAS<double>(double *, double *, int, int);
