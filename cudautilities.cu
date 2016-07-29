#include "cudautilities.cuh"

template float * addGPU<float>(float *, float *, int, int);
template double * addGPU<double>(double *, double *, int, int);

template float * multGPU<float>(float *, float *, int, int);
template double * multGPU<double>(double *, double *, int, int);
