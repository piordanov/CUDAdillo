# CUDAdillo

A static c++ library to complete basic matrix operations on Armadillo matrices with the intent to speed up runtime through the use of the CUDA and cuBLAS GPU libraries.

#Qt Creator Integration
- Have the `cudadillo.h` file in the same working directory
- Have the compiled `libCUDAdillo.a` in the project, preferably a Libraries folder
- Then include the following in the project's .pro file

```
macx{
    LIBS += -L/Developer/NVIDIA/CUDA-7.5/lib/ -lcudart -lcublas
    LIBS += $$PWD/../../Libraries/libCUDAdillo.a
}
```

- And in the files using CUDAdillo,
```cpp
#include "armadillo.h"
#include "cudadillo.h"
```

#Examples
```cpp
mat matA = randu<mat>(5,5);
mat matB = randu<mat>(5,5);

CUDAdillo::init();

mat * sum = CUDAdillo::addMat<double>(&matA, &matB); //equivalent to matA + matB
mat * mult = CUDAdillo::multMat<double>(&matA, &matB); //equivalent to matA * matB
mat * trans = CUDAdillo::transposeMat<double>(&matA); // equivalent to matA.t()
mat * cov = CUDAdillo::covMat<double>(&matA,&matB); //equivalent to matA * matB.t()
//the above work for fmats and floats in a similar fashion

CUDAdillo::destroy();
delete sum;
delete mult;
delete trans;
delete cov;
```

Note that these functions allocate heap memory to create these new matrices, so they **must be freed by the user**, and the functions assume that the inputs share the same dimensions.

#References
[cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/index.html)

Specific Functions to note are:

- [cublasXtDgemm()](http://docs.nvidia.com/cuda/cublas/index.html#cublasxt_gemm)
- [cublasDgeam()](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam)

The Test subdirectory makes use of [google-benchmark](https://github.com/google/benchmark) to benchmark running time of these functions and the default CPU operations.

#Future Work
- Transpose currently fails on non-square inputs, and covMat fails to get any correct output.

- Matrix multiply is still slower than CPU version alone. Is memory bandwidth an issue?
-- Some links worth looking at:
-- <http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/fatahalian04_gpumatmat.pdf>
-- <http://stackoverflow.com/questions/2952277/when-is-a-program-limited-by-the-memory-bandwidth>
-- <http://pertsserver.cs.uiuc.edu/~mcaccamo/papers/private/IEEE_TC_journal_submitted_C.pdf>

- Implement a function to multiply a matrix by its transpose
