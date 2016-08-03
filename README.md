## CUDAdillo

A static c++ library to complete basic matrix operations on Armadillo matrices with the intent to speed up runtime through the use of the CUDA and cuBLAS GPU libraries.

###Qt Creator Integration

- Have the `cudadillo.h` file in the same working directory
- Have the compiled `libCUDAdillo.a` in the project
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

###Examples
```cpp
mat matA = randu<mat>(5,5);
mat matB = randu<mat>(5,5);

mat * sum = CUDAdillo::addMat<double>(&matA, &matB);
mat * mult = CUDAdillo::multMat<double>(&matA, &matB);

delete sum;
delete mult;
```

Note that these functions allocate heap memory to create these new matrices, so they **must be freed by the user**.

