#ifndef ARMUDA_H
#define ARMUDA_H

#include <armadillo>
using namespace arma;

template <typename T>
T * addGPU(T *, T *, int, int);
template <typename T>
T * multGPUcuBLAS(T *, T *, int, int);

class CUDAdillo
{
public:
    template<typename T>
    static Mat<T> * addMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = addGPU<T>(A->memptr(),B->memptr(),A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr,A->n_rows,A->n_cols, true);
        return result;
    }

    template<typename T>
    static Mat<T> * multMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = multGPUcuBLAS<T>(A->memptr(),B->memptr(),A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr,A->n_rows,A->n_cols, true);
        return result;
    }

private:
    CUDAdillo(){}
};

#endif // ARMUDA_H
