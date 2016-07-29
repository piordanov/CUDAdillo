#ifndef ARMACUDAWRAPPER_H
#define ARMACUDAWRAPPER_H

#include <armadillo>
using namespace arma;

template <typename T>
T * addGPU(T *, T *, int, int);

class ArmaCudaWrapper
{
public:
    template<typename T>
    static Mat<T> * addMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = addGPU<T>(A->memptr(),B->memptr(),A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr,A->n_rows,A->n_cols, true);
        return result;
    }


private:
    ArmaCudaWrapper(){}
};

#endif // ARMACUDAWRAPPER_H