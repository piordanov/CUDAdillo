/*
    This file is part of CUDAdillo.

    CUDAdillo is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAdillo is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CUDAdillo.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CUDADILLO_H
#define CUDADILLO_H

#include <armadillo>
using namespace arma;

template <typename T>
T * addGPU(T *, T *, int, int);
template <typename T>
T * multGPUcuBLAS(T *, T *, int, int);
template <typename T>
T * transposeGPUcuBLAS(T *, int, int);
template<typename T>
T * covGPUcuBLAS(T *, T *, int, int);
void cublasinit();
void  cublasdestroy();

class CUDAdillo
{
public:
    static void init()
    {
        cublasinit();
    }
    static void destroy()
    {
        cublasdestroy();
    }

    template<typename T>
    static Mat<T> * addMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = addGPU<T>(A->memptr(),B->memptr(),A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr,A->n_rows,A->n_cols, true);
        free(arr);
        return result;
    }

    template<typename T>
    static Mat<T> * multMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = multGPUcuBLAS<T>(A->memptr(),B->memptr(),A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr,A->n_rows,A->n_cols, true);
        free(arr);
        return result;
    }

    template<typename T>
    static Mat<T> * transposeMat(Mat<T> * M)
    {
        T * arr = transposeGPUcuBLAS<T>(M->memptr(),M->n_rows, M->n_cols);
        Mat<T> * result = new Mat<T>(arr,M->n_rows,M->n_cols, true);
        free(arr);
        return result;
    }

    template<typename T>
    static Mat<T> * covMat(Mat<T> * A, Mat<T> * B)
    {
        T * arr = covGPUcuBLAS<T>(A->memptr(), B->memptr(), A->n_rows,A->n_cols);
        Mat<T> * result = new Mat<T>(arr, A->n_rows, A->n_cols, true);
        free(arr);
        return result;
    }

private:
    CUDAdillo(){}
};

#endif // CUDADILLO_H
