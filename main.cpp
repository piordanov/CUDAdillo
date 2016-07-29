#include <armadillo>
#include <benchmark/benchmark_api.h>
#include <chrono>
#include "armacudawrapper.h"

using namespace arma;

#define BASIC_BENCHMARK_TEST(x) \
    BENCHMARK(x)->Arg(8)->Unit(benchmark::kMillisecond)

static void BM_matrixAdditionCPU(benchmark::State& state) {
    int n = int(state.range_x());
    fmat matA(n,n);
    matA.fill(5);
    fmat matB(n,n);
    matB.fill(7);
    while (state.KeepRunning())
        fmat temp = matA + matB;
}

BASIC_BENCHMARK_TEST(BM_matrixAdditionCPU);

static void BM_matrixAdditionGPU(benchmark::State& state) {
    int n = int(state.range_x());
    fmat matA(n,n);
    matA.fill(5);
    fmat matB(n,n);
    matB.fill(7);
    while (state.KeepRunning())
        ArmaCudaWrapper::addMat<float>(&matA,&matB);
}
BASIC_BENCHMARK_TEST(BM_matrixAdditionGPU);

static void BM_matrixMultiplyCPU(benchmark::State& state) {
    int n = int(state.range_x());
    fmat matA(n,n);
    matA.fill(5);
    fmat matB(n,n);
    matB.fill(7);
    while (state.KeepRunning())
        fmat temp = matA * matB;
}
BASIC_BENCHMARK_TEST(BM_matrixMultiplyCPU);


//static void BM_matrixMultiplyGPU(benchmark::State& state) {
//    int n = int(state.range_x());
//    fmat matA(n,n);
//    matA.fill(5);
//    fmat matB(n,n);
//    matB.fill(7);
//    while (state.KeepRunning())
//        multiply(matA.memptr(),matB.memptr(),n,n);
//}
//BASIC_BENCHMARK_TEST(BM_matrixMultiplyGPU);

void test_addition()
{
    mat matA = randu<mat>(4,4);
    mat matB = randu<mat>(4,4);

    mat cpuSum = matA + matB;
    mat * gpuSum = ArmaCudaWrapper::addMat<double>(&matA,&matB);
    bool same = approx_equal(cpuSum, *gpuSum,"absdiff", 0.01);
    delete gpuSum;
    printf("approx-equal: %s\n", same ? "true" : "false");
    printf("A:\n");
    matA.print();
    printf("\nB:\n");
    matB.print();

    printf("\nCPU Output:\n");
    cpuSum.print();
    printf("\nGPU Output:\n");
    gpuSum->print();
}


int main(int argc, char *argv[])
{
//    std::ofstream benchmarks("benchmark.json");
//    std::streambuf *coutbuf = std::cout.rdbuf();
//    std::cout.rdbuf(benchmarks.rdbuf());

    test_addition();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

//    std::cout.rdbuf(coutbuf);
}
