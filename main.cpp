#include <armadillo>
#include <benchmark/benchmark_api.h>
#include <chrono>
#include "armacudawrapper.h"

using namespace arma;

#define BASIC_BENCHMARK_TEST(x) \
    BENCHMARK(x)->Arg(8)->Arg(512)->Arg(8192)->Unit(benchmark::kMillisecond)

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



int main(int argc, char *argv[])
{
//    std::ofstream benchmarks("benchmark.json");
//    std::streambuf *coutbuf = std::cout.rdbuf();
//    std::cout.rdbuf(benchmarks.rdbuf());

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

//    std::cout.rdbuf(coutbuf);
}
