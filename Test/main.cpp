#include <armadillo>
#include <benchmark/benchmark_api.h>
#include <chrono>
#include "json.hpp"
#include <sstream>
#include <limits>
#include "cudadillo.h"

using namespace arma;
using json = nlohmann::json;

#define BASIC_BENCHMARK_TEST(x) \
    BENCHMARK(x)->Arg(8<<10)->Unit(benchmark::kNanosecond)

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
        CUDAdillo::addMat<float>(&matA,&matB);
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

static void BM_matrixMultiplyGPU(benchmark::State & state) {
    int n = int(state.range_x());
    fmat matA(n,n);
    matA.fill(5);
    fmat matB(n,n);
    matB.fill(7);
    while (state.KeepRunning())
    {
        CUDAdillo::multMat<float>(&matA,&matB);
    }
}

BASIC_BENCHMARK_TEST(BM_matrixMultiplyGPU);

template <typename T>
void setupMat(Mat<T> * A, int size, int offset)
{
    for(int i = 0; i < size; i++)
        A->at(i) = i + offset;
}

void test_addition()
{
    mat matA = randu<mat>(5,5);
    mat matB = randu<mat>(5,5);

    mat cpuSum = matA + matB;
    mat * gpuSum = CUDAdillo::addMat<double>(&matA,&matB);
    bool same = approx_equal(cpuSum, *gpuSum,"absdiff", 0.01);
    printf("Addition approx-equal: %s\n", same ? "success" : "failure");
    if(!same){
        printf("A:\n");
        matA.print();
        printf("\nB:\n");
        matB.print();

        printf("\nCPU Output:\n");
        cpuSum.print();
        printf("\nGPU Output:\n");
        gpuSum->print();
    }
    delete gpuSum;
}

void test_multiply()
{
    mat matA = randu<mat>(5,5);
    mat matB = randu<mat>(5,5);

    mat cpuMult = matA * matB;
    mat * gpuMult = CUDAdillo::multMat<double>(&matA,&matB);
    bool same = approx_equal(cpuMult, *gpuMult,"absdiff", 0.01);
    printf("Multiplication approx-equal: %s\n", same ? "success" : "failure");
    if(!same){
        printf("A:\n");
        matA.print();
        printf("\nB:\n");
        matB.print();

        printf("\nCPU Output:\n");
        cpuMult.print();
        printf("\nGPU Output:\n");
        gpuMult->print();
    }
    delete gpuMult;
}
bool contains(std::string s, std::string c){
    return s.find(c) != std::string::npos;
}

int main(int argc, char *argv[])
{
    std::stringstream benchmarks;
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(benchmarks.rdbuf());

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    std::cout.rdbuf(coutbuf);

    test_addition();
    test_multiply();

    std::cout << benchmarks.str();

//    auto j =  json::parse(benchmarks.str());
//    auto benchmks = j["benchmarks"];

//    std::string bestAdd, bestMul;
//    long long mintimeAdd = std::numeric_limits<int>::max();
//    long long mintimeMul = std::numeric_limits<int>::max();

//    for (json::iterator it = benchmks.begin(); it != benchmks.end(); ++it) {
//        long long time = (*it)["real_time"];
//        std::string name = (*it)["name"];
//        if(contains(name,"Addition") && (time < mintimeAdd)) {
//            bestAdd = name;
//            mintimeAdd = time;
//        }
//        if(contains(name,"Multiply") && (time < mintimeMul)) {
//            bestMul = name;
//            mintimeMul = time;
//        }

//    }
//    std::cout << bestAdd << " had best time of " << mintimeAdd / 1e+9 << "secs\n";
//    std::cout << bestMul << " had best time of " << mintimeMul / 1e+9 << "secs\n";
//    std::cout << "Benchmarks output:\n\n";
//    std::cout << benchmks.dump(4) << std::endl;


}
