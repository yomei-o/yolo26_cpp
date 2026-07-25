// Micro-bench: bk::gemm at a representative conv size (im2col GEMM), baseline loops vs -DUSE_EIGEN.
//   baseline: cc.sh -std:c++17 -O2 -EHsc bench_gemm.cpp
//   eigen   : cc.sh -std:c++17 -O2 -EHsc -DUSE_EIGEN /arch:AVX2 -Ipure/third_party/eigen_flat bench_gemm.cpp
#include "backend.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>

int main() {
  // a mid-network conv as im2col GEMM: Cout=256, K=Cin*k*k=256*9, N=H*W=40*40
  int64_t M = 256, K = 256*9, N = 40*40;
  std::vector<float> A((size_t)M*K), B((size_t)K*N), C((size_t)M*N);
  std::mt19937 rng(0); std::normal_distribution<float> nd(0,0.1f);
  for (auto& v : A) v = nd(rng); for (auto& v : B) v = nd(rng);
  bk::gemm(A.data(), B.data(), C.data(), M, K, N);         // warmup
  int reps = 30;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) bk::gemm(A.data(), B.data(), C.data(), M, K, N);
  double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/reps;
  double gflops = 2.0*M*K*N/1e9 / (ms/1e3);
  printf("gemm %lldx%lldx%lld : %.3f ms/call  %.1f GFLOP/s  [%s]\n",
         (long long)M,(long long)K,(long long)N, ms, gflops,
#ifdef USE_EIGEN
         "EIGEN");
#else
         "baseline loops");
#endif
  return 0;
}
