// yolo26n UNFUSED (conv+BN separate) eval-forward parity vs the fused Ultralytics reference.
// Eval-mode BN uses running stats, so unfused == fused; must reproduce ref_*.bin.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\m6_unfused26.cpp
#include "net26_unfused.hpp"
#include <cstdio>
#include <cmath>
#include <string>

int main() {
  const std::string D = "pure/ref/data_net/";
  int64_t S = 64;
  ProviderU p = load_net_unfused(D);
  Tensor x = from_data({1, 3, S, S}, rd(D + "input.bin"));
  Head26U h = yolo26n_forward_u(x, p, /*train=*/false);
  printf("consumed %zu/%zu layers\n", p.i, p.layers.size());
  double worst = 0;
  auto cmp = [&](const char* tag, int i, const Tensor& t, const char* which) {
    auto ref = rd(D + "ref_" + tag + "_" + which + std::to_string(i) + ".bin");
    double d = 0; size_t n = std::min(ref.size(), t->data.size());
    for (size_t k = 0; k < n; ++k) d = std::max(d, (double)std::fabs(t->data[k] - ref[k]));
    worst = std::max(worst, d);
  };
  for (int i = 0; i < 3; ++i) { cmp("o2m", i, h.o2m[i].first, "box"); cmp("o2m", i, h.o2m[i].second, "cls"); }
  for (int i = 0; i < 3; ++i) { cmp("o2o", i, h.o2o[i].first, "box"); cmp("o2o", i, h.o2o[i].second, "cls"); }
  printf("yolo26n unfused(eval) forward: worst |pure - ultralytics| = %.3e   %s\n",
         worst, worst < 1e-3 ? "MATCH" : "MISMATCH");
  return worst < 1e-3 ? 0 : 1;
}
