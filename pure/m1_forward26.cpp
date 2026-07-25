// yolo26n fused-forward parity vs the Ultralytics reference (pure/ref/export_yolo26.py output).
// Compares both head branches (one2many + one2one), box(4) and cls(nc), per level.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\m1_forward26.cpp   (or g++)
//   run:   m1_forward26     (expects pure/ref/data_net/ from export_yolo26.py)
#include "net26.hpp"
#include <cstdio>
#include <cmath>
#include <string>

int main() {
  const std::string D = "pure/ref/data_net/";
  int64_t S = 64;                                    // export_yolo26.py default imgsz
  Provider p = load_net(D);
  printf("loaded %zu convs\n", p.convs.size());
  Tensor x = from_data({1, 3, S, S}, rd(D + "input.bin"));
  Head26 h = yolo26n_forward(x, p);
  printf("consumed %zu/%zu convs\n", p.i, p.convs.size());

  double worst = 0;
  auto cmp = [&](const char* tag, int i, const Tensor& t, const char* which) {
    auto ref = rd(D + "ref_" + tag + "_" + which + std::to_string(i) + ".bin");
    double d = 0; size_t n = std::min(ref.size(), t->data.size());
    for (size_t k = 0; k < n; ++k) d = std::max(d, (double)std::fabs(t->data[k] - ref[k]));
    worst = std::max(worst, d);
    printf("  %-3s L%d %-3s: out=%lldx%lldx%lldx%lld ref=%zu  max|d|=%.3e%s\n",
           tag, i, which, (long long)t->shape[0], (long long)t->shape[1], (long long)t->shape[2],
           (long long)t->shape[3], ref.size(), d, ref.size()==t->data.size()?"":"  (SIZE MISMATCH!)");
  };
  for (int i = 0; i < 3; ++i) { cmp("o2m", i, h.o2m[i].first, "box"); cmp("o2m", i, h.o2m[i].second, "cls"); }
  for (int i = 0; i < 3; ++i) { cmp("o2o", i, h.o2o[i].first, "box"); cmp("o2o", i, h.o2o[i].second, "cls"); }
  printf("yolo26n fused forward: worst |pure - ultralytics| = %.3e   %s\n",
         worst, worst < 1e-3 ? "MATCH" : "MISMATCH");
  return worst < 1e-3 ? 0 : 1;
}
