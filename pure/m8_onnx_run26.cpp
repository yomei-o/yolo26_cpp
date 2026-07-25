// Run the exported yolo26n.onnx in PURE C++ (onnx_run.hpp, no onnxruntime) and check it
// reproduces the Ultralytics reference head outputs (o2o branch). Proves both the export AND
// our own ONNX reader/runner are correct.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\m8_onnx_run26.cpp
//   run:   m8_onnx_run26   (needs yolo26n.onnx at imgsz 64 + data_net refs from export_yolo26.py)
#include "onnx_run.hpp"
#include "net26.hpp"
#include <cstdio>
#include <cmath>
#include <string>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto g = onx::load_onnx("yolo26n.onnx");
  auto x = from_data({1, 3, 64, 64}, rd(D + "input.bin"));
  printf("%zu nodes\n", g.nodes.size());
  auto vals = onx::run_onnx(g, x);
  double worst = 0;
  for (const char* tag : {"o2o", "o2m"})
    for (int i = 0; i < 3; ++i)
      for (const char* which : {"box", "cls"}) {
        auto t = vals.at(std::string(tag) + "_" + which + std::to_string(i));
        auto ref = rd(D + "ref_" + tag + "_" + which + std::to_string(i) + ".bin");
        double d = 0; size_t n = std::min(ref.size(), t->data.size());
        for (size_t k = 0; k < n; ++k) d = std::max(d, (double)std::fabs(t->data[k] - ref[k]));
        worst = std::max(worst, d);
      }
  printf("pure-C++ ONNX run vs ultralytics ref: worst |d| = %.3e   %s\n",
         worst, worst < 1e-3 ? "OK" : "FAIL");
  return worst < 1e-3 ? 0 : 1;
}
