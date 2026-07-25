// yolo26_cpp: export yolo26 to a standard .onnx (opset 13, no deps) from the FUSED data_net.
// Graph build lives in onnx_build26.hpp (shared with the `yolo26 export` CLI subcommand).
//   run: onnx_export26 [imgsz] [arch_dir]   (needs data_net/{manifest,weights,arch26,head26})
#include "onnx_build26.hpp"
#include <fstream>

int main(int argc, char** argv) {
  int64_t IMG = argc > 1 ? atoll(argv[1]) : 640;
  std::string D = argc > 2 ? argv[2] : "pure/ref/data_net/"; if (D.back() != '/') D += '/';
  auto prov = load_net(D);
  int64_t psa_n; std::vector<int64_t> cn, ci; std::vector<int> cc;
  { std::ifstream f(D + "arch26.txt"); f >> psa_n; int64_t a,b,c; while (f >> a >> b >> c) { cn.push_back(a); cc.push_back((int)b); ci.push_back(c); } }
  int64_t nc = 80; { std::ifstream f(D + "head26.txt"); if (f) f >> nc; }
  Graph g = build_yolo26_onnx(prov, psa_n, cn, cc, ci, nc, IMG);
  save_onnx(g, "yolo26n.onnx");
  printf("wrote yolo26n.onnx (%zu nodes, %zu f-init, %zu i-init, consumed %zu/%zu convs, imgsz=%lld)\n",
         g.nodes.size(), g.init_f.size(), g.init_i.size(), prov.i, prov.convs.size(), (long long)IMG);
  return 0;
}
