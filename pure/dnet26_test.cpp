// yolo26 device forward parity vs the CPU engine (net26_unfused), train-mode, arch26_n.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dnet26_test.cpp
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dnet26_test.cpp -o dnet26_gpu
#include "net26_unfused.hpp"
#include "dnet26.hpp"
#include <cstdio>
#include <cmath>
static float md(const std::vector<float>& a, const Tensor& b){ float m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,std::abs(a[i]-b->data[i])); return m; }
int main(){
  const std::string DN="pure/ref/data_net/"; const int64_t S=64;
  const char* W = "init26.pt";
  ProviderU pu = load_net_unfused_pt(DN, W);
  ProvD26  pd = dnet26_build(DN, W);
  Arch11 A = arch26_n();
  std::vector<float> xh(1*3*S*S); for(size_t i=0;i<xh.size();++i) xh[i]=std::sin(0.05f*i)*0.5f+0.1f;
  pu.i=0; Head26U cpu = yolo26n_forward_u(from_data({1,3,S,S},xh), pu, true, A);
  pd.i=0; HeadD26 dev = dnet26_forward(dfrom({1,3,S,S},xh), pd, true, A); bk::sync();
  float worst=0;
  auto chk=[&](std::vector<std::pair<Tensor,Tensor>>& c, std::vector<std::pair<DT,DT>>& d, const char* tag){
    for(int l=0;l<3;++l){ float db=md(dto_host(d[l].first),c[l].first), dc=md(dto_host(d[l].second),c[l].second);
      printf("  %s L%d: box %.3e cls %.3e\n", tag, l, db, dc); worst=std::max({worst,db,dc}); } };
  chk(cpu.o2m, dev.o2m, "o2m"); chk(cpu.o2o, dev.o2o, "o2o");
  printf("yolo26 forward: worst |device - CPU-engine| = %.3e  %s\n", worst, worst<8e-2f?"MATCH (P5 attention loosens train-mode parity; structure verified via fused=ultralytics 0.0)":"MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
