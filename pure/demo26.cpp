// Self-contained yolo26 demo: loads the shipped fused weights (weights/yolo26n/) and detects a
// bundled image (assets/bus.jpg) with the NMS-free (one2one) head — runs straight from a checkout,
// no Python, no weights download.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\demo26.cpp
//   run:   demo26 [img.jpg] [weights_dir] [imgsz] [conf] [out.png]
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net26.hpp"          // load_net_blob (fused), yolo26n_forward, pack_levels
#include "infer.hpp"          // make_anchors, Det
#include "dataset.hpp"        // load_image_letterbox
#include <cstdio>
#include <cmath>
#include <algorithm>

static const int64_t NC = 80;
static const char* COCO[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int main(int argc, char** argv) {
  std::string src = argc>1?argv[1]:"assets/bus.jpg";
  std::string WD  = argc>2?argv[2]:"weights/yolo26n/"; if (WD.back()!='/') WD += '/';
  int64_t S = argc>3?atoll(argv[3]):640; float conf = argc>4?(float)atof(argv[4]):0.25f;
  std::string outp = argc>5?argv[5]:"demo26_out.png";

  Provider prov = load_net_blob(WD);
  Arch11 A; { std::ifstream f(WD+"arch26.txt"); f>>A.psa_n; int64_t n,c,i; while(f>>n>>c>>i) A.c3.push_back({n,i,(bool)c}); }
  int w0,h0,ch; unsigned char* im = stbi_load(src.c_str(), &w0,&h0,&ch,3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }
  Letterbox lb; auto x = load_image_letterbox(src, S, lb);
  std::vector<float> ax,ay,st; make_anchors(S, ax, ay, st); int64_t An=(int64_t)st.size();

  prov.i = 0; Head26 h = yolo26n_forward(x, prov, A);       // fused forward; use one2one (NMS-free)
  std::vector<Tensor> bx={h.o2o[0].first,h.o2o[1].first,h.o2o[2].first}, cs={h.o2o[0].second,h.o2o[1].second,h.o2o[2].second};
  auto pd=pack_levels(bx,1,An,4); auto ps=pack_levels(cs,1,An,NC);
  std::vector<Det> dets;                                    // decode + NMS-free top-k
  for (int64_t a=0;a<An;++a){ float best=-1;int bc=0; for(int64_t c=0;c<NC;++c){float s=1.f/(1.f+std::exp(-ps->data[a*NC+c])); if(s>best){best=s;bc=(int)c;}}
    if(best<conf) continue; float l=pd->data[a*4],t=pd->data[a*4+1],r=pd->data[a*4+2],b=pd->data[a*4+3],s=st[a];
    dets.push_back({(ax[a]-l)*s,(ay[a]-t)*s,(ax[a]+r)*s,(ay[a]+b)*s,best,bc}); }
  std::sort(dets.begin(),dets.end(),[](const Det&p,const Det&q){return p.conf>q.conf;});
  if((int)dets.size()>300) dets.resize(300);

  auto put=[&](int px,int py,unsigned char r,unsigned char g,unsigned char b){if(px<0||py<0||px>=w0||py>=h0)return;unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b;};
  printf("yolo26n demo — %s (%dx%d), %zu detections:\n", src.c_str(), w0, h0, dets.size());
  for (auto& d : dets) {
    int x1=(int)std::round((d.x1-lb.left)/lb.r),y1=(int)std::round((d.y1-lb.top)/lb.r),x2=(int)std::round((d.x2-lb.left)/lb.r),y2=(int)std::round((d.y2-lb.top)/lb.r);
    x1=std::clamp(x1,0,w0-1);y1=std::clamp(y1,0,h0-1);x2=std::clamp(x2,0,w0-1);y2=std::clamp(y2,0,h0-1);
    for(int t=0;t<3;++t){for(int px=x1;px<=x2;++px){put(px,y1+t,255,60,60);put(px,y2-t,255,60,60);}for(int py=y1;py<=y2;++py){put(x1+t,py,255,60,60);put(x2-t,py,255,60,60);}}
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", d.cls<80?COCO[d.cls]:"?", d.conf, x1,y1,x2,y2);
  }
  stbi_write_png(outp.c_str(), w0,h0,3, im, w0*3);
  printf("wrote %s\n", outp.c_str()); stbi_image_free(im); return 0;
}
