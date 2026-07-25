// yolo26 CPU inference. Forward (one2many head) -> direct box decode (reg_max=1, no DFL) ->
// NMS -> draw boxes. (Faithful NMS-free o2o decode comes once the o2o branch is trained.)
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\detect26.cpp
//   run:   detect26 <weights.pt> <img.jpg> [imgsz=64] [conf=0.25] [out.png] [arch_dir]
#define STB_IMAGE_IMPLEMENTATION        // dataset.hpp includes stb_image.h -> impl compiles here (once)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"            // write header (not pulled in by dataset.hpp)
#include "net26_unfused.hpp"
#include "infer.hpp"          // make_anchors, nms, Det
#include "dataset.hpp"        // load_image_letterbox, Letterbox
#include <cstdio>
#include <cmath>
#include <algorithm>

static const int64_t NC = 80;
static const char* COCO[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

// direct decode: box[a*4..] are ltrb distances (grid units) -> xyxy (image units); cls sigmoid.
static void decode26(const std::vector<float>& box, const std::vector<float>& cls,
                     const std::vector<float>& ax, const std::vector<float>& ay,
                     const std::vector<float>& st, int64_t A, std::vector<float>& out) {
  out.assign((4 + NC) * A, 0.f);
  for (int64_t a = 0; a < A; ++a) {
    float l = box[a*4+0], t = box[a*4+1], r = box[a*4+2], b = box[a*4+3], sc = st[a];
    out[0*A+a] = (ax[a]-l)*sc; out[1*A+a] = (ay[a]-t)*sc; out[2*A+a] = (ax[a]+r)*sc; out[3*A+a] = (ay[a]+b)*sc;
    for (int64_t c = 0; c < NC; ++c) out[(4+c)*A+a] = 1.f/(1.f+std::exp(-cls[a*NC+c]));
  }
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: detect26 <weights.pt> <img.jpg> [imgsz] [conf] [out.png] [arch_dir]\n"); return 1; }
  std::string weights = argv[1], src = argv[2];
  int64_t S = argc>3?atoll(argv[3]):64; float conf = argc>4?(float)atof(argv[4]):0.25f;
  std::string outp = argc>5?argv[5]:"det26.png";
  std::string DN = argc>6?argv[6]:"pure/ref/data_net/"; if (!DN.empty() && DN.back()!='/') DN += '/';

  ProviderU prov = load_net_unfused_pt(DN, weights);
  int w0,h0,ch; unsigned char* im = stbi_load(src.c_str(), &w0,&h0,&ch,3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }
  Letterbox lb; auto x = load_image_letterbox(src, S, lb);
  std::vector<float> ax,ay,st; make_anchors(S, ax, ay, st); int64_t A = (int64_t)st.size();

  prov.i = 0; Head26U h = yolo26n_forward_u(x, prov, false, arch26_n());
  std::vector<Tensor> bx={h.o2m[0].first,h.o2m[1].first,h.o2m[2].first}, cs={h.o2m[0].second,h.o2m[1].second,h.o2m[2].second};
  auto pd = pack_levels(bx, 1, A, 4); auto ps = pack_levels(cs, 1, A, NC);
  std::vector<float> pred; decode26(pd->data, ps->data, ax, ay, st, A, pred);
  auto dets = nms(pred, A, NC, conf, 0.7f, 300);

  auto put=[&](int px,int py,unsigned char r,unsigned char g,unsigned char b){ if(px<0||py<0||px>=w0||py>=h0)return; unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b; };
  printf("%zu detections:\n", dets.size());
  for (auto& d : dets) {
    int x1=(int)std::round((d.x1-lb.left)/lb.r), y1=(int)std::round((d.y1-lb.top)/lb.r);
    int x2=(int)std::round((d.x2-lb.left)/lb.r), y2=(int)std::round((d.y2-lb.top)/lb.r);
    x1=std::clamp(x1,0,w0-1);y1=std::clamp(y1,0,h0-1);x2=std::clamp(x2,0,w0-1);y2=std::clamp(y2,0,h0-1);
    for(int t=0;t<3;++t){for(int px=x1;px<=x2;++px){put(px,y1+t,255,60,60);put(px,y2-t,255,60,60);}for(int py=y1;py<=y2;++py){put(x1+t,py,255,60,60);put(x2-t,py,255,60,60);}}
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", d.cls<80?COCO[d.cls]:"?", d.conf, x1,y1,x2,y2);
  }
  stbi_write_png(outp.c_str(), w0,h0,3, im, w0*3);
  printf("wrote %s\n", outp.c_str()); stbi_image_free(im); return 0;
}
