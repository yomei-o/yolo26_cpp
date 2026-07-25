// End-to-end self-test: train yolo26 from random init on a tiny synthetic 3-colour dataset
// (dual head, no-DFL loss), then run NMS-free inference on a held-out image and draw boxes.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\m9_synth26.cpp
//   run:   m9_synth26 [iters]   (needs make_synth.py output + export_unfused26.py manifest)
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net26_unfused.hpp"
#include "v26loss.hpp"
#include "optim.hpp"
#include "infer.hpp"
#include <cstdio>
#include <string>

static const char* NAMES[3] = {"red","green","blue"};
static const int64_t NC = 80;

int main(int argc, char** argv) {
  int ITERS = argc>1?atoi(argv[1]):200;
  const std::string DN="pure/ref/data_net/", DS="pure/ref/data_synth/";
  Batch bt = load_batch(DS+"list.txt"); int64_t B=bt.B, M=bt.M, S=bt.x->shape[2];
  printf("synth train: %lld imgs %lldpx, up to %lld obj\n",(long long)B,(long long)S,(long long)M);

  ProviderU prov; { std::ifstream t("init26.pt"); if(t.good()) prov=load_net_unfused_pt(DN,"init26.pt"); else prov=load_net_unfused(DN); }
  Arch11 ARC=arch26_n();
  std::vector<Tensor> params; for(auto&L:prov.layers){params.push_back(L.w); if(L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b);}
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);
  struct Lv{int64_t h,w;float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A=(int64_t)ss.size(), R=B*A;
  std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}

  printf("training %d iters...\n", ITERS);
  for(int it=0; it<ITERS; ++it){
    prov.i=0; Head26U h=yolo26n_forward_u(bt.x, prov, true, ARC);
    auto branch=[&](std::vector<std::pair<Tensor,Tensor>>&hd,int topk){ std::vector<Tensor> bx={hd[0].first,hd[1].first,hd[2].first},cs={hd[0].second,hd[1].second,hd[2].second};
      auto pd=pack_levels(bx,B,A,4); auto ps=pack_levels(cs,B,A,NC); std::vector<float> pdb,pss; decode_for_tal26(pd,ps,ax,ay,ss,R,A,NC,pdb,pss);
      auto tal=tal_assign(pss,pdb,anc_img,bt.gt_labels,bt.gt_boxes,bt.mask,B,A,M,NC,topk,0.5f,6.0f); return pure_v26_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC); };
    auto Lm=branch(h.o2m,10),Lo=branch(h.o2o,1); auto tot=add(Lm.total,Lo.total);
    backward(tot); opt.lr=cosine_lr(it,ITERS,2e-3f,5); opt.step();
    if(it%25==0||it==ITERS-1) printf("  iter %3d  loss %7.3f\n", it, tot->data[0]);
    free_graph(tot);
  }
  printf("training done.\n");

  // infer (NMS-free o2o) on a held-out val image
  std::string timg=DS+"va00.png"; int w0,h0,ch; unsigned char* im=stbi_load(timg.c_str(),&w0,&h0,&ch,3);
  auto x=make_tensor({1,3,S,S}); for(int c=0;c<3;++c)for(int y=0;y<S;++y)for(int xx=0;xx<S;++xx)x->data[(c*S+y)*S+xx]=im[(y*w0+xx)*3+c]/255.f;
  prov.i=0; Head26U h=yolo26n_forward_u(x,prov,false,ARC);
  std::vector<Tensor> bx={h.o2o[0].first,h.o2o[1].first,h.o2o[2].first},cs={h.o2o[0].second,h.o2o[1].second,h.o2o[2].second};
  auto pd=pack_levels(bx,1,A,4); auto ps=pack_levels(cs,1,A,NC);
  std::vector<Det> dets; for(int64_t a=0;a<A;++a){float best=-1;int bc=0;for(int64_t c=0;c<NC;++c){float s=1.f/(1.f+std::exp(-ps->data[a*NC+c]));if(s>best){best=s;bc=(int)c;}}
    if(best<0.25f)continue; float l=pd->data[a*4],t=pd->data[a*4+1],r=pd->data[a*4+2],b=pd->data[a*4+3];
    dets.push_back({(ax[a]-l)*ss[a],(ay[a]-t)*ss[a],(ax[a]+r)*ss[a],(ay[a]+b)*ss[a],best,bc}); }
  std::sort(dets.begin(),dets.end(),[](const Det&p,const Det&q){return p.conf>q.conf;}); if((int)dets.size()>50)dets.resize(50);
  printf("inference on va00.png: %zu detections\n", dets.size());
  for(auto&d:dets) printf("  cls %d(%s) conf %.2f xyxy=(%.0f,%.0f,%.0f,%.0f)\n", d.cls, d.cls<3?NAMES[d.cls]:"?", d.conf, d.x1,d.y1,d.x2,d.y2);
  stbi_write_png("synth26_det.png", w0,h0,3, im, w0*3); stbi_image_free(im);
  return 0;
}
