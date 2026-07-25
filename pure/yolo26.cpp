// Unified pure-C++ yolo26 CLI: `yolo26 <train|val|detect> [--flags]`, reading a standard
// Ultralytics data.yaml. No Python at run time. yolo26 head: reg_max=1 (no DFL) + end2end —
// training uses both branches (one2many topk=10 + one2one topk=1); val/detect are NMS-free (o2o).
//   train  --data d.yaml --weights init26.pt [--epochs 100 --batch 16 --imgsz 64 --mosaic 1 ...]
//   val    --data d.yaml --weights best.pt   [--imgsz 64]
//   detect --weights best.pt --source img.jpg [--imgsz 64 --conf 0.25 --out out.png]
// build: cc.sh -std:c++20 -O2 -EHsc -Ipure/third_party pure/yolo26.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net26_unfused.hpp"
#include "v26loss.hpp"
#include "optim.hpp"
#include "infer.hpp"          // make_anchors, nms, Det, load_boxes_orig, lb_map
#include "metrics.hpp"
#include "ptio.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>
#include <sstream>
#include <map>

static const int64_t NC0 = 80;
static std::string DN = "pure/ref/data_net/";
static const char* COCO[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

// ---- data.yaml + args ----
struct DataYaml { std::string path, train, val; int nc = 80; std::vector<std::string> names; };
static std::string trim(std::string s){ size_t a=s.find_first_not_of(" \t\r\n'\""), b=s.find_last_not_of(" \t\r\n'\""); return a==std::string::npos?std::string():s.substr(a,b-a+1); }
static DataYaml parse_yaml(const std::string& p){
  std::ifstream f(p); if(!f){ printf("cannot open %s\n",p.c_str()); std::exit(1);} DataYaml d; std::string line;
  while(std::getline(f,line)){ auto h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h); auto c=line.find(':'); if(c==std::string::npos) continue;
    std::string k=trim(line.substr(0,c)),v=trim(line.substr(c+1));
    if(k=="path")d.path=v; else if(k=="train")d.train=v; else if(k=="val")d.val=v; else if(k=="nc")d.nc=atoi(v.c_str());
    else if(k=="names"){ auto lb=v.find('['),rb=v.rfind(']'); if(lb!=std::string::npos&&rb!=std::string::npos){ std::stringstream ss(v.substr(lb+1,rb-lb-1)); std::string t; while(std::getline(ss,t,',')){t=trim(t); if(!t.empty())d.names.push_back(t);} } } }
  return d;
}
static std::string join(const std::string& a, const std::string& b){ return a.empty()?b:a+"/"+b; }
struct Args{ std::map<std::string,std::string> m;
  std::string get(const std::string&k,const std::string&d="")const{auto it=m.find(k);return it==m.end()?d:it->second;}
  int geti(const std::string&k,int d)const{auto it=m.find(k);return it==m.end()?d:atoi(it->second.c_str());}
  float getf(const std::string&k,float d)const{auto it=m.find(k);return it==m.end()?d:(float)atof(it->second.c_str());} };
static Args parse_args(int argc,char**argv,int start){ Args a; for(int i=start;i<argc;++i){std::string s=argv[i]; if(s.rfind("--",0)!=0)continue; s=s.substr(2); auto eq=s.find('=');
  if(eq!=std::string::npos)a.m[s.substr(0,eq)]=s.substr(eq+1); else if(i+1<argc&&std::string(argv[i+1]).rfind("--",0)!=0)a.m[s]=argv[++i]; else a.m[s]="1"; } return a; }

static ProviderU load_model(const std::string& weights){ std::ifstream t(weights);
  if(t.good()){ printf("weights <- %s (pure C++)\n",weights.c_str()); return load_net_unfused_pt(DN,weights);} return load_net_unfused(DN); }
// arch26.txt (psa_n + 8x "n c3k inner") selects the size; falls back to yolo26n if absent.
static Arch11 load_arch26(){ std::ifstream f(DN+"arch26.txt"); if(!f) return arch26_n();
  Arch11 A; f>>A.psa_n; int64_t n,c,i; while(f>>n>>c>>i) A.c3.push_back({n,i,(bool)c});
  if(A.psa_n<1||A.c3.size()!=8){ printf("bad %sarch26.txt\n",DN.c_str()); std::exit(1);} return A; }

// direct decode (o2o head): box ltrb (grid) -> xyxy (image) into (4+nc)*A; cls sigmoid.
static void decode26(const std::vector<float>& box, const std::vector<float>& cls,
                     const std::vector<float>& ax, const std::vector<float>& ay, const std::vector<float>& st,
                     int64_t A, std::vector<float>& out){
  out.assign((4+NC0)*A,0.f);
  for(int64_t a=0;a<A;++a){ float l=box[a*4+0],t=box[a*4+1],r=box[a*4+2],b=box[a*4+3],s=st[a];
    out[0*A+a]=(ax[a]-l)*s; out[1*A+a]=(ay[a]-t)*s; out[2*A+a]=(ax[a]+r)*s; out[3*A+a]=(ay[a]+b)*s;
    for(int64_t c=0;c<NC0;++c) out[(4+c)*A+a]=1.f/(1.f+std::exp(-cls[a*NC0+c])); }
}
static std::vector<Det> nms_free(const std::vector<float>& pred,int64_t A,float conf,int max_det){
  std::vector<Det> cand;
  for(int64_t a=0;a<A;++a){ float best=-1.f; int bc=0; for(int64_t c=0;c<NC0;++c){float s=pred[(4+c)*A+a]; if(s>best){best=s;bc=(int)c;}}
    if(best<conf) continue; cand.push_back({pred[0*A+a],pred[1*A+a],pred[2*A+a],pred[3*A+a],best,bc}); }
  std::sort(cand.begin(),cand.end(),[](const Det&x,const Det&y){return x.conf>y.conf;});
  if((int)cand.size()>max_det) cand.resize(max_det); return cand;
}

// run val mAP (NMS-free o2o decode)
static double run_val(Dataset& va, ProviderU& prov, int64_t S){
  std::vector<float> ax,ay,st; make_anchors(S,ax,ay,st); int64_t A=(int64_t)st.size();
  std::vector<mapeval::Image> imgs;
  for(auto& s: va.items){ Letterbox lb; auto xi=load_image_letterbox(s.img,S,lb);
    prov.i=0; Head26U h=yolo26n_forward_u(xi,prov,false,load_arch26());
    std::vector<Tensor> bx={h.o2o[0].first,h.o2o[1].first,h.o2o[2].first}, cs={h.o2o[0].second,h.o2o[1].second,h.o2o[2].second};
    auto pd=pack_levels(bx,1,A,4); auto ps=pack_levels(cs,1,A,NC0);
    std::vector<float> pred; decode26(pd->data,ps->data,ax,ay,st,A,pred);
    auto dets=nms_free(pred,A,0.001f,300); mapeval::Image im;
    for(auto& d: dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.conf});
    std::vector<float> gb; std::vector<int64_t> gl; int m=load_boxes_orig(s.lbl,va.yolo,lb.w0,lb.h0,gb,gl); lb_map(gb,lb);
    for(int j=0;j<m;++j) im.gts.push_back({gb[j*4],gb[j*4+1],gb[j*4+2],gb[j*4+3],(int)gl[j]});
    imgs.push_back(im); free_graph(pd); free_graph(ps);
  }
  auto mp=mapeval::coco_map(imgs);
  printf("val: mAP@0.5 %.4f  mAP@0.5:0.95 %.4f  (%zu images)\n", mp.first, mp.second, va.items.size());
  return mp.first;
}

static int cmd_train(const Args& a){
  DataYaml dy=parse_yaml(a.get("data")); int64_t S=a.geti("imgsz",64);
  int EPOCHS=a.geti("epochs",100), BATCH=a.geti("batch",8); std::string weights=a.get("weights","init26.pt");
  AugCfg baseAug; baseAug.mosaic=a.geti("mosaic",1)!=0; baseAug.mixup=a.geti("mixup",1)!=0;
  int closeMosaic=a.geti("close-mosaic",std::max(1,EPOCHS/10)); float lr0=a.getf("lr",1e-3f);
  Dataset tr=read_yolo_dataset(join(dy.path,dy.train),S), va=read_yolo_dataset(join(dy.path,dy.val),S);
  printf("yolo26n train=%zu val=%zu imgsz=%lld batch=%d epochs=%d\n", tr.items.size(),va.items.size(),(long long)S,BATCH,EPOCHS);
  auto prov=load_model(weights); Arch11 ARC=load_arch26();
  std::vector<Tensor> params; for(auto&L:prov.layers){ params.push_back(L.w); if(L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b);}
  int warmup=a.geti("warmup",std::max(1,((int)tr.items.size()+BATCH-1)/BATCH)); Adam opt(params,lr0,0.9f,0.999f,1e-8f,5e-4f,false);
  struct Lv{int64_t h,w;float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A=(int64_t)ss.size();
  std::vector<std::string> names; { std::ifstream f(DN+"names.txt"); std::string s; while(f>>s)names.push_back(s);}
  auto save_ckpt=[&](const std::string& path){ std::vector<pt::Tensor> ck; size_t k=0;
    auto push=[&](const std::vector<float>&d,const std::vector<int64_t>&shp){pt::Tensor t; if(k<names.size())t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k;};
    for(auto&L:prov.layers){ std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end()); push(L.w->data,ws);
      if(L.kind==1){std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c);push(L.beta->data,c);push(L.rm,c);push(L.rv,c);} else push(L.b->data,{L.b->shape[0]});}
    pt::save_pt(ck,path); };
  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);
  int total=EPOCHS*(((int)tr.items.size()+BATCH-1)/BATCH), gstep=0; double best=-1;
  for(int ep=0;ep<EPOCHS;++ep){ AugCfg aug=baseAug; if(ep>=EPOCHS-closeMosaic){aug.mosaic=false;aug.mixup=false;}
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nb=0;
    for(size_t off=0;off<order.size();off+=BATCH){ std::vector<int> idx(order.begin()+off,order.begin()+std::min(order.size(),off+BATCH));
      Batch bt=load_minibatch(tr,idx,true,rng(),aug); int64_t B=bt.B,R=B*A,Mx=bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t q=r%A;ancx[r]=ax[q];ancy[r]=ay[q];strd[r]=ss[q];}
      prov.i=0; Head26U h=yolo26n_forward_u(bt.x,prov,true,ARC);
      auto branch=[&](std::vector<std::pair<Tensor,Tensor>>&hd,int topk){ std::vector<Tensor> bx={hd[0].first,hd[1].first,hd[2].first},cs={hd[0].second,hd[1].second,hd[2].second};
        auto pd=pack_levels(bx,B,A,4); auto ps=pack_levels(cs,B,A,NC0); std::vector<float> pdb,pss; decode_for_tal26(pd,ps,ax,ay,ss,R,A,NC0,pdb,pss);
        auto tal=tal_assign(pss,pdb,anc_img,bt.gt_labels,bt.gt_boxes,bt.mask,B,A,Mx,NC0,topk,0.5f,6.0f); return pure_v26_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC0); };
      auto Lm=branch(h.o2m,10), Lo=branch(h.o2o,1); auto tot=add(Lm.total,Lo.total);
      backward(tot); opt.lr=cosine_lr(gstep,total,lr0,warmup); opt.step(); ++gstep; eloss+=tot->data[0]; ++nb; free_graph(tot);
    }
    Dataset vac=va; double m50=run_val(vac,prov,S);
    printf("epoch %d/%d  loss %.3f%s\n", ep+1,EPOCHS,eloss/nb,m50>best?"  *best*":"");
    save_ckpt("last.pt"); if(m50>best){best=m50; save_ckpt("best.pt");}
  }
  printf("done. best val mAP@0.5 = %.4f. wrote last.pt / best.pt\n",best); return 0;
}

static int cmd_val(const Args& a){ DataYaml dy=parse_yaml(a.get("data")); int64_t S=a.geti("imgsz",64);
  auto prov=load_model(a.get("weights","best.pt")); Dataset va=read_yolo_dataset(join(dy.path,dy.val),S); run_val(va,prov,S); return 0; }

static int cmd_detect(const Args& a){
  int64_t S=a.geti("imgsz",64); float conf=a.getf("conf",0.25f); std::string src=a.get("source"),outp=a.get("out","out.png");
  auto prov=load_model(a.get("weights","best.pt"));
  int w0,h0,ch; unsigned char* im=stbi_load(src.c_str(),&w0,&h0,&ch,3); if(!im){printf("cannot load %s\n",src.c_str());return 1;}
  Letterbox lb; auto x=load_image_letterbox(src,S,lb); std::vector<float> ax,ay,st; make_anchors(S,ax,ay,st); int64_t A=(int64_t)st.size();
  prov.i=0; Head26U h=yolo26n_forward_u(x,prov,false,load_arch26());
  std::vector<Tensor> bx={h.o2o[0].first,h.o2o[1].first,h.o2o[2].first},cs={h.o2o[0].second,h.o2o[1].second,h.o2o[2].second};
  auto pd=pack_levels(bx,1,A,4); auto ps=pack_levels(cs,1,A,NC0); std::vector<float> pred; decode26(pd->data,ps->data,ax,ay,st,A,pred);
  auto dets=nms_free(pred,A,conf,300);
  auto put=[&](int px,int py,unsigned char r,unsigned char g,unsigned char b){if(px<0||py<0||px>=w0||py>=h0)return;unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b;};
  printf("%zu detections:\n",dets.size());
  for(auto&d:dets){ int x1=(int)std::round((d.x1-lb.left)/lb.r),y1=(int)std::round((d.y1-lb.top)/lb.r),x2=(int)std::round((d.x2-lb.left)/lb.r),y2=(int)std::round((d.y2-lb.top)/lb.r);
    x1=std::clamp(x1,0,w0-1);y1=std::clamp(y1,0,h0-1);x2=std::clamp(x2,0,w0-1);y2=std::clamp(y2,0,h0-1);
    for(int t=0;t<3;++t){for(int px=x1;px<=x2;++px){put(px,y1+t,255,60,60);put(px,y2-t,255,60,60);}for(int py=y1;py<=y2;++py){put(x1+t,py,255,60,60);put(x2-t,py,255,60,60);}}
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", d.cls<80?COCO[d.cls]:"?", d.conf, x1,y1,x2,y2); }
  stbi_write_png(outp.c_str(),w0,h0,3,im,w0*3); printf("wrote %s\n",outp.c_str()); stbi_image_free(im); return 0;
}

int main(int argc,char**argv){ setvbuf(stdout,nullptr,_IONBF,0);
  std::string cmd=argc>1?argv[1]:""; Args a=parse_args(argc,argv,2);
  DN=a.get("arch",DN); if(!DN.empty()&&DN.back()!='/')DN+='/';
  if(cmd=="train")return cmd_train(a); if(cmd=="val")return cmd_val(a); if(cmd=="detect")return cmd_detect(a);
  printf("usage: yolo26 <train|val|detect> --flags   (see header of pure/yolo26.cpp)\n"); return 1;
}
