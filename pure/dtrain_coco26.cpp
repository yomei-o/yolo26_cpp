// Device-resident yolo26 training over a standard-YOLO dataset (COCO128), with checkpoint.
// Device fwd(train BN, EMA running stats)+bwd+Adam; trusted host v26 loss (no-DFL, dual head:
// one2many topk=10 + one2one topk=1) bridged via device-head -> host-leaf -> loss -> inject grad
// -> dbackward_from. Saves last.pt/best.pt. Times s/epoch.
//   run: dtrain_coco26 <images_dir> <imgsz> <batch> <epochs> [weights.pt] [arch_dir] [lr]
//        (lr default 1e-3 for from-scratch; pass ~1e-4 to fine-tune a pretrained model without drifting it)
//   GPU: nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA [-DUSE_CUBLAS -lcublas] -Ipure/third_party pure/dtrain_coco26.cpp -o dtrain_coco26
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "dnet26.hpp"
#include "v26loss.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>

static const int64_t NC = 80;

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc>1?argv[1]:"coco128/images/train2017";
  int64_t S = argc>2?atoll(argv[2]):96;
  int BATCH = argc>3?atoi(argv[3]):4, EPOCHS = argc>4?atoi(argv[4]):3;
  std::string weights = argc>5?argv[5]:"init26.pt";
  std::string DN = argc>6?argv[6]:"pure/ref/data_net/"; if(!DN.empty()&&DN.back()!='/') DN+='/';
  float LR = argc>7?(float)atof(argv[7]):1e-3f;   // use ~1e-4 to fine-tune a pretrained model (1e-3 drifts its calibration)

  ProvD26 prov = dnet26_build(DN, weights);
  std::vector<DT> params = dnet26_params(prov);
  DAdam opt(params, LR);
  Arch11 A; { std::ifstream f(DN+"arch26.txt"); if(f){ f>>A.psa_n; int64_t n,c,i; while(f>>n>>c>>i) A.c3.push_back({n,i,(bool)c}); } else A=arch26_n(); }
  Dataset tr = read_yolo_dataset(dir, S);
  printf("yolo26 device train=%zu imgsz=%lld batch=%d epochs=%d lr=%.0e arch=%s\n", tr.items.size(),(long long)S,BATCH,EPOCHS,LR,DN.c_str());

  struct Lv{int64_t h,w;float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A_=(int64_t)ss.size();
  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);
  double best=1e30;

  for(int ep=0; ep<EPOCHS; ++ep){
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nb=0; auto t0=std::chrono::steady_clock::now();
    for(size_t off=0; off<order.size(); off+=BATCH){
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(),off+BATCH));
      Batch bt=load_minibatch(tr,idx,false,rng()); int64_t B=bt.B,R=B*A_,Mx=bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t a=r%A_;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
      opt.zero_grad();
      prov.i=0; HeadD26 dev = dnet26_forward(dfrom({B,3,S,S}, bt.x->data), prov, true, A);
      // bridge one branch: device heads -> host leaves -> v26 loss
      auto bridge=[&](std::vector<std::pair<DT,DT>>& hd, int topk, std::vector<Tensor>& lb, std::vector<Tensor>& lc){
        for(int l=0;l<3;++l){ auto&bl=hd[l].first;auto&cl=hd[l].second;
          lb.push_back(from_data({bl->shape[0],bl->shape[1],bl->shape[2],bl->shape[3]}, dto_host(bl), true));
          lc.push_back(from_data({cl->shape[0],cl->shape[1],cl->shape[2],cl->shape[3]}, dto_host(cl), true)); }
        auto pd=pack_levels(lb,B,A_,4); auto ps=pack_levels(lc,B,A_,NC);
        std::vector<float> pdb,pss; decode_for_tal26(pd,ps,ax,ay,ss,R,A_,NC,pdb,pss);
        auto tal=tal_assign(pss,pdb,anc_img,bt.gt_labels,bt.gt_boxes,bt.mask,B,A_,Mx,NC,topk,0.5f,6.0f);
        return pure_v26_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC);
      };
      std::vector<Tensor> mbx,mcs,obx,ocs;
      auto Lm=bridge(dev.o2m,10,mbx,mcs); auto Lo=bridge(dev.o2o,1,obx,ocs);
      auto total=add(Lm.total,Lo.total);
      backward(total);
      for(int l=0;l<3;++l){ thrust::copy(mbx[l]->grad.begin(),mbx[l]->grad.end(),dev.o2m[l].first->grad.begin());
        thrust::copy(mcs[l]->grad.begin(),mcs[l]->grad.end(),dev.o2m[l].second->grad.begin());
        thrust::copy(obx[l]->grad.begin(),obx[l]->grad.end(),dev.o2o[l].first->grad.begin());
        thrust::copy(ocs[l]->grad.begin(),ocs[l]->grad.end(),dev.o2o[l].second->grad.begin()); }
      std::vector<DT> heads; for(int l=0;l<3;++l){heads.push_back(dev.o2m[l].first);heads.push_back(dev.o2m[l].second);heads.push_back(dev.o2o[l].first);heads.push_back(dev.o2o[l].second);}
      dbackward_from(heads); opt.step();
      eloss+=total->data[0]; ++nb;
    }
    bk::sync(); double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); double avg=eloss/std::max(1,nb);
    dnet26_save(prov,DN,"last.pt"); if(avg<best){best=avg; dnet26_save(prov,DN,"best.pt");}
    printf("epoch %d/%d  loss %.4f  %.1f s/epoch%s\n", ep+1,EPOCHS,avg,secs,avg<=best?"  *best*":"");
  }
  printf("done. best loss %.4f (pure C++, yolo26)\n", best);
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
