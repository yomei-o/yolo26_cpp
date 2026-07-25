// yolo26 CPU training — real loop, pure C++ / no Python at run time. Uses the one2many head
// branch with TAL + the reg_max=1 (no-DFL) box loss. Saves last.pt / best.pt (lowest loss).
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\train_cli26.cpp
//   run:   train_cli26 <train_list> <val_list> <epochs> <batch> <init.pt> <imgsz> [mosaic] [arch_dir]
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "net26_unfused.hpp"
#include "v26loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>

static const int64_t NC = 80;

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string trainL = argc>1?argv[1]:"pure/ref/data_yolo/images/train";
  std::string valL   = argc>2?argv[2]:trainL;
  int EPOCHS = argc>3?atoi(argv[3]):8, BATCH = argc>4?atoi(argv[4]):4;
  std::string initpt = argc>5?argv[5]:"init26.pt";
  int imgsz = argc>6?atoi(argv[6]):64;
  bool mosaic = argc>7?atoi(argv[7])!=0:true;
  std::string DN = argc>8?argv[8]:"pure/ref/data_net/"; if (!DN.empty() && DN.back()!='/') DN += '/';

  Dataset tr = read_yolo_dataset(trainL, imgsz);
  int64_t S = tr.S;
  printf("yolo26n train=%zu imgsz=%lld batch=%d epochs=%d mosaic=%d\n", tr.items.size(), (long long)S, BATCH, EPOCHS, (int)mosaic);

  ProviderU prov; { std::ifstream t(initpt); if (t.good()) { printf("init weights <- %s\n", initpt.c_str()); prov = load_net_unfused_pt(DN, initpt); } else prov = load_net_unfused(DN); }
  Arch11 ARC = arch26_n();
  std::vector<Tensor> params; for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  struct Lv { int64_t h,w; float s; }; std::vector<Lv> lv = {{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for (auto& L:lv) for (int64_t y=0;y<L.h;++y) for (int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A = ss.size();

  std::vector<std::string> names; { std::ifstream f(DN + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, const std::vector<int64_t>& shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end()); push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  AugCfg aug; aug.mosaic = mosaic; aug.mixup = mosaic;
  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0); std::mt19937 rng(0);
  int steps = ((int)tr.items.size()+BATCH-1)/BATCH, total = EPOCHS*steps, gstep = 0; double best = 1e30;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    if (ep >= EPOCHS - std::max(1, EPOCHS/10)) { aug.mosaic = false; aug.mixup = false; }
    std::shuffle(order.begin(), order.end(), rng); double eloss=0, ebox=0, ecls=0; int nb=0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(), off+BATCH));
      Batch bt = load_minibatch(tr, idx, true, rng(), aug);
      int64_t B = bt.B, R = B*A, Mx = bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for (int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
      prov.i = 0; Head26U h = yolo26n_forward_u(bt.x, prov, true, ARC);
      std::vector<Tensor> bx={h.o2m[0].first,h.o2m[1].first,h.o2m[2].first}, cs={h.o2m[0].second,h.o2m[1].second,h.o2m[2].second};
      auto pd = pack_levels(bx, B, A, 4); auto ps = pack_levels(cs, B, A, NC);
      std::vector<float> pdb, pss; decode_for_tal26(pd, ps, ax, ay, ss, R, A, NC, pdb, pss);
      auto tal = tal_assign(pss, pdb, anc_img, bt.gt_labels, bt.gt_boxes, bt.mask, B, A, Mx, NC, 10, 0.5f, 6.0f);
      auto Lo = pure_v26_loss(pd, ps, ancx, ancy, strd, tal.tb, tal.ts, R, NC);
      backward(Lo.total);
      opt.lr = cosine_lr(gstep, total, 1e-3f, std::max(1, total/20)); opt.step(); ++gstep;
      eloss += Lo.total->data[0]; ebox += Lo.box->data[0]; ecls += Lo.cls->data[0]; ++nb; free_graph(Lo.total);
    }
    printf("epoch %2d/%d  loss %6.3f (box %.3f cls %.3f)\n", ep+1, EPOCHS, eloss/nb, ebox/nb, ecls/nb);
    save_ckpt("last.pt"); if (eloss/nb < best) { best = eloss/nb; save_ckpt("best.pt"); }
  }
  printf("done. best loss %.4f. wrote last.pt / best.pt (pure C++, yolo26n)\n", best);
  return 0;
}
