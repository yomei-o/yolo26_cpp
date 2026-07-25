// Device-resident yolo26 net (Thrust device_vector; one source CPU/GPU) built on dnet11.hpp's
// device block helpers. yolo26 deltas vs yolo11: SPPF residual (add=True), L22 Bottleneck+PSABlock,
// and the no-DFL dual head (box=4). Reuses dnet11_build/params/save (manifest-driven, generic).
#pragma once
#include "dnet11.hpp"        // ProvD11, dnet11_build/params/save, d11_* helpers, DT ops
#include "net26.hpp"         // arch26_n

// yolo26 SPPF with residual (self.add=True): cv2(cat) + x.
static inline DT d26_sppf(DT x, ProvD11& p, bool tr) {
  DT x1 = d11_apply(x, p.next(), tr);
  DT q1 = dmaxpool2d(x1,5,1,2), q2 = dmaxpool2d(q1,5,1,2), q3 = dmaxpool2d(q2,5,1,2);
  return dadd(d11_apply(dconcat({x1,q1,q2,q3}), p.next(), tr), x);
}
// yolo26 L22 C3k2 block: cv1 -> split -> [Bottleneck -> PSABlock] -> concat -> cv2.
static inline DT d26_c3k2_psa(DT x, ProvD11& p, bool tr) {
  DT y0 = d11_apply(x, p.next(), tr); int64_t twoc = y0->shape[1], c = twoc/2;
  std::vector<DT> outs = {dslice(y0,0,c), dslice(y0,c,twoc)};
  DT last = d11_bott(outs[1], p, true, tr);
  int64_t heads = c/64, kd = 32, hd = 64;
  last = dadd(last, d11_attention(last, p, heads, kd, hd, tr));
  DT f = d11_apply(last, p.next(), tr); f = d11_apply(f, p.next(), tr); last = dadd(last, f);
  outs.push_back(last);
  return d11_apply(dconcat(outs), p.next(), tr);
}

struct HeadD26 { std::vector<std::pair<DT,DT>> o2m, o2o; };   // (box, cls) per level

inline HeadD26 dnet26_forward(DT x, ProvD11& p, bool tr, const Arch11& A) {
  auto C = [&](DT t, int i){ return d11_c3k2(t, p, A.c3[i].n, A.c3[i].c3k, A.c3[i].inner, true, tr); };
  DT x0=d11_cL(x,p,tr), x1=d11_cL(x0,p,tr);
  DT x2=C(x1,0); DT x3=d11_cL(x2,p,tr);
  DT x4=C(x3,1); DT x5=d11_cL(x4,p,tr);
  DT x6=C(x5,2); DT x7=d11_cL(x6,p,tr);
  DT x8=C(x7,3); DT x9=d26_sppf(x8,p,tr);
  DT x10=d11_c2psa(x9,p,A.psa_n,tr);
  DT x11=dupsample2x(x10); DT x12=dconcat({x11,x6}); DT x13=C(x12,4);
  DT x14=dupsample2x(x13); DT x15=dconcat({x14,x4}); DT x16=C(x15,5);
  DT x17=d11_cL(x16,p,tr); DT x18=dconcat({x17,x13}); DT x19=C(x18,6);
  DT x20=d11_cL(x19,p,tr); DT x21=dconcat({x20,x10}); DT x22=d26_c3k2_psa(x21,p,tr);
  HeadD26 h; DT xs[3]={x16,x19,x22};
  for (auto& xi : xs) h.o2m.push_back(d11_detect(xi,p,tr));   // one2many
  for (auto& xi : xs) h.o2o.push_back(d11_detect(xi,p,tr));   // one2one (NMS-free)
  return h;
}

// build / params / save are generic (manifest-driven) — reuse dnet11's.
using ProvD26 = ProvD11;
inline ProvD26 dnet26_build(const std::string& arch, const std::string& weights){ return dnet11_build(arch, weights); }
inline std::vector<DT> dnet26_params(ProvD26& p){ return dnet11_params(p); }
inline void dnet26_save(ProvD26& p, const std::string& arch, const std::string& path){ dnet11_save(p, arch, path); }
