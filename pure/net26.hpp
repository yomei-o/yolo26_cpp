// yolo26 forward (fused conv+BN+SiLU), built on the yolo11 block helpers in net11.hpp.
// yolo26 = yolo11 backbone/neck (driven by arch26) with two deltas:
//   * L22 (P5-head C3k2) block = Bottleneck + PSABlock (the C2PSA attention unit)   [c3k2_psa]
//   * Detect head: reg_max=1 (NO DFL, box=4) + end2end (a second, one2one branch)   [two branches]
// The head's per-level conv layout is exactly yolo11's detect_level (box: k3,k3,plain;
// cls: dw,pw,dw,pw,plain) — only the box output width differs (4 vs 64), which comes from
// the weights. See pure/ref/ARCH.md.
#pragma once
#include "net11.hpp"          // Provider, conv_apply/cL, bottleneck, attention, sppf, c2psa,
                              // c3k2, detect_level, Arch11/C3D, pack_levels, load_net, rd

// C3k2 config for yolo26n (n c3k inner per stage): more nested C3k in the neck than yolo11.
inline Arch11 arch26_n() {
  return {{{1,0,false},{1,0,false},{1,2,true},{1,2,true},
           {1,2,true},{1,2,true},{1,2,true},{1,0,false}}, 1};   // L22 handled by c3k2_psa
}

// yolo26 L22 block: cv1 -> split -> [Bottleneck -> PSABlock] -> concat -> cv2.
// PSABlock == one C2PSA unit: x += attn(x); x += ffn(x).  Conv order matches export_yolo26.py.
inline Tensor c3k2_psa(const Tensor& x, Provider& p) {
  auto y0 = conv_apply(x, p.next());                    // cv1 -> 2c
  int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = bottleneck(outs[1], p, true);           // Bottleneck (shortcut)
  int64_t heads = c / 64, kd = 32, hd = 64;             // PSABlock attention (dim=c)
  last = add(last, attention(last, p, heads, kd, hd));  // qkv, proj, pe
  auto f = conv_apply(last, p.next()); f = conv_apply(f, p.next());   // ffn: Conv, Conv
  last = add(last, f);
  outs.push_back(last);
  return conv_apply(concat_ch(outs), p.next());         // cv2
}

// Two head branches, each = 3x detect_level (box k3k3plain -> 4, cls dw/pw/dw/pw/plain -> nc).
struct Head26 { std::vector<std::pair<Tensor, Tensor>> o2m, o2o; };  // (box, cls) per level

// yolo26 SPPF: like yolo11's but with a residual (self.add=True) -> cv2(cat) + x.
inline Tensor sppf26(const Tensor& x, Provider& p) {
  auto x1 = conv_apply(x, p.next());
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  auto y = conv_apply(concat_ch({x1, q1, q2, q3}), p.next());
  return add(y, x);
}

inline Head26 yolo26n_forward(const Tensor& x, Provider& p, const Arch11& A = arch26_n()) {
  auto C = [&](const Tensor& t, int i){ return c3k2(t, p, A.c3[i].n, A.c3[i].c3k, A.c3[i].inner, true); };
  auto x0 = cL(x, p);
  auto x1 = cL(x0, p);
  auto x2 = C(x1, 0);
  auto x3 = cL(x2, p);
  auto x4 = C(x3, 1);
  auto x5 = cL(x4, p);
  auto x6 = C(x5, 2);
  auto x7 = cL(x6, p);
  auto x8 = C(x7, 3);
  auto x9 = sppf26(x8, p);                               // yolo26 SPPF has a residual (add=True)
  auto x10 = c2psa(x9, p, A.psa_n);
  auto x11 = upsample_nearest(x10, 2);
  auto x12 = concat_ch({x11, x6});
  auto x13 = C(x12, 4);
  auto x14 = upsample_nearest(x13, 2);
  auto x15 = concat_ch({x14, x4});
  auto x16 = C(x15, 5);                                  // P3
  auto x17 = cL(x16, p);
  auto x18 = concat_ch({x17, x13});
  auto x19 = C(x18, 6);                                  // P4
  auto x20 = cL(x19, p);
  auto x21 = concat_ch({x20, x10});
  auto x22 = c3k2_psa(x21, p);                           // P5  (Bottleneck + PSABlock)
  Head26 h;
  for (auto& xi : {x16, x19, x22}) h.o2m.push_back(detect_level(xi, p));   // one2many branch
  for (auto& xi : {x16, x19, x22}) h.o2o.push_back(detect_level(xi, p));   // one2one branch (NMS-free)
  return h;
}
