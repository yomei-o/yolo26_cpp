// yolo26 forward with conv + BatchNorm2d + SiLU kept SEPARATE (BN not folded), for BN training
// and .pt write-back. Same topology as net26.hpp; built on net11_unfused.hpp's helpers.
// Deltas vs yolo11: L22 = Bottleneck+PSABlock (c3k2_psaU), head = two branches (o2m + o2o).
#pragma once
#include "net11_unfused.hpp"   // LayerU/ProviderU, applyU/cLU, bottU, attentionU, c2psaU,
                              // c3k2U, sppfU, detect_level_u, load_net_unfused[_pt], Arch11
#include "net26.hpp"           // arch26_n()

// yolo26 L22 block (unfused): cv1 -> split -> [Bottleneck -> PSABlock] -> concat -> cv2.
inline Tensor c3k2_psaU(const Tensor& x, ProviderU& p, bool tr) {
  auto y0 = applyU(x, p.next(), tr); int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = bottU(outs[1], p, true, tr);                       // Bottleneck (shortcut)
  int64_t heads = c / 64, kd = 32, hd = 64;                        // PSABlock (dim=c)
  last = add(last, attentionU(last, p, heads, kd, hd, tr));        // qkv, proj, pe
  auto f = applyU(last, p.next(), tr); f = applyU(f, p.next(), tr);// ffn
  last = add(last, f);
  outs.push_back(last);
  return applyU(concat_ch(outs), p.next(), tr);                    // cv2
}

struct Head26U { std::vector<std::pair<Tensor, Tensor>> o2m, o2o; };

// yolo26 SPPF (unfused) with residual (self.add=True).
inline Tensor sppf26U(const Tensor& x, ProviderU& p, bool tr) {
  auto x1 = applyU(x, p.next(), tr);
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  auto y = applyU(concat_ch({x1, q1, q2, q3}), p.next(), tr);
  return add(y, x);
}

inline Head26U yolo26n_forward_u(const Tensor& x, ProviderU& p, bool tr, const Arch11& A = arch26_n()) {
  auto C = [&](const Tensor& t, int i){ return c3k2U(t, p, A.c3[i].n, A.c3[i].c3k, A.c3[i].inner, true, tr); };
  auto x0 = cLU(x, p, tr); auto x1 = cLU(x0, p, tr);
  auto x2 = C(x1, 0); auto x3 = cLU(x2, p, tr);
  auto x4 = C(x3, 1); auto x5 = cLU(x4, p, tr);
  auto x6 = C(x5, 2); auto x7 = cLU(x6, p, tr);
  auto x8 = C(x7, 3); auto x9 = sppf26U(x8, p, tr);
  auto x10 = c2psaU(x9, p, A.psa_n, tr);
  auto x11 = upsample_nearest(x10, 2); auto x12 = concat_ch({x11, x6}); auto x13 = C(x12, 4);
  auto x14 = upsample_nearest(x13, 2); auto x15 = concat_ch({x14, x4}); auto x16 = C(x15, 5);
  auto x17 = cLU(x16, p, tr); auto x18 = concat_ch({x17, x13}); auto x19 = C(x18, 6);
  auto x20 = cLU(x19, p, tr); auto x21 = concat_ch({x20, x10}); auto x22 = c3k2_psaU(x21, p, tr);
  Head26U h;
  for (auto& xi : {x16, x19, x22}) h.o2m.push_back(detect_level_u(xi, p, tr));
  for (auto& xi : {x16, x19, x22}) h.o2o.push_back(detect_level_u(xi, p, tr));
  return h;
}
