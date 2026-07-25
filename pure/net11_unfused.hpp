// (unfused variant)
// yolo11n forward with conv + BatchNorm2d + SiLU kept SEPARATE (BN not folded), for BN
// training and .pt write-back. Same topology/attention as net11.hpp. Consumes layers in
// the order export_unfused11.py emits them.
#pragma once
#include "net11.hpp"       // rd(), pack_levels(), ops
#include "bn.hpp"
#include <fstream>
#include <string>
#include <cmath>

struct LayerU { int kind; int64_t stride, pad, groups; float eps; int act; Tensor w, b, gamma, beta; std::vector<float> rm, rv; };
struct ProviderU { std::vector<LayerU> layers; size_t i = 0; LayerU& next() { return layers[i++]; } };

inline ProviderU load_net_unfused(const std::string& D) {
  std::ifstream f(D + "manifest_unfused.txt"); if (!f) { printf("run: python pure/ref/export_unfused11.py\n"); std::exit(1); }
  int n; f >> n; ProviderU p;
  for (int i = 0; i < n; ++i) {
    int kind; int64_t Co, Ci, k, s, pad, g; float eps; int act; f >> kind >> Co >> Ci >> k >> s >> pad >> g >> eps >> act;
    LayerU L; L.kind = kind; L.stride = s; L.pad = pad; L.groups = g; L.eps = eps; L.act = act;
    L.w = from_data({Co, Ci, k, k}, rd(D + "cw" + std::to_string(i) + ".bin"), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, rd(D + "bg" + std::to_string(i) + ".bin"), true);
      L.beta  = from_data({Co}, rd(D + "bb" + std::to_string(i) + ".bin"), true);
      L.rm = rd(D + "rm" + std::to_string(i) + ".bin"); L.rv = rd(D + "rv" + std::to_string(i) + ".bin");
    } else L.b = from_data({Co}, rd(D + "cb" + std::to_string(i) + ".bin"), true);
    p.layers.push_back(std::move(L));
  }
  return p;
}

// Same as load_net_unfused but weights come from a state_dict .pt (read in pure C++ by
// ptio) instead of per-layer .bin files. Arch is from the tiny committable manifest; each
// tensor is looked up by its state_dict key from names.txt (engine order). No Python.
#include "ptio.hpp"
#include <map>
inline ProviderU load_net_unfused_pt(const std::string& D, const std::string& pt_path) {
  std::ifstream f(D + "manifest_unfused.txt"); if (!f) { printf("missing %smanifest_unfused.txt\n", D.c_str()); std::exit(1); }
  std::vector<std::string> names; { std::ifstream nf(D + "names.txt"); std::string s; while (nf >> s) names.push_back(s); }
  // state_dict .pt (make_init_pt) OR a raw {'model': nn.Module} checkpoint (yolo11n.pt).
  auto ts = pt::load_pt(pt_path); if (ts.empty()) ts = pt::load_pt_module(pt_path);
  std::map<std::string, std::vector<float>> W; for (auto& t : ts) W[t.name] = t.data;
  auto get = [&](size_t idx) -> std::vector<float>& {
    auto it = W.find(names[idx]); if (it == W.end()) { printf("key not in %s: %s\n", pt_path.c_str(), names[idx].c_str()); std::exit(1); } return it->second; };
  int n; f >> n; ProviderU p; size_t ni = 0;
  for (int i = 0; i < n; ++i) {
    int kind; int64_t Co, Ci, k, s, pad, g; float eps; int act; f >> kind >> Co >> Ci >> k >> s >> pad >> g >> eps >> act;
    LayerU L; L.kind = kind; L.stride = s; L.pad = pad; L.groups = g; L.eps = eps; L.act = act;
    L.w = from_data({Co, Ci, k, k}, get(ni++), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, get(ni++), true);
      L.beta  = from_data({Co}, get(ni++), true);
      L.rm = get(ni++); L.rv = get(ni++);
    } else L.b = from_data({Co}, get(ni++), true);
    p.layers.push_back(std::move(L));
  }
  return p;
}

inline Tensor applyU(const Tensor& x, LayerU& L, bool tr) {
  Tensor y;
  if (L.kind == 1) { y = conv2d(x, L.w, nullptr, L.stride, L.pad, L.groups);
    y = batchnorm2d(y, L.gamma, L.beta, L.rm, L.rv, L.eps, tr, 0.03f); }
  else y = conv2d(x, L.w, L.b, L.stride, L.pad, L.groups);
  return L.act ? silu(y) : y;
}
inline Tensor cLU(const Tensor& x, ProviderU& p, bool tr) { return applyU(x, p.next(), tr); }
inline Tensor bottU(const Tensor& x, ProviderU& p, bool sc, bool tr) {
  auto h = applyU(x, p.next(), tr); h = applyU(h, p.next(), tr); return sc ? add(h, x) : h;
}
inline Tensor c3kU(const Tensor& x, ProviderU& p, int64_t inner, bool sc, bool tr) {
  auto last = applyU(x, p.next(), tr);
  for (int64_t j = 0; j < inner; ++j) last = bottU(last, p, sc, tr);
  auto y2 = applyU(x, p.next(), tr);
  return applyU(concat_ch({last, y2}), p.next(), tr);
}
inline Tensor c3k2U(const Tensor& x, ProviderU& p, int64_t n, bool is_c3k, int64_t inner, bool sc, bool tr) {
  auto y0 = applyU(x, p.next(), tr); int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)}; Tensor last = outs[1];
  for (int64_t j = 0; j < n; ++j) { last = is_c3k ? c3kU(last, p, inner, sc, tr) : bottU(last, p, sc, tr); outs.push_back(last); }
  return applyU(concat_ch(outs), p.next(), tr);
}
inline Tensor sppfU(const Tensor& x, ProviderU& p, bool tr) {
  auto x1 = applyU(x, p.next(), tr);
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return applyU(concat_ch({x1, q1, q2, q3}), p.next(), tr);
}
inline Tensor attentionU(const Tensor& x, ProviderU& p, int64_t heads, int64_t key_dim, int64_t head_dim, bool tr) {
  LayerU& qkv = p.next(); LayerU& proj = p.next(); LayerU& pe = p.next();
  auto qkvo = applyU(x, qkv, tr);
  int64_t B = x->shape[0], H = x->shape[2], W = x->shape[3], N = H * W, per = 2 * key_dim + head_dim;
  float scale = 1.f / std::sqrt((float)key_dim);
  std::vector<Tensor> xa_b, vf_b;
  for (int64_t b = 0; b < B; ++b) {
    auto qb = slice_batch(qkvo, b); std::vector<Tensor> xatt, vfull;
    for (int64_t hh = 0; hh < heads; ++hh) {
      int64_t off = hh * per;
      auto q = reshape(slice_ch(qb, off, off + key_dim), {key_dim, N});
      auto k = reshape(slice_ch(qb, off + key_dim, off + 2 * key_dim), {key_dim, N});
      auto v = reshape(slice_ch(qb, off + 2 * key_dim, off + per), {head_dim, N});
      auto attn = softmax_rows(mul_scalar(matmul(transpose2d(q), k), scale));
      xatt.push_back(reshape(matmul(v, transpose2d(attn)), {1, head_dim, H, W}));
      vfull.push_back(reshape(v, {1, head_dim, H, W}));
    }
    xa_b.push_back(concat_ch(xatt)); vf_b.push_back(concat_ch(vfull));
  }
  auto xa = concat_batch(xa_b);
  auto pev = applyU(concat_batch(vf_b), pe, tr);
  return applyU(add(xa, pev), proj, tr);
}
inline Tensor c2psaU(const Tensor& x, ProviderU& p, int64_t n, bool tr) {
  auto y = applyU(x, p.next(), tr); int64_t twoc = y->shape[1], dim = twoc / 2;
  int64_t heads = dim / 64, kd = 32, hd = 64;
  auto a = slice_ch(y, 0, dim); auto b = slice_ch(y, dim, twoc);
  for (int64_t j = 0; j < n; ++j) { b = add(b, attentionU(b, p, heads, kd, hd, tr));
    auto f = applyU(b, p.next(), tr); f = applyU(f, p.next(), tr); b = add(b, f); }
  return applyU(concat_ch({a, b}), p.next(), tr);
}
inline std::pair<Tensor, Tensor> detect_level_u(const Tensor& x, ProviderU& p, bool tr) {
  auto hb = cLU(x, p, tr); hb = cLU(hb, p, tr); auto box = applyU(hb, p.next(), tr);
  auto hc = cLU(x, p, tr); hc = cLU(hc, p, tr); hc = cLU(hc, p, tr); hc = cLU(hc, p, tr);
  auto cls = applyU(hc, p.next(), tr);
  return {box, cls};
}
inline std::vector<std::pair<Tensor, Tensor>> yolo11n_forward_u(const Tensor& x, ProviderU& p, bool tr,
                                                               const Arch11& A = arch11_n()) {
  auto C = [&](const Tensor& t, int i){ return c3k2U(t, p, A.c3[i].n, A.c3[i].c3k, A.c3[i].inner, true, tr); };
  auto x0 = cLU(x, p, tr); auto x1 = cLU(x0, p, tr);
  auto x2 = C(x1, 0); auto x3 = cLU(x2, p, tr);
  auto x4 = C(x3, 1); auto x5 = cLU(x4, p, tr);
  auto x6 = C(x5, 2); auto x7 = cLU(x6, p, tr);
  auto x8 = C(x7, 3); auto x9 = sppfU(x8, p, tr);
  auto x10 = c2psaU(x9, p, A.psa_n, tr);
  auto x11 = upsample_nearest(x10, 2); auto x12 = concat_ch({x11, x6}); auto x13 = C(x12, 4);
  auto x14 = upsample_nearest(x13, 2); auto x15 = concat_ch({x14, x4}); auto x16 = C(x15, 5);
  auto x17 = cLU(x16, p, tr); auto x18 = concat_ch({x17, x13}); auto x19 = C(x18, 6);
  auto x20 = cLU(x19, p, tr); auto x21 = concat_ch({x20, x10}); auto x22 = C(x21, 7);
  std::vector<std::pair<Tensor, Tensor>> out;
  for (auto& xi : {x16, x19, x22}) out.push_back(detect_level_u(xi, p, tr));
  return out;
}
