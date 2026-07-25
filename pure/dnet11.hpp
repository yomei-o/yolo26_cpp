// Device-resident yolo11 net + trainable provider + checkpoint save. Mirrors net11_unfused.hpp:
// C3k2 (Bottleneck / nested C3k), SPPF, C2PSA multi-head attention (qkv/softmax/proj + pe),
// depthwise detect head. Uses device ops incl. dreshape/dtranspose2d/dsoftmax_rows/dmul_scalar/
// dslice_batch/dconcat_batch. Per-layer pad/groups/act from the manifest. One source CPU/GPU.
#pragma once
#include "net11_unfused.hpp"   // ProviderU (LayerU: kind,stride,pad,groups,act,eps), load_net_unfused_pt
#include "net11.hpp"           // Arch11, arch11_n
#include "dtensor.hpp"
#include <string>
#include <fstream>
#include <cmath>

struct DTL11 { int kind; int64_t stride, pad, groups; int act; float eps; DT w, gamma, beta, b, rm, rv; };
struct ProvD11 { std::vector<DTL11> L; size_t i = 0; DTL11& next() { return L[i++]; } };

inline ProvD11 dnet11_build(const std::string& arch, const std::string& weights) {
  ProviderU pu = load_net_unfused_pt(arch, weights);
  ProvD11 p;
  for (auto& L : pu.layers) {
    DTL11 d; d.kind=L.kind; d.stride=L.stride; d.pad=L.pad; d.groups=L.groups; d.act=L.act; d.eps=L.eps;
    int64_t Co=L.w->shape[0],Ci=L.w->shape[1],k=L.w->shape[2];
    d.w = dfrom({Co,Ci,k,k}, L.w->data);
    if (L.kind==1){ d.gamma=dfrom({Co},L.gamma->data); d.beta=dfrom({Co},L.beta->data); d.rm=dfrom({Co},L.rm); d.rv=dfrom({Co},L.rv); }
    else d.b=dfrom({Co}, L.b->data);
    p.L.push_back(std::move(d));
  }
  return p;
}
inline std::vector<DT> dnet11_params(ProvD11& p) {
  std::vector<DT> ps; for (auto& L : p.L) { ps.push_back(L.w); if (L.kind==1){ps.push_back(L.gamma);ps.push_back(L.beta);} else ps.push_back(L.b); } return ps;
}

static inline DT d11_apply(DT x, DTL11& L, bool tr) {
  DT y;
  if (L.kind==1) y = dbn(dconv2d(x, L.w, DT(), L.stride, L.pad, L.groups), L.gamma, L.beta, L.eps, tr?L.rm:DT(), tr?L.rv:DT(), 0.03f);
  else           y = dconv2d(x, L.w, L.b, L.stride, L.pad, L.groups);
  return L.act ? dsilu(y) : y;
}
static inline DT d11_cL(DT x, ProvD11& p, bool tr) { return d11_apply(x, p.next(), tr); }
static inline DT d11_bott(DT x, ProvD11& p, bool sc, bool tr) {
  DT h=d11_apply(x,p.next(),tr); h=d11_apply(h,p.next(),tr); return sc?dadd(h,x):h;
}
static inline DT d11_c3k(DT x, ProvD11& p, int64_t inner, bool sc, bool tr) {
  DT last=d11_apply(x,p.next(),tr);
  for (int64_t j=0;j<inner;++j) last=d11_bott(last,p,sc,tr);
  DT y2=d11_apply(x,p.next(),tr);
  return d11_apply(dconcat({last,y2}), p.next(), tr);
}
static inline DT d11_c3k2(DT x, ProvD11& p, int64_t n, bool is_c3k, int64_t inner, bool sc, bool tr) {
  DT y0=d11_apply(x,p.next(),tr); int64_t twoc=y0->shape[1], c=twoc/2;
  std::vector<DT> outs={dslice(y0,0,c), dslice(y0,c,twoc)}; DT last=outs[1];
  for (int64_t j=0;j<n;++j){ last = is_c3k ? d11_c3k(last,p,inner,sc,tr) : d11_bott(last,p,sc,tr); outs.push_back(last); }
  return d11_apply(dconcat(outs), p.next(), tr);
}
static inline DT d11_sppf(DT x, ProvD11& p, bool tr) {
  DT x1=d11_apply(x,p.next(),tr); DT q1=dmaxpool2d(x1,5,1,2),q2=dmaxpool2d(q1,5,1,2),q3=dmaxpool2d(q2,5,1,2);
  return d11_apply(dconcat({x1,q1,q2,q3}), p.next(), tr);
}
static inline DT d11_attention(DT x, ProvD11& p, int64_t heads, int64_t kd, int64_t hd, bool tr) {
  DTL11& qkv=p.next(); DTL11& proj=p.next(); DTL11& pe=p.next();
  DT qkvo = d11_apply(x, qkv, tr);
  int64_t B=x->shape[0],H=x->shape[2],W=x->shape[3],N=H*W,per=2*kd+hd; float scale=1.f/std::sqrt((float)kd);
  std::vector<DT> xa_b, vf_b;
  for (int64_t b=0;b<B;++b) {
    DT qb=dslice_batch(qkvo,b); std::vector<DT> xatt, vfull;
    for (int64_t hh=0;hh<heads;++hh) { int64_t off=hh*per;
      DT q=dreshape(dslice(qb,off,off+kd),{kd,N});
      DT k=dreshape(dslice(qb,off+kd,off+2*kd),{kd,N});
      DT v=dreshape(dslice(qb,off+2*kd,off+per),{hd,N});
      DT attn=dsoftmax_rows(dmul_scalar(dmatmul(dtranspose2d(q),k),scale));
      xatt.push_back(dreshape(dmatmul(v,dtranspose2d(attn)),{1,hd,H,W}));
      vfull.push_back(dreshape(v,{1,hd,H,W}));
    }
    xa_b.push_back(dconcat(xatt)); vf_b.push_back(dconcat(vfull));
  }
  DT xa=dconcat_batch(xa_b); DT pev=d11_apply(dconcat_batch(vf_b), pe, tr);
  return d11_apply(dadd(xa,pev), proj, tr);
}
static inline DT d11_c2psa(DT x, ProvD11& p, int64_t n, bool tr) {
  DT y=d11_apply(x,p.next(),tr); int64_t twoc=y->shape[1], dim=twoc/2; int64_t heads=dim/64, kd=32, hd=64;
  DT a=dslice(y,0,dim); DT b=dslice(y,dim,twoc);
  for (int64_t j=0;j<n;++j){ b=dadd(b, d11_attention(b,p,heads,kd,hd,tr));
    DT f=d11_apply(b,p.next(),tr); f=d11_apply(f,p.next(),tr); b=dadd(b,f); }
  return d11_apply(dconcat({a,b}), p.next(), tr);
}
static inline std::pair<DT,DT> d11_detect(DT x, ProvD11& p, bool tr) {
  DT hb=d11_cL(x,p,tr); hb=d11_cL(hb,p,tr); DT box=d11_apply(hb,p.next(),tr);
  DT hc=d11_cL(x,p,tr); hc=d11_cL(hc,p,tr); hc=d11_cL(hc,p,tr); hc=d11_cL(hc,p,tr); DT cls=d11_apply(hc,p.next(),tr);
  return {box,cls};
}
inline std::vector<std::pair<DT,DT>> dnet11_forward(DT x, ProvD11& p, bool tr, const Arch11& A) {
  auto C = [&](DT t, int i){ return d11_c3k2(t, p, A.c3[i].n, A.c3[i].c3k, A.c3[i].inner, true, tr); };
  DT x0=d11_cL(x,p,tr), x1=d11_cL(x0,p,tr);
  DT x2=C(x1,0); DT x3=d11_cL(x2,p,tr);
  DT x4=C(x3,1); DT x5=d11_cL(x4,p,tr);
  DT x6=C(x5,2); DT x7=d11_cL(x6,p,tr);
  DT x8=C(x7,3); DT x9=d11_sppf(x8,p,tr);
  DT x10=d11_c2psa(x9,p,A.psa_n,tr);
  DT x11=dupsample2x(x10); DT x12=dconcat({x11,x6}); DT x13=C(x12,4);
  DT x14=dupsample2x(x13); DT x15=dconcat({x14,x4}); DT x16=C(x15,5);
  DT x17=d11_cL(x16,p,tr); DT x18=dconcat({x17,x13}); DT x19=C(x18,6);
  DT x20=d11_cL(x19,p,tr); DT x21=dconcat({x20,x10}); DT x22=C(x21,7);
  std::vector<std::pair<DT,DT>> out; DT xs[3]={x16,x19,x22};
  for (auto& xi : xs) out.push_back(d11_detect(xi,p,tr)); return out;
}
inline void dnet11_save(ProvD11& p, const std::string& arch, const std::string& path) {
  std::vector<std::string> names; { std::ifstream f(arch+"names.txt"); std::string s; while (f>>s) names.push_back(s); }
  std::vector<pt::Tensor> ck; size_t k=0;
  auto push=[&](DT t){ pt::Tensor o; if(k<names.size())o.name=names[k]; o.shape.assign(t->shape.begin(),t->shape.end()); o.data=dto_host(t); ck.push_back(std::move(o)); ++k; };
  for (auto& L:p.L){ push(L.w); if(L.kind==1){push(L.gamma);push(L.beta);push(L.rm);push(L.rv);} else push(L.b); }
  pt::save_pt(ck, path);
}
