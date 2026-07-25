// yolo26 detection loss: like v8's, but reg_max=1 (NO DFL) — the box branch's 4 values are
// the ltrb distances directly. Reuses ciou_rows / bce_logits / TAL from the v8 pieces.
//   loss = 7.5 * (1 - CIoU)·w  +  0.5 * BCE(cls)   (no DFL term)
#pragma once
#include "v8pure.hpp"    // dfl_decode(unused), ciou_rows, bce_logits, narrow_col, concat_cols...
#include "tal.hpp"       // tal_assign
#include <algorithm>
#include <cmath>

// direct box decode: pred (R,4) ltrb distances (grid units) + anchor centers -> xyxy (grid).
inline Tensor direct_decode(const Tensor& pred4, const Tensor& AX, const Tensor& AY, int64_t R) {
  auto d = reshape(pred4, {R, 4});
  auto l = narrow_col(d, 0), t = narrow_col(d, 1), r = narrow_col(d, 2), b = narrow_col(d, 3);
  return concat_cols({sub(AX, l), sub(AY, t), add(AX, r), add(AY, b)});
}

// host decode for TAL: raw ltrb -> pred boxes in IMAGE units (xyxy) + sigmoid scores.
inline void decode_for_tal26(const Tensor& pd, const Tensor& ps, const std::vector<float>& ax,
                             const std::vector<float>& ay, const std::vector<float>& ss,
                             int64_t R, int64_t A, int64_t nc,
                             std::vector<float>& pdb, std::vector<float>& pss) {
  pdb.assign(R * 4, 0); pss.assign(R * nc, 0);
  for (int64_t r = 0; r < R; ++r) {
    int64_t a = r % A; float axr = ax[a], ayr = ay[a], st = ss[a];
    float l = pd->data[r*4+0], t = pd->data[r*4+1], rr = pd->data[r*4+2], bb = pd->data[r*4+3];
    pdb[r*4+0] = (axr - l) * st; pdb[r*4+1] = (ayr - t) * st;
    pdb[r*4+2] = (axr + rr) * st; pdb[r*4+3] = (ayr + bb) * st;
    for (int64_t c = 0; c < nc; ++c) pss[r*nc+c] = 1.f / (1.f + std::exp(-ps->data[r*nc+c]));
  }
}

// loss given precomputed TAL targets (tb in image units, ts scores). anc in grid units.
inline LossOut pure_v26_loss(const Tensor& pred4, const Tensor& pred_scores,
                             const std::vector<float>& ancx, const std::vector<float>& ancy,
                             const std::vector<float>& stride, const std::vector<float>& tb_img,
                             const std::vector<float>& ts, int64_t R, int64_t nc) {
  auto AX = from_data({R, 1}, ancx), AY = from_data({R, 1}, ancy);
  auto boxes_grid = direct_decode(pred4, AX, AY, R);

  std::vector<float> tbg(R * 4), wdat(R, 0.f); double tss_d = 0;
  for (int64_t r = 0; r < R; ++r) {
    for (int j = 0; j < 4; ++j) tbg[r*4+j] = tb_img[r*4+j] / stride[r];   // GT -> grid units
    for (int64_t c = 0; c < nc; ++c) { wdat[r] += ts[r*nc+c]; tss_d += ts[r*nc+c]; }
  }
  float inv_tss = 1.f / (float)std::max(1.0, tss_d);
  auto weight = from_data({R, 1}, wdat), tbg_t = from_data({R, 4}, tbg), ts_t = from_data({R, nc}, ts);

  auto ciou = ciou_rows(boxes_grid, tbg_t);                                   // (R,1)
  auto box = mul_scalar(sum(mul(add_scalar(mul_scalar(ciou, -1.f), 1.f), weight)), inv_tss);
  auto cls = mul_scalar(sum(bce_logits(pred_scores, ts_t)), inv_tss);
  auto total = add(mul_scalar(box, 7.5f), mul_scalar(cls, 0.5f));             // no DFL
  return {total, box, cls, cls};                                             // (dfl slot unused)
}
