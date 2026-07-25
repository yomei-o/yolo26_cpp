# yolo26 architecture (reverse-engineered from ultralytics 8.4.104, `yolo26.yaml`)

**yolo26 = yolo11 backbone/neck + a new head + one block variant.** Everything is built from
parts already in the sibling repos (conv-bn-silu, Bottleneck, C3k, C3k2, SPPF, attention/PSA,
depthwise). Widths follow `scales` (n/s/m/l/x), same as yolo11.

## Layer graph (24 layers) — identical topology to yolo11
```
0 Conv  1 Conv  2 C3k2  3 Conv  4 C3k2  5 Conv  6 C3k2  7 Conv  8 C3k2
9 SPPF  10 C2PSA  11 Up  12 Concat[-1,6]  13 C3k2  14 Up  15 Concat[-1,4]
16 C3k2(P3)  17 Conv  18 Concat[-1,13]  19 C3k2(P4)  20 Conv  21 Concat[-1,10]
22 C3k2(P5)  23 Detect[16,19,22]
```

## Δ1 — C3k2 config differs from yolo11 (more nested C3k in the neck)
`arch26_n` (the 8 C3k2 blocks as `n c3k inner`, + C2PSA repeats):
```
psa_n = 1
L2  1 0 0     L4  1 0 0     L6  1 1 2     L8  1 1 2
L13 1 1 2     L16 1 1 2     L19 1 1 2     L22 1 0 0   (see Δ2)
```
(yolo11n was `…,1 0 0,1 0 0,1 0 0,1 1 2` for L13/16/19/22 — yolo26 c3k-ifies the neck and the
P5 block L22 is not a plain C3k.)

## Δ2 — L22 (P5-head C3k2) block = `Sequential[Bottleneck, PSABlock]`
Its single block runs a Bottleneck then a **PSABlock**, which is exactly the C2PSA attention
unit already implemented: `attn(qkv/proj/pe) + ffn(Conv,Conv)`. So L22 = `cv1 → split →
[Bottleneck → PSABlock] (+ the split's other half) → cv2`. Conv-emit order inside the block:
Bottleneck.cv1, Bottleneck.cv2, PSA qkv, proj, pe, ffn[0], ffn[1].

## Δ3 — Detect head: NO DFL + end2end (NMS-free)
`nc=80, reg_max=1 (dfl=Identity), no=84, end2end=True, stride=[8,16,32]`, per level:
- **cv2 (box, one2many)**: `Conv(Cin→16,3) → Conv(16→16,3) → Conv2d(16→4,1)` → **4 raw ltrb** (no 16-bin DFL).
- **cv3 (cls, one2many)**: `[DWConv(Cin,3,g=Cin)→Conv(Cin→nc,1)] → [DWConv(nc,3,g=nc)→Conv(nc→nc,1)] → Conv2d(nc→nc,1)` → nc.
- **one2one_cv2 / one2one_cv3**: same structure, parallel branch used for the NMS-free (o2o) path.
Head conv-emit order: for the o2m branch, `for L in 0..2: cv2[L] convs then cv3[L] convs`; then the
same for the o2o branch.

## Decode / loss notes
- **box**: reg_max=1 ⇒ the 4 values are ltrb distances directly; `bbox = dist2bbox(4, anchor)*stride`
  (anchor-free, no softmax-integral). cls: sigmoid.
- **inference**: NMS-free — use the **one2one** branch, take top-`max_det`(=300) by score (no NMS).
- **train**: one2many branch (TAL, topk>1) + one2one branch (topk=1); box loss = IoU (no DFL loss term).
  (ultralytics also adds ProgLoss/STAL/MuSGD recipe tweaks — start without, add later.)

## Export
`python pure/ref/export_yolo26.py [imgsz]` dumps (into `data_net/`, gitignored, regenerate any time):
BN-folded convs `w*/b*.bin` + `manifest.txt` (126 convs) + `arch26.txt` + `head26.txt` + a fixed
`input.bin` + reference head outputs `ref_{o2m,o2o}_{box,cls}{0..2}.bin` for C++ forward parity.
