"""Export yolo26n (BN-folded) convs in the exact order the pure-C++ forward will consume them,
plus a fixed input and reference head outputs for parity. yolo26 = yolo11 backbone/neck
(C3k2/SPPF/C2PSA) + a NEW head: reg_max=1 (NO DFL, direct 4-value box) and end2end (an extra
one2one branch, NMS-free). Consumption order per detect level: cv2, cv3, one2one_cv2, one2one_cv3.
Usage: python export_yolo26.py [imgsz] [yolo26|yolo26n|...]"""
import os, sys, torch, torch.nn as nn
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MODEL = sys.argv[2] if len(sys.argv) > 2 else "yolo26"
def is_c3k(m): return type(m).__name__ == "C3k"

ym = YOLO(MODEL + ".yaml"); L = ym.model.model.eval()   # random init (parity is weight-agnostic)
convs = []   # (w, b, k, s, pad, groups, act)
def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * (bn.weight / std).reshape(-1,1,1,1), bn.bias - bn.weight * bn.running_mean / std
def emitC(cv):
    w, b = fuse(cv); c = cv.conv; act = 1 if isinstance(cv.act, nn.SiLU) else 0
    convs.append((w, b, c.kernel_size[0], c.stride[0], c.padding[0], c.groups, act))
def emitP(c): convs.append((c.weight, c.bias, c.kernel_size[0], c.stride[0], c.padding[0], c.groups, 0))
def emit_bott(b): emitC(b.cv1); emitC(b.cv2)
def emit_c3k(m): emitC(m.cv1); [emit_bott(b) for b in m.m]; emitC(m.cv2); emitC(m.cv3)
def emit_psablock(p):                              # yolo26 P5 head block (== C2PSA attention unit)
    a = p.attn; emitC(a.qkv); emitC(a.proj); emitC(a.pe); emitC(p.ffn[0]); emitC(p.ffn[1])
def emit_c3k2_block(mm):
    t = type(mm).__name__
    if t == "C3k": emit_c3k(mm)
    elif t == "Sequential":                        # yolo26 L22: [Bottleneck, PSABlock]
        for c in mm:
            emit_psablock(c) if type(c).__name__ == "PSABlock" else emit_bott(c)
    else: emit_bott(mm)                            # plain Bottleneck
def emit_c3k2(m):
    emitC(m.cv1)
    for mm in m.m: emit_c3k2_block(mm)
    emitC(m.cv2)
def emit_sppf(m): emitC(m.cv1); emitC(m.cv2)
def emit_c2psa(m):
    emitC(m.cv1)
    for psa in m.m:
        a = psa.attn; emitC(a.qkv); emitC(a.proj); emitC(a.pe); emitC(psa.ffn[0]); emitC(psa.ffn[1])
    emitC(m.cv2)
def emit_seq(s):
    for x in s: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
def emit_branch(cv2, cv3):                         # one (box, cls) branch across the 3 levels
    for i in range(3):
        for x in cv2[i]: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)   # box: Conv,Conv,Conv2d -> 4
        for x in cv3[i]:                                                     # cls: [DW,Conv],[DW,Conv],Conv2d -> nc
            if isinstance(x, nn.Sequential): emit_seq(x)
            elif isinstance(x, nn.Conv2d): emitP(x)
            else: emitC(x)
def emit_detect26(det):
    emit_branch(det.cv2, det.cv3)                                           # one2many (training)
    if getattr(det, "end2end", False):
        emit_branch(det.one2one_cv2, det.one2one_cv3)                       # one2one (NMS-free)

EMIT = {"Conv": emitC, "C3k2": emit_c3k2, "SPPF": emit_sppf, "C2PSA": emit_c2psa}
for mod in L[:-1]:
    fn = EMIT.get(type(mod).__name__)
    if fn: fn(mod)
det = L[-1]; emit_detect26(det)

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]
import numpy as np; blob = []
for i, (w, b, k, s, p, g, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    blob.append(w.detach().cpu().numpy().ravel()); blob.append(b.detach().cpu().numpy().ravel())
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {p} {g} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")
np.concatenate(blob).astype(np.float32).tofile(os.path.join(D, "weights.bin"))

# arch26.txt: C3k2 stages (n c3k inner) + C2PSA repeats  (drives the shared net11-style forward)
_arch, _psa = [], 1
for _m in L:
    t = type(_m).__name__
    if t == "C3k2":
        c3k = is_c3k(_m.m[0]); inn = len(_m.m[0].m) if c3k else 0
        _arch.append((len(_m.m), 1 if c3k else 0, inn))
    elif t == "C2PSA": _psa = len(_m.m)
open(os.path.join(D, "arch26.txt"), "w").write(f"{_psa}\n" + "\n".join(f"{n} {c} {i}" for n,c,i in _arch) + "\n")
# head meta: nc reg_max end2end
open(os.path.join(D, "head26.txt"), "w").write(f"{det.nc} {det.reg_max} {1 if det.end2end else 0}\n")

# fixed input + reference head outputs (raw, per level & per branch) for C++ parity
torch.manual_seed(0); x = torch.rand(1, 3, IMG, IMG)
save("input.bin", x)
feats = []
def hook(m, i, o): feats.append(i[0] if isinstance(i, tuple) else i)  # not used; compute head manually below
# run backbone/neck to get the 3 detect inputs (det.f = [16,19,22])
y, saved = [], {}
xt = x
for m in L[:-1]:
    if m.f != -1:
        xt = saved[m.f] if isinstance(m.f, int) else [xt if j == -1 else saved[j] for j in m.f]
    xt = m(xt)
    saved[m.i] = xt
p3, p4, p5 = (saved[j] for j in det.f)
levels = [p3, p4, p5]
def run_branch(cv2, cv3, tag):
    for i, feat in enumerate(levels):
        bx = cv2[i](feat); cl = cv3[i](feat)
        save(f"ref_{tag}_box{i}.bin", bx); save(f"ref_{tag}_cls{i}.bin", cl)
        if i == 0: print(f"  {tag} L{i}: box{tuple(bx.shape)} cls{tuple(cl.shape)}")
run_branch(det.cv2, det.cv3, "o2m")
if det.end2end: run_branch(det.one2one_cv2, det.one2one_cv3, "o2o")
print(f"exported {len(convs)} convs, imgsz={IMG}, arch26 + head26 + input + refs -> {os.path.relpath(D)}")
print("head:", f"nc={det.nc} reg_max={det.reg_max} end2end={det.end2end}")
