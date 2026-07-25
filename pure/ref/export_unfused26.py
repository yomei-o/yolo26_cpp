"""Export yolo26n WITHOUT folding BN, in the SAME conv order as export_yolo26.py (so the C++
net26_unfused consumes them identically), plus names.txt (state_dict keys) for make_init_pt /
checkpoint write-back. Manifest line: kind Co Ci k s pad groups eps act.
Usage: python export_unfused26.py [imgsz] [yolo26]"""
import os, sys, torch, torch.nn as nn
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
MODEL = sys.argv[2] if len(sys.argv) > 2 else "yolo26"
def is_c3k(m): return type(m).__name__ == "C3k"

import os as _os
_pt = MODEL + ".pt"
ym = YOLO(_pt) if _os.path.exists(_pt) else YOLO(MODEL + ".yaml")   # pretrained if present, else random
L = ym.model.model.eval()
print("loaded", _pt if _os.path.exists(_pt) else MODEL+".yaml")
qn = {id(m): nm for nm, m in ym.model.named_modules()}
lines, names, idx = [], [], [0]
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
def emitC(cv):                                   # ultralytics Conv/DWConv (conv+bn[+act])
    i = idx[0]; idx[0] += 1; conv, bn = cv.conv, cv.bn
    act = 1 if isinstance(cv.act, nn.SiLU) else 0; p = qn[id(cv)]
    save(f"cw{i}.bin", conv.weight); save(f"bg{i}.bin", bn.weight); save(f"bb{i}.bin", bn.bias)
    save(f"rm{i}.bin", bn.running_mean); save(f"rv{i}.bin", bn.running_var)
    names.extend([f"{p}.conv.weight", f"{p}.bn.weight", f"{p}.bn.bias", f"{p}.bn.running_mean", f"{p}.bn.running_var"])
    lines.append(f"1 {conv.weight.shape[0]} {conv.weight.shape[1]} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {conv.groups} {bn.eps} {act}")
def emitP(c):                                    # plain nn.Conv2d (has bias)
    i = idx[0]; idx[0] += 1; p = qn[id(c)]
    save(f"cw{i}.bin", c.weight); save(f"cb{i}.bin", c.bias)
    names.extend([f"{p}.weight", f"{p}.bias"])
    lines.append(f"0 {c.weight.shape[0]} {c.weight.shape[1]} {c.kernel_size[0]} {c.stride[0]} {c.padding[0]} {c.groups} 0 0")

def emit_bott(b): emitC(b.cv1); emitC(b.cv2)
def emit_c3k(m): emitC(m.cv1); [emit_bott(b) for b in m.m]; emitC(m.cv2); emitC(m.cv3)
def emit_psa(pb):
    a = pb.attn; emitC(a.qkv); emitC(a.proj); emitC(a.pe); emitC(pb.ffn[0]); emitC(pb.ffn[1])
def emit_block(mm):
    t = type(mm).__name__
    if t == "C3k": emit_c3k(mm)
    elif t == "Sequential": [emit_psa(c) if type(c).__name__=="PSABlock" else emit_bott(c) for c in mm]
    else: emit_bott(mm)
def emit_c3k2(m): emitC(m.cv1); [emit_block(mm) for mm in m.m]; emitC(m.cv2)
def emit_sppf(m): emitC(m.cv1); emitC(m.cv2)
def emit_c2psa(m):
    emitC(m.cv1)
    for pb in m.m: emit_psa(pb)
    emitC(m.cv2)
def emit_seq(s):
    for x in s: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
def emit_branch(cv2, cv3):
    for i in range(3):
        for x in cv2[i]: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
        for x in cv3[i]:
            if isinstance(x, nn.Sequential): emit_seq(x)
            elif isinstance(x, nn.Conv2d): emitP(x)
            else: emitC(x)

EMIT = {"Conv": emitC, "C3k2": emit_c3k2, "SPPF": emit_sppf, "C2PSA": emit_c2psa}
for mod in L[:-1]:
    fn = EMIT.get(type(mod).__name__)
    if fn: fn(mod)
det = L[-1]; emit_branch(det.cv2, det.cv3)
if getattr(det, "end2end", False): emit_branch(det.one2one_cv2, det.one2one_cv3)

open(os.path.join(D, "manifest_unfused.txt"), "w").write(f"{idx[0]}\n" + "\n".join(lines) + "\n")
open(os.path.join(D, "names.txt"), "w").write("\n".join(names) + "\n")
print(f"unfused: {idx[0]} layers, {len(names)} tensors -> {os.path.relpath(D)}")
