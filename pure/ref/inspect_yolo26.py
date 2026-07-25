"""Dump the yolo26 architecture so the pure-C++ forward can be written to match it EXACTLY.
Loads the arch from the .yaml (random init, NO weight download), prints the full layer graph
(index | from | type), each block's internal module repr, and the detection head. Run this on
a machine/Colab where `ultralytics` (with yolo26) is installed, and paste the whole output.

  pip install -U ultralytics
  python inspect_yolo26.py            # defaults to yolo26n
  python inspect_yolo26.py yolo26s    # other size
"""
import sys
try:
    import ultralytics, torch, torch.nn as nn
    from ultralytics import YOLO
except Exception as e:
    print("need ultralytics + torch:", e); sys.exit(1)

name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n"
print("ultralytics", getattr(ultralytics, "__version__", "?"), "| torch", torch.__version__)
try:
    model = YOLO(name + ".yaml").model          # architecture only, random init, no download
except Exception as e:
    print(f"could not build {name}.yaml: {e}")
    print("try a different name (yolo26.yaml / yolo26n.yaml / yolov26n.yaml), or `yolo settings` to see configs")
    sys.exit(1)

seq = model.model                                # nn.Sequential of layers, each has .i, .f, .type
scales = getattr(model, "yaml", {}).get("scales", None)
print(f"\n### {name}: {len(seq)} layers   scales={scales}")

print("\n=== layer graph:  i | from | type ===")
for m in seq:
    print(f"{m.i:2d}  from={str(m.f):>14}  {type(m).__name__}")

def conv_info(mod):
    outs = []
    for sub in mod.modules():
        if isinstance(sub, nn.Conv2d):
            outs.append(f"Conv2d(in={sub.in_channels},out={sub.out_channels},k={sub.kernel_size[0]},s={sub.stride[0]},p={sub.padding[0]},g={sub.groups})")
    return outs

print("\n=== per-layer conv channels ===")
for m in seq:
    ci = conv_info(m)
    print(f"{m.i:2d} {type(m).__name__}: {len(ci)} convs" + (("  " + " ".join(ci[:2]) + (" ..." if len(ci) > 2 else "")) if ci else ""))

print("\n=== unique block types — full module repr (one each) ===")
seen = set()
for m in seq:
    t = type(m).__name__
    if t not in seen:
        seen.add(t); print(f"\n----- {t} (layer {m.i}) -----"); print(m)

print("\n=== detection head (last layer) — full repr ===")
print(seq[-1])
head = seq[-1]
for attr in ("nc", "reg_max", "no", "stride", "end2end", "dfl", "cv2", "cv3"):
    if hasattr(head, attr):
        v = getattr(head, attr)
        print(f"head.{attr} = {v if not isinstance(v, nn.Module) else type(v).__name__}")
print("\n(paste ALL of the above)")
