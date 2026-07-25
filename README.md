# yolo26_cpp — pure C++ YOLO26 (no PyTorch, no CMake) — WIP

Same approach as [yolov11_cpp](https://github.com/yomei-o/yolov11_cpp) /
[yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) etc.: a dependency-free C++ reverse-mode
autograd engine that reproduces the real Ultralytics model **exactly** (parity ~1e-5), with
training + inference and no Python at run time.

## Status
**Bootstrapping.** The shared, model-agnostic engine is in place (autograd / conv-bn-silu /
dataset+augmentation / Adam / TAL / mAP / `.pt` I/O, copied from yolov11_cpp). The **yolo26-specific
forward topology + detection head + loss** are next — they are written to match the *real* yolo26
architecture, extracted from Ultralytics (not guessed).

## Next step: extract the real yolo26 architecture
Run this where `ultralytics` (with yolo26) is installed — Colab is fine — and share the output:
```sh
pip install -U ultralytics
python pure/ref/inspect_yolo26.py            # prints the full layer graph + head
```
It dumps the layer graph (`i | from | type`), each block's internals, and the detection head
(NMS-free/end2end? DFL removed?). From that the C++ `net26` forward + loss/decode are written to
match, then a CPU train + infer CLI (mirroring `train_cli`/`yolo` in the sibling repos).

## Roadmap
1. **[here]** scaffold + arch inspection
2. `net26.hpp` / `net26_unfused.hpp` — forward matching real yolo26 (fused + BN-training)
3. loss + label assignment (yolo26 head: likely NMS-free, DFL-free) + decode/NMS-free inference
4. `make_init_pt` + `train_cli` (CPU training) + `yolo` detect CLI
5. parity check vs Ultralytics reference forward
6. (later) all sizes n/s/m/l/x; device/GPU (Thrust) + Eigen/cuDNN backends like the sibling repos

License: own code BSD-3-Clause; bundled deps keep their licenses (stb: public-domain/MIT).
