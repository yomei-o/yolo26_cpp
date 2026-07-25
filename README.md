# yolo26_cpp — pure C++ YOLO26 (no PyTorch, no CMake) — WIP

Same approach as [yolov11_cpp](https://github.com/yomei-o/yolov11_cpp) /
[yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) etc.: a dependency-free C++ reverse-mode
autograd engine that reproduces the real Ultralytics model **exactly** (parity ~1e-5), with
training + inference and no Python at run time.

## Status — CPU train + infer WORKING
yolo26 was reverse-engineered from Ultralytics (see [pure/ref/ARCH.md](pure/ref/ARCH.md)):
**yolo26 = yolo11 backbone/neck + a new head**. Three deltas: (1) the neck's C3k2 config
(`arch26`), (2) the P5-head block is `Bottleneck + PSABlock` (the C2PSA attention unit), and
(3) the Detect head drops DFL (`reg_max=1`, direct 4-value box) and is `end2end` (a one2one
branch, NMS-free). Everything is built from parts the sibling repos already had.

Done & verified (pure C++, no Python at run time):
- **forward** (`net26.hpp`, fused): **exact parity** vs Ultralytics — `pure/m1_forward26.cpp` = `0.0e+00`.
- **BN-training forward** (`net26_unfused.hpp`): `pure/m6_unfused26.cpp` = `9.5e-7` MATCH.
- **CPU training** (`train_cli26.cpp` + `v26loss.hpp`): TAL + no-DFL CIoU/BCE loss; loss decreases,
  saves `last.pt`/`best.pt`. `make_init_pt` bootstraps init weights from the manifest (594 tensors).
- **CPU inference** (`detect26.cpp`): forward → direct box decode → NMS → draw.

```sh
python pure/ref/export_yolo26.py 64 && python pure/ref/export_unfused26.py 64   # weights+manifest (once)
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\make_init_pt.cpp && make_init_pt init26.pt rand x pure/ref/data_net/
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\train_cli26.cpp && train_cli26 <imgs> <imgs> 100 8 init26.pt 64
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\detect26.cpp   && detect26 best.pt img.jpg 64 0.25 out.png
```

## Roadmap
1. ✅ scaffold + arch inspection
2. ✅ `net26.hpp` / `net26_unfused.hpp` forward (fused + BN-train) — exact parity
3. ✅ no-DFL loss + TAL + `train_cli26` (CPU) + `detect26` (CPU)
4. **next:** train the **one2one** branch too → faithful NMS-free inference; val mAP in the CLI
5. all sizes n/s/m/l/x (arch26 per scale); real-data convergence + Ultralytics-parity check
6. (later) device/GPU (Thrust) + Eigen/cuDNN backends like the sibling repos

License: own code BSD-3-Clause; bundled deps keep their licenses (stb: public-domain/MIT).
