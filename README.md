# yolo26_cpp — pure C++ YOLO26 (no PyTorch, no CMake)

Same approach as [yolov11_cpp](https://github.com/yomei-o/yolov11_cpp) /
[yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) etc.: a dependency-free C++ reverse-mode
autograd engine that reproduces the real Ultralytics model **exactly** (parity ~1e-5), with
training + inference and no Python at run time. CPU **and** GPU (Thrust device engine, cuBLAS,
cuDNN) are all verified on real hardware.

## Build variants (same sources, backend chosen by compile flags)

The **host** engine (autograd on CPU) and the **device** engine (`dtensor.hpp`/`dnet26.hpp`, one
source that compiles for CPU *or* GPU) share the code; you pick the backend at build time:

| Variant | Engine | Compiler & flags | Notes |
|---|---|---|---|
| **CPU (host)** | host | `cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party` (MSVC) or `g++ -O2 -std=c++17 -Ipure/third_party` | default; no GPU needed. Used by `detect26`, `train_cli26`, `yolo26`, `demo26`. |
| **CPU + Eigen** | host | add `-DUSE_EIGEN -Ipure/third_party/eigen_flat` (+ `/arch:AVX2` or `-march=native`) | ~7× faster GEMM, same results (~8e-5). |
| **Device on CPU** | device | add `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I"%CUDA%/include/cccl" -I"%CUDA%/include"` | runs the *device* engine on the CPU (no GPU) — for developing/verifying the one source. |
| **GPU (CUDA)** | device | `nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party` | Thrust `device_vector`; real GPU. |
| **GPU + cuBLAS** | device | add `-DUSE_CUBLAS -lcublas` | cuBLAS GEMM fast path. |
| **GPU + cuDNN** | device | add `-DUSE_CUDNN -I<cudnn_inc> -L<cudnn_lib> -lcudnn` | cuDNN grouped conv fast path (`dtensor.hpp`). |

`-DUSE_CUDA`, `-DUSE_CUBLAS`, `-DUSE_CUDNN` compose (e.g. training with all three:
`nvcc … -DUSE_CUDA -DUSE_CUDNN -DUSE_CUBLAS … pure/dtrain_coco26.cpp -lcudnn -lcublas`). On a
build box with **no GPU**, the *host* variants and the *device-on-CPU* variant fully verify
correctness; the `-DUSE_CUDA` variants need a GPU (see the Colab notebooks below).

### Colab notebooks (`colab/`)
Runtime → GPU (T4), then Run all.
- **[gpu_check.ipynb](https://colab.research.google.com/github/yomei-o/yolo26_cpp/blob/main/colab/gpu_check.ipynb)**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yomei-o/yolo26_cpp/blob/main/colab/gpu_check.ipynb)
  — device forward parity, device training, and the cuDNN path on a T4.
- **[train_detect_cudnn.ipynb](https://colab.research.google.com/github/yomei-o/yolo26_cpp/blob/main/colab/train_detect_cudnn.ipynb)**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yomei-o/yolo26_cpp/blob/main/colab/train_detect_cudnn.ipynb)
  — train with cuDNN → infer with the trained weights → show the image.

## Status — CPU + GPU train + infer WORKING
yolo26 was reverse-engineered from Ultralytics (see [pure/ref/ARCH.md](pure/ref/ARCH.md)):
**yolo26 = yolo11 backbone/neck + a new head**. Three deltas: (1) the neck's C3k2 config
(`arch26`), (2) the P5-head block is `Bottleneck + PSABlock` (the C2PSA attention unit), and
(3) the Detect head drops DFL (`reg_max=1`, direct 4-value box) and is `end2end` (a one2one
branch, NMS-free). Everything is built from parts the sibling repos already had.

Done & verified (pure C++, no Python at run time):
- **forward** (`net26.hpp`, fused): **exact parity** vs Ultralytics — `pure/m1_forward26.cpp` = `0.0e+00`.
- **BN-training forward** (`net26_unfused.hpp`): `pure/m6_unfused26.cpp` = `9.5e-7` MATCH.
- **pretrained parity**: `yolo26 detect` on `bus.jpg` reproduces Ultralytics (bus 0.93 + 4 persons,
  NMS-free o2o) to ~1px/~0.01 conf. (The one non-obvious delta: SPPF has a residual, `self.add=True`.)
- **CPU training** (`train_cli26.cpp` + `v26loss.hpp`): dual head (one2many TAL topk=10 + one2one
  topk=1), no-DFL CIoU/BCE loss. Real fine-tune of pretrained yolo26n on COCO128: loss 10.86→6.96,
  val mAP@0.5 **0.504→0.540**. `make_init_pt` bootstraps init weights (594 tensors).
- **CPU inference** (`detect26.cpp`): forward → direct box decode → NMS-free o2o top-k → draw.
- **ONNX** (`yolo26 export` + `onnx_build26.hpp`): opset-13, verified vs onnxruntime (5e-5) and a
  pure-C++ runner (`m8_onnx_run26.cpp`, 9.9e-5).
- **all sizes** n/s/m/l/x (`--arch`, per-scale `arch26` dirs), **standalone `demo26`** (zero-arg).
- **GPU / device engine** (`dnet26.hpp`, Thrust) — verified on a real Colab T4:
  forward parity `MATCH` (3.1e-2, train-mode), device training loss **10.54→5.77** (COCO128, 10ep),
  and the **cuDNN** path `MATCH` (2.15e-2). CPU + Eigen backends also verified.

```sh
python pure/ref/export_yolo26.py 64 && python pure/ref/export_unfused26.py 64   # weights+manifest (once)
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\make_init_pt.cpp && make_init_pt init26.pt rand x pure/ref/data_net/
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\yolo26.cpp && yolo26 train <imgs> --epochs 100 --imgsz 640
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\detect26.cpp && detect26 best.pt img.jpg 640 0.25 out.png
```

## Roadmap
1. ✅ scaffold + arch inspection
2. ✅ `net26.hpp` / `net26_unfused.hpp` forward (fused + BN-train) — exact parity
3. ✅ no-DFL loss + TAL + dual-head training (`train_cli26`/`yolo26`) + NMS-free `detect26`
4. ✅ ONNX I/O, unified `yolo26 <train|val|detect|export>` CLI + val mAP, all sizes n/s/m/l/x
5. ✅ real-data convergence (COCO128 fine-tune, mAP 0.504→0.540) + Ultralytics pretrained parity
6. ✅ device/GPU (Thrust) + Eigen/cuBLAS/cuDNN backends — verified on a real T4
7. (nice-to-have) on-device loss for full GPU throughput; per-size pretrained weights

License: own code BSD-3-Clause; bundled deps keep their licenses (stb: public-domain/MIT).
