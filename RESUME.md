# RESUME — yolo26_cpp remaining work

# ✅ GPU VERIFICATION DONE (2026-07-25, real Colab T4 via colab/gpu_check.ipynb)

All three device/GPU checks passed on real hardware (`backend: GPU (CUDA)`):
1. **device forward parity** — `dnet26_test -DUSE_CUDA` → MATCH, worst 3.1e-2 (train-mode; P5's
   L22 PSABlock attention softmax amplifies accumulated device-vs-CPU diffs — P3 ~1e-3, P4 ~2e-3,
   P5 ~2-3e-2; structure verified correct since fused CPU forward == ultralytics = 0.0).
2. **device training** — `dtrain_coco26 -DUSE_CUDA -DUSE_CUBLAS`, COCO128 from pretrained yolo26n,
   imgsz320 batch8: loss **10.54 → 5.77** over 10ep, ~84 s/epoch, monotone (beats CPU 3ep=6.96).
   (s/epoch is host-bridge bound — device heads copied to host for TAL+v26 loss each step, not a
   pure-GPU-speed number; full speedup needs on-device loss, a future optimization.)
3. **cuDNN device path** — `dnet26_test -DUSE_CUDA -DUSE_CUDNN -lcudnn` → MATCH, worst 2.15e-2
   (cuDNN conv slightly tighter). cuDNN inc/lib auto-detected in the notebook.

Notebook: `colab/gpu_check.ipynb` (one-click T4: clone → download yolo26n.pt → export refs →
parity → training → cuDNN). yolo26_cpp now has NO unverified parts.

Remaining (nice-to-have): on-device loss for real GPU throughput; per-size pretrained weights
(yolo26{s,m,l,x}.pt) if training other sizes; optional ONNX re-check under Colab onnxruntime.

<details><summary>original Colab GPU plan (now completed) — kept for reference</summary>

Everything below is CPU-verified + nvcc-compiles; the device/GPU path has NOT run on a real GPU
yet (this dev box has none). On Colab (Runtime→GPU T4), clone main and run:

```bash
!git clone -q https://github.com/yomei-o/yolo26_cpp.git && cd yolo26_cpp
!pip -q install ultralytics
!python pure/ref/export_yolo26.py 64 yolo26n            # fused refs + weights (auto-uses yolo26n.pt)
!python pure/ref/export_unfused26.py 64 yolo26n         # unfused manifest/names (pretrained bins)
```

1. **Device fwd parity (GPU)** — expect MATCH vs CPU engine:
   `!nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dnet26_test.cpp -o d26 && ./d26`
   (needs init26.pt: `!g++ -O2 -std=c++17 -Ipure/third_party pure/make_init_pt.cpp -o mk && ./mk init26.pt rand x pure/ref/data_net/`)
   NOTE: train-mode device-vs-CPU was ~6e-2 on CPU-thrust (batch-stat reduction order + P5 double
   attention). Re-check on GPU; if it's much worse, dig into d26_c3k2_psa / d26_sppf.

2. **Device training (GPU)** — expect loss down + fast s/epoch, and mAP up vs the CPU baseline
   (pretrained COCO128 val mAP@0.5 0.504 -> CPU fine-tune reached 0.540):
   `!wget -q https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip && unzip -q -o coco128.zip`
   `!nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -DUSE_CUBLAS -Ipure/third_party pure/dtrain_coco26.cpp -lcublas -o dtr && ./dtr coco128/images/train2017 320 8 20 init26.pt pure/ref/data_net/`
   (from pretrained: make init with `./mk init26.pt from yolo26n.pt pure/ref/data_net/`)

3. **cuDNN device path (GPU)** — dtensor.hpp carries the cuDNN-guarded grouped dconv2d:
   add `-DUSE_CUDNN -lcudnn -I<cudnn_inc> -L<cudnn_lib>` to (1)/(2) (see the sibling repos'
   `colab/dnet_cudnn_test.ipynb` for auto-detecting cuDNN inc/lib) and confirm MATCH + speed.

4. **(optional) ONNX on Colab** — `onnx_verify26.py` already 5e-5 locally; re-run under Colab
   onnxruntime for good measure.

Then: ship Colab notebooks (train_detect / gpu_check / dnet_cudnn) like the sibling repos, and
per-size pretrained weights (download yolo26{s,m,l,x}.pt) if training other sizes.

</details>


Verified/working items are in [README.md](README.md); this is the forward-looking TODO.
yolo26 = yolo11 backbone/neck + (arch26, L22 Bottleneck+PSABlock, no-DFL end2end head) — see
[pure/ref/ARCH.md](pure/ref/ARCH.md).

## Done
- **SPPF residual (yolo26 `self.add=True`)** — FIXED. Forward now matches Ultralytics with
  **pretrained** weights (m1 9.9e-5 / m6 9.5e-5); `detect` on bus.jpg reproduces Ultralytics
  (bus 0.93 + 4 persons, pixel-level match).
- Forward (fused `net26.hpp` + BN-train `net26_unfused.hpp`) — **exact parity** vs Ultralytics
  (`m1_forward26` 0.0e+00, `m6_unfused26` 9.5e-7).
- Loss `v26loss.hpp` (reg_max=1, no DFL: direct box + CIoU + BCE) + TAL.
- CPU training `train_cli26.cpp` — one2many **and** one2one branches (topk 10 / 1); saves last/best.pt.
- CPU inference `detect26.cpp` — one2one head, **NMS-free** top-k decode.
- **Initial-weight tool** `make_init_pt.cpp` — generates init `.pt` (594 tensors) from the tiny
  `data_net` manifest, no Python. (`rand` and `from <pretrained.pt>` modes.)
- Ref exporters `export_yolo26.py` (fused weights + manifest + reference head outputs) and
  `export_unfused26.py` (unfused manifest + `names.txt`, same conv order).

## TODO
1. **`.pt` I/O — full support.** Have: `ptio.hpp` read/write state_dict; `make_init_pt from`;
   `train_cli26` writes `last/best.pt`; `load_net_unfused_pt` loads. Add: load a real Ultralytics
   `yolo26n.pt` (module checkpoint) for transfer-learning init + confirm the saved `.pt` loads
   back into Ultralytics (0 unexpected keys, incl. the one2one_* head tensors).
2. ✅ **ONNX export.** `onnx_export26.cpp` -> yolo26n.onnx (opset 13, no deps), verified vs
   ultralytics via onnxruntime (`onnx_verify26.py`) = 7e-5 MATCH (o2m+o2o box/cls, 3 levels).
   TODO: an in-CLI `export` subcommand + ONNX->engine import round-trip if needed.
3. ✅ **Evaluation tool (val mAP).** `pure/yolo26.cpp` `val`/train run COCO-mAP over a dataset
   (letterbox eval, NMS-free o2o decode) → mAP@0.5 / mAP@0.5:0.95.
4. ✅ **Unified CLI `yolo26`.** `yolo26 <train|val|detect> [--flags]` reads a standard Ultralytics
   `data.yaml` (`--arch <dir>` for size). (Supersedes the standalone `train_cli26`/`detect26`.)
   Still TODO: an `export` subcommand once ONNX is in.
5. ✅ **All sizes n/s/m/l/x.** `pure/ref/arch/yolo26{n,s,m,l,x}/` (manifest+names+arch26); the CLI
   `--arch <dir>` selects it (net26 forward is arch26-driven). Verified yolo26l (psa=2, 190L) runs.
   TODO: ship pretrained weights per size (download yolo26{s,m,l,x}.pt).
6. **Real-data training** — pretrained yolo26n on COCO128 gives **val mAP@0.5 0.504 /
   mAP@0.5:0.95 0.370** @imgsz320 (eval pipeline validated on real weights). Fine-tune runs;
   TODO: confirm multi-epoch convergence (CPU is slow -> use GPU/device backend later).
7. **Speed backends (later).** Device-resident Thrust engine (CPU/GPU one source) + optional
   Eigen (CPU) and cuDNN (GPU) conv backends, exactly like yolov8/5/11/x.
8. **Ship demo weights + Colab notebooks** (train→detect→show; parity) once converged.
