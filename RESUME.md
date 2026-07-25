# RESUME — yolo26_cpp remaining work

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
2. **ONNX I/O.** Port `onnx.hpp` / `onnx_run.hpp` + an `export_onnx26` (like the sibling repos):
   export the trained net to ONNX and verify against onnxruntime. Handle the no-DFL box (4) +
   dual-branch / NMS-free head in the exported graph.
3. ✅ **Evaluation tool (val mAP).** `pure/yolo26.cpp` `val`/train run COCO-mAP over a dataset
   (letterbox eval, NMS-free o2o decode) → mAP@0.5 / mAP@0.5:0.95.
4. ✅ **Unified CLI `yolo26`.** `yolo26 <train|val|detect> [--flags]` reads a standard Ultralytics
   `data.yaml` (`--arch <dir>` for size). (Supersedes the standalone `train_cli26`/`detect26`.)
   Still TODO: an `export` subcommand once ONNX is in.
5. **All sizes n/s/m/l/x.** arch26 per scale (widths from the `scales` in the yaml); per-size
   arch dirs + init weights (like the sibling repos' `pure/ref/arch/<model>/`).
6. **Real-data convergence + parity.** Train COCO128 to sensible boxes; add a decoded-boxes
   parity check vs Ultralytics (ultralytics is available locally).
7. **Speed backends (later).** Device-resident Thrust engine (CPU/GPU one source) + optional
   Eigen (CPU) and cuDNN (GPU) conv backends, exactly like yolov8/5/11/x.
8. **Ship demo weights + Colab notebooks** (train→detect→show; parity) once converged.
