import numpy as np, onnxruntime as ort, os
D="pure/ref/data_net/"
x=np.fromfile(D+"input.bin",dtype=np.float32).reshape(1,3,64,64)
sess=ort.InferenceSession("yolo26n.onnx", providers=["CPUExecutionProvider"])
outs={o.name:v for o,v in zip(sess.get_outputs(), sess.run(None,{"images":x}))}
worst=0
for tag in ("o2m","o2o"):
  for i in range(3):
    for which,nm in (("box","box"),("cls","cls")):
      o=outs[f"{tag}_{which}{i}"]
      ref=np.fromfile(D+f"ref_{tag}_{nm}{i}.bin",dtype=np.float32).reshape(o.shape)
      d=np.abs(o-ref).max(); worst=max(worst,d)
print(f"ONNX vs ultralytics ref: worst |d| = {worst:.3e}  {'MATCH' if worst<1e-3 else 'MISMATCH'}")
