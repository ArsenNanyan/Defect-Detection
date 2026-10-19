# ONNX Export, Parity & Latency (Step 8)

Both deployment models exported to ONNX and verified against their PyTorch sources, then
benchmarked. Reproduce:

```bash
python -m inspection.export_onnx --n 30     # export + parity  (PYTHONPATH=src)
python scripts/benchmark.py --runs 100      # latency
```

## Exports

| Model | Source | ONNX | Size |
|---|---|---|---|
| Classifier | EfficientNet-B0 (full) | `artifacts/checkpoints/classifier.onnx` | 16.0 MB |
| Detector | YOLO26n | `artifacts/checkpoints/detector.onnx` | 9.8 MB |

The detector ONNX is loaded back through the **Ultralytics runtime** so box decoding + NMS
match training; raw ONNX tensors are not assumed to equal boxes.

## Parity (30 held-out test images)

| Model | Result |
|---|---|
| Classifier | top-1 agreement **100%**, max abs softmax diff **2e-6** |
| Detector | **0** detection-count mismatches, **0** label mismatches, max box-coord delta **0.08 px** |

Parity is effectively exact — ONNX is a faithful export of both models.

## Latency (batch 1, 10 warmup + 100 runs, RTX 3050 / i7-13620H)

ONNX Runtime provider: **CPUExecutionProvider** (onnxruntime-gpu not installed).

| Workload | CPU PyTorch p50 | CPU ONNX p50 | GPU PyTorch p50 | Speedup (ONNX vs CPU PyTorch) |
|---|---|---|---|---|
| Classifier inference | 37.6 ms | **4.7 ms** | 8.6 ms | **8.0×** |
| Detector end-to-end | 54.8 ms | **17.3 ms** | — | **3.2×** |

Classifier preprocess: ~1.1 ms (p50), ~930 img/s.

### Reading it
- **ONNX Runtime on CPU is the deployment win here**: 8× faster than CPU PyTorch for the
  classifier and even faster than GPU PyTorch for this small model (GPU kernel-launch
  overhead dominates at batch 1). This is why the serving path uses ONNX on CPU — it is
  fast, dependency-light, and needs no GPU in production.
- The detector's end-to-end time includes decode + letterbox preprocess + NMS, so it is the
  honest number a client would see; 17 ms ≈ 58 FPS on CPU is comfortable for inspection.

The FastAPI service (Step 9) serves `detector.onnx` and `classifier.onnx`; the Streamlit UI
(Step 10) shows these latencies alongside predictions.
