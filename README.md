# Industrial Surface Defect Inspector

End-to-end, experiment-driven portfolio project for **steel surface defect inspection**:
image classification, object detection, segmentation/anomaly localization, ONNX
deployment, a FastAPI inference service, and a Streamlit demo.

> Status: **complete** — all 10 milestones done (see table at the bottom). Core pipeline
> runs end-to-end on an RTX 3050 6 GB laptop.

## Results at a glance

| Task | Model | Headline test metric | Size (ONNX) | CPU latency (p50) |
|---|---|---|---|---|
| Classification | EfficientNet-B0 | macro-F1 **0.986**, AUROC 0.9995 | 16 MB | 4.7 ms |
| Detection | YOLO26n | mAP@0.5 **0.782**, mAP@0.5:0.95 0.449 | 9.8 MB | 17 ms (e2e) |
| Anomaly (MVTec grid) | PatchCore | image AUROC **0.927**, pixel AUROC 0.962 | — | — |
| Segmentation (MVTec grid) | U-Net *(nonstandard)* | Dice 0.421, clean-FP 0.0 | — | — |

ONNX on CPU is **8× / 3.2×** faster than CPU PyTorch (classifier / detector). All metrics
are on frozen, leakage-audited test splits; NEU-DET and MVTec are never ranked against
each other. **Key finding:** *crazing* is the easiest class to classify (F1 1.0) yet the
hardest to localize (AP 0.54) — classification answers *what*, detection answers *where*.

## System architecture

```
          steel image (upload)
                 │
        ┌────────▼─────────┐   Streamlit UI (ui/streamlit_app.py)
        │  task + conf     │   original│annotated view, latency, metrics panel
        └────────┬─────────┘
                 │ HTTP multipart
        ┌────────▼─────────┐   FastAPI (api/main.py): /health /models /predict
        │  inference layer │   OnnxClassifier + OnnxDetector (src/inspection/inference.py)
        └───┬─────────┬────┘
            │         │
   classifier.onnx  detector.onnx       (exported + parity-checked, Step 8)
   EfficientNet-B0  YOLO26n
```
Segmentation/anomaly (MVTec) runs offline via `scripts/run_seg.py`, kept separate from the
NEU-DET serving path.

## Datasets

| Dataset | Role | Task | License | In repo? |
|---|---|---|---|---|
| [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | Primary | 6-class classification + bounding-box detection | Verify on original host | No (gitignored) |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Secondary | Segmentation / anomaly localization | CC BY-NC-SA 4.0 (non-commercial) | No (gitignored) |

**NEU-DET**: 1,800 images, 200×200 grayscale steel textures, Pascal VOC XML boxes, six
balanced classes (300 each): `crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`.

> **Important:** NEU-DET does *not* provide pixel segmentation masks. MVTec AD is a
> *separate* dataset used only for the segmentation/anomaly track. The two datasets are
> **not** a single unified benchmark and results are never ranked across them.

Raw images are kept out of Git. Only split manifests under `data/splits/` are committed.

## Six defect classes

| Class | Meaning |
|---|---|
| crazing | Network-like cracks |
| inclusion | Embedded impurity |
| patches | Surface patches |
| pitted_surface | Small localized pits |
| rolled-in_scale | Rolled oxide scale |
| scratches | Linear scratches |

## Repository layout

```
configs/        classification.yaml, detection.yaml, segmentation.yaml
data/           raw/ processed/ mvtec/ splits/   (only splits/*.csv committed)
src/inspection/ data, transforms, models, train_cls/eval_cls, inference, export_onnx,
                mvtec_data, patchcore, seg_models, train_seg
scripts/        audit_data, repair_raw, make_splits, run_cls_sweep, voc_to_yolo,
                train_det, benchmark, feature_lab, run_seg, failure_gallery
api/            main.py (FastAPI)
ui/             streamlit_app.py
tests/          test_splits.py, test_api.py
docs/model_cards/  classifier.md, detector.md, segmentation.md
artifacts/      reports/ (committed) · figures/ · checkpoints/ (large/dataset files ignored)
```

## Setup

```bash
python3.11 -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA build of torch for your GPU first (see below), then:
python -m pip install -r requirements.txt
```

### PyTorch + CUDA (this machine: RTX 3050 6 GB laptop)

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

## Reproduce

```bash
# Step 1 — audit + immutable splits
python scripts/repair_raw.py  --raw data/raw/neu_det
python scripts/audit_data.py  --raw data/raw/neu_det --out artifacts/reports
python scripts/make_splits.py --raw data/raw/neu_det --out data/splits --seed 42

# Step 2 — classification sweep (needs PYTHONPATH=src)
python scripts/run_cls_sweep.py --epochs 20

# Steps 4-5 — detection
python scripts/voc_to_yolo.py --raw data/raw/neu_det --splits data/splits --out data/processed/neu_det
python scripts/train_det.py   --config configs/detection.yaml   # --resume to continue

# Step 8 — ONNX export + parity + benchmark
python -m inspection.export_onnx --n 30
python scripts/benchmark.py --runs 100
```

### Run the demo (Steps 9-10)

```bash
# Terminal 1 — API (PYTHONPATH=src)
uvicorn api.main:app --host 127.0.0.1 --port 8000
# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501. The UI calls the API's `/predict`; set
`INSPECTOR_API_URL` if the API is elsewhere. No image handy? Use the built-in **sample
buttons**.

📖 **Step-by-step walkthrough with screenshots: [docs/USAGE.md](docs/USAGE.md)**

![Detection in the Streamlit demo](docs/screenshots/03_detection.jpg)

### Segmentation (Step 6) — MVTec AD setup

MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)** and must be downloaded manually (it is
not redistributed here and stays gitignored):

1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad and accept the license.
2. Download **one category** (e.g. `grid`) — or the full `mvtec_anomaly_detection.tar.xz`.
3. Extract so the layout is:
   ```
   data/mvtec/grid/train/good/*.png
   data/mvtec/grid/test/<defect>/*.png
   data/mvtec/grid/ground_truth/<defect>/*_mask.png
   ```
4. Run both tracks (PYTHONPATH=src):
   ```bash
   python scripts/run_seg.py --config configs/segmentation.yaml
   # PatchCore only:  python scripts/run_seg.py --skip-unet
   ```

- **Track A (PatchCore)** — benchmark-compatible, normal-only; reports image/pixel AUROC.
- **Track B (U-Net)** — *explicitly NONSTANDARD* supervised split of the test defects;
  reports Dice/IoU + clean false-positive rate. Never ranked against NEU-DET.

## Hardware used

- GPU: NVIDIA GeForce RTX 3050 6 GB Laptop GPU
- CPU: Intel Core i7-13620H (10C/16T) · RAM: 16 GB · Windows 11

## Milestones

| Step | Status |
|---|---|
| 1. Data audit + immutable splits | ✅ done — see [DATA_REPORT](artifacts/reports/DATA_REPORT.md) |
| 2. Classification (CNN, ResNet, EfficientNet) | ✅ done — 10-run sweep, see [CLASSIFICATION_REPORT](artifacts/reports/CLASSIFICATION_REPORT.md) |
| 3. Feature maps / receptive field lab | ✅ done — CNN RF 18px vs ResNet 435px, see [FEATURE_MAP_REPORT](artifacts/reports/FEATURE_MAP_REPORT.md) |
| 4. VOC→YOLO conversion | ✅ done (4,189 boxes, 0 rejected, same frozen split) |
| 5. YOLO26n detection | ✅ done — test mAP@0.5 0.782, see [DETECTION_REPORT](artifacts/reports/DETECTION_REPORT.md) |
| 6. MVTec segmentation / anomaly | ✅ done (grid) — PatchCore img/pix AUROC 0.93/0.96, see [SEGMENTATION_REPORT](artifacts/reports/SEGMENTATION_REPORT.md) |
| 7. Evaluation + model selection | ✅ done — test macro-F1 0.986 (cls) / mAP50 0.782 (det), see [EVALUATION_REPORT](artifacts/reports/EVALUATION_REPORT.md) |
| 8. ONNX export + parity | ✅ done — exact parity, 8× CPU speedup, see [ONNX_REPORT](artifacts/reports/ONNX_REPORT.md) |
| 9. FastAPI service | ✅ done — /health, /models, /predict (cls+det), 7 tests pass |
| 10. Streamlit demo | ✅ done — uploader, boxes, latency, metrics panel |

## Reports & model cards

- **Model cards:** [classifier](docs/model_cards/classifier.md) ·
  [detector](docs/model_cards/detector.md) · [segmentation](docs/model_cards/segmentation.md)
- **Reports:** [DATA](artifacts/reports/DATA_REPORT.md) ·
  [CLASSIFICATION](artifacts/reports/CLASSIFICATION_REPORT.md) ·
  [FEATURE_MAP](artifacts/reports/FEATURE_MAP_REPORT.md) ·
  [DETECTION](artifacts/reports/DETECTION_REPORT.md) ·
  [SEGMENTATION](artifacts/reports/SEGMENTATION_REPORT.md) ·
  [EVALUATION](artifacts/reports/EVALUATION_REPORT.md) ·
  [ONNX](artifacts/reports/ONNX_REPORT.md)
- **Qualitative failures:** `scripts/failure_gallery.py` →
  `artifacts/reports/figures/failures/` (4/278 classification errors; 121/278 detection
  images with a false positive or miss).

## Limitations (read before any real use)

- **No clean/defective screening:** NEU-DET is defect-focused, so the classifier/detector
  cannot certify a surface as defect-free. For that, use the normal-only anomaly track.
- **Coarse localization:** 200×200 source images; *crazing* in particular is poorly
  localized (detector AP 0.54). False negatives (missed defects) are the costly error in
  inspection — tune thresholds for recall before deployment.
- **Not production-validated / not calibrated:** small academic datasets, uncalibrated
  confidences, no OOD/abstention. Revalidate on real line data before trusting any output.
- **Segmentation U-Net is explicitly nonstandard** (self-split MVTec test defects) and is an
  educational baseline only; PatchCore is the benchmark-compatible number.

## License & attribution notes

- NEU-DET and MVTec AD images are **not** redistributed here; download them from the
  official sources and respect their licenses (MVTec AD is non-commercial).
- Ultralytics YOLO is AGPL-3.0 / Enterprise — review obligations before any commercial use.
