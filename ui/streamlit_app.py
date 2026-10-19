"""STEP 10 — Streamlit demo for the Surface Defect Inspector.

Uploads an image, calls the FastAPI /predict endpoint (no second copy of model state),
and shows original vs annotated views, a predictions table, latency, and an experiment
metrics panel loaded read-only from artifacts/reports.

Run (API must be up first):
    streamlit run ui/streamlit_app.py
Env: INSPECTOR_API_URL (default http://127.0.0.1:8000)
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API = os.getenv("INSPECTOR_API_URL", "http://127.0.0.1:8000")
REPORTS = Path("artifacts/reports")
MAX_BYTES = 10_000_000

st.set_page_config(page_title="Surface Defect Inspector", layout="wide")
st.title("🔩 Industrial Surface Defect Inspector")
st.caption("NEU-DET steel defects · EfficientNet-B0 classifier + YOLO26n detector (ONNX)")


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def draw_boxes(img: Image.Image, detections: list[dict]) -> Image.Image:
    shown = img.copy()
    d = ImageDraw.Draw(shown)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        x0, y0, x1, y1 = det["xyxy"]
        d.rectangle([x0, y0, x1, y1], outline="red", width=2)
        tag = f'{det["label"]} {det["confidence"]:.2f}'
        d.text((x0, max(0, y0 - 14)), tag, fill="red", font=font)
    return shown


# ---- sidebar ----
health = api_health()
with st.sidebar:
    st.header("Service")
    if health:
        st.success(f"API: {health['status']}")
        st.json(health["models_ready"])
    else:
        st.error(f"API unreachable at {API}")
        st.info("Start it:\n\n`uvicorn api.main:app --port 8000`")
    task = st.radio("Task", ["detection", "classification"])
    conf = st.slider("Confidence (detection)", 0.05, 0.95, 0.25, 0.05,
                     disabled=(task != "detection"))

# ---- main ----
upload = st.file_uploader("Upload a steel surface image", type=["png", "jpg", "jpeg", "bmp"])
if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        st.error("Maximum upload size is 10 MB.")
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        left, right = st.columns(2)
        left.image(img, caption="Input", use_container_width=True)

        if st.button("Run inspection", type="primary", disabled=health is None):
            try:
                r = requests.post(
                    f"{API}/predict",
                    files={"file": (upload.name, data, upload.type or "image/jpeg")},
                    data={"task": task, "conf": conf},
                    timeout=60,
                )
                r.raise_for_status()
                payload = r.json()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
            else:
                if task == "detection":
                    dets = payload.get("detections", [])
                    right.image(draw_boxes(img, dets), caption=f"{len(dets)} detections",
                                use_container_width=True)
                    if dets:
                        st.dataframe(
                            [{"label": d["label"], "confidence": round(d["confidence"], 3),
                              "xyxy (px)": [round(v, 1) for v in d["xyxy"]]} for d in dets],
                            use_container_width=True)
                    else:
                        st.info("No defects above the confidence threshold.")
                else:
                    right.metric("Predicted class", payload["label"],
                                 f'{payload["confidence"]*100:.1f}%')
                    probs = payload.get("probabilities", {})
                    st.bar_chart(probs)
                c1, c2 = st.columns(2)
                c1.metric("Model inference", f'{payload["inference_ms"]} ms')
                c2.metric("End-to-end", f'{payload["total_ms"]} ms')
                st.caption(f'request_id: {payload["request_id"]} · model: {payload["model"]}')

# ---- experiment metrics panel (read-only) ----
with st.expander("📊 Experiment metrics (from artifacts/reports)"):
    det = REPORTS / "detection_metrics.json"
    onnx = REPORTS / "benchmark.json"
    cls = REPORTS / "classification_comparison.json"
    if cls.exists():
        st.subheader("Classification — top configs (val)")
        rows = json.loads(cls.read_text())[:5]
        st.dataframe([{k: row[k] for k in ("run", "val_macro_f1", "val_auroc_ovr",
                      "params_M", "ckpt_size_MB")} for row in rows],
                     use_container_width=True)
    if det.exists():
        st.subheader("Detection — YOLO26n (test split)")
        st.json(json.loads(det.read_text())["results"]["test"])
    if onnx.exists():
        st.subheader("Latency (batch 1, p50)")
        b = json.loads(onnx.read_text())
        st.write(f"Device: {b['device']} · providers: {b['onnx_providers']}")
        st.json({"classifier": b["classifier"], "detector": b["detector"]})

st.caption("⚠️ Research/portfolio demo on NEU-DET. Not validated for production "
           "inspection; a missed defect (false negative) is costlier than a false alarm.")
