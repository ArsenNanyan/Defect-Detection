"""STEP 10 — Streamlit demo for the Surface Defect Inspector.

Uploads an image, calls the FastAPI /predict endpoint (no second copy of model state),
and shows original vs annotated views, a predictions table, latency, and an experiment
metrics panel loaded read-only from artifacts/reports.

Run (API must be up first):
    streamlit run ui/streamlit_app.py
Env: INSPECTOR_API_URL (default http://127.0.0.1:8000)
"""
from __future__ import annotations

import html
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

# Palette for detection boxes (cycles per class label).
BOX_COLORS = [
    "#f43f5e", "#f59e0b", "#22d3ee", "#a78bfa", "#34d399", "#fb7185",
]

st.set_page_config(
    page_title="Surface Defect Inspector",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg: #0b0f1a;
  --panel: #141a2a;
  --panel-2: #1a2235;
  --border: rgba(255,255,255,0.08);
  --text: #e6e9f0;
  --muted: #9aa4bd;
  --brand: #6366f1;
  --brand-2: #8b5cf6;
  --accent: #22d3ee;
  --ok: #34d399;
  --warn: #f59e0b;
  --bad: #f43f5e;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
  background:
    radial-gradient(1100px 600px at 12% -5%, rgba(99,102,241,0.16), transparent 60%),
    radial-gradient(900px 600px at 95% 0%, rgba(34,211,238,0.10), transparent 55%),
    var(--bg);
}

/* Trim the default top padding so the hero sits high */
.block-container { padding-top: 2.2rem; max-width: 1280px; }

/* Hide Streamlit chrome for a cleaner app feel */
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

/* ---------- Hero ---------- */
.hero {
  position: relative;
  border-radius: 22px;
  padding: 30px 34px;
  margin-bottom: 26px;
  background:
    linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.12) 45%, rgba(34,211,238,0.10)),
    var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 20px 60px -25px rgba(99,102,241,0.55), inset 0 1px 0 rgba(255,255,255,0.05);
  overflow: hidden;
}
.hero::after {
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(500px 220px at 85% -20%, rgba(34,211,238,0.18), transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 12px; font-weight: 600; letter-spacing: .4px;
  text-transform: uppercase; color: var(--accent);
  background: rgba(34,211,238,0.10);
  border: 1px solid rgba(34,211,238,0.25);
  padding: 5px 12px; border-radius: 999px; margin-bottom: 14px;
}
.hero h1 {
  font-size: 2.35rem; font-weight: 800; line-height: 1.1; margin: 0;
  background: linear-gradient(90deg, #ffffff, #c7d2fe 55%, #a5f3fc);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1.02rem; margin: 10px 0 0; max-width: 640px; }
.hero-chips { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  font-size: 12.5px; font-weight: 500; color: var(--text);
  background: rgba(255,255,255,0.05); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 10px;
}
.chip b { color: var(--accent); font-weight: 600; }

/* ---------- Generic card ---------- */
.card {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--border);
  border-radius: 18px; padding: 20px 22px;
  box-shadow: 0 16px 40px -30px rgba(0,0,0,0.9);
}
.section-title {
  font-size: 1.05rem; font-weight: 700; color: var(--text);
  margin: 6px 0 14px; display: flex; align-items: center; gap: 9px;
}
.section-title .dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: linear-gradient(135deg, var(--brand), var(--accent));
  box-shadow: 0 0 10px rgba(99,102,241,0.8);
}

/* ---------- Status pill ---------- */
.status {
  display: flex; align-items: center; gap: 11px;
  padding: 12px 15px; border-radius: 13px; font-weight: 600; font-size: 0.92rem;
  border: 1px solid var(--border);
}
.status .pulse { width: 10px; height: 10px; border-radius: 50%; position: relative; }
.status.ok  { background: rgba(52,211,153,0.10); color: var(--ok);  border-color: rgba(52,211,153,0.3); }
.status.bad { background: rgba(244,63,94,0.10);  color: var(--bad); border-color: rgba(244,63,94,0.3); }
.status.ok  .pulse { background: var(--ok); box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pulse 1.8s infinite; }
.status.bad .pulse { background: var(--bad); }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.6); }
  70%  { box-shadow: 0 0 0 9px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
.model-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 0.9rem;
}
.model-row:last-child { border-bottom: none; }
.model-row span.k { color: var(--muted); }
.model-row span.v { font-weight: 600; }
.model-row span.v.yes { color: var(--ok); }
.model-row span.v.no  { color: var(--bad); }

/* ---------- Metric tiles ---------- */
.metric-card {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--border); border-radius: 16px;
  padding: 18px 20px; height: 100%;
  position: relative; overflow: hidden;
}
.metric-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--brand), var(--accent));
}
.metric-card .label {
  font-size: 0.78rem; font-weight: 600; letter-spacing: .4px;
  text-transform: uppercase; color: var(--muted);
}
.metric-card .value {
  font-size: 1.9rem; font-weight: 800; color: var(--text); margin-top: 6px;
  font-feature-settings: "tnum"; letter-spacing: -0.5px;
}
.metric-card .value small { font-size: 0.95rem; font-weight: 600; color: var(--muted); }
.metric-card .sub {
  margin-top: 4px; font-size: 0.82rem; color: var(--accent); font-weight: 600;
}

/* Verdict banner */
.verdict {
  display: flex; align-items: center; gap: 14px;
  border-radius: 16px; padding: 18px 22px; margin: 4px 0 2px;
  border: 1px solid var(--border);
}
.verdict .icon { font-size: 1.7rem; }
.verdict .t { font-size: 1.15rem; font-weight: 700; }
.verdict .s { font-size: 0.88rem; color: var(--muted); font-weight: 500; }
.verdict.clean { background: rgba(52,211,153,0.08);  border-color: rgba(52,211,153,0.28); }
.verdict.clean .t { color: var(--ok); }
.verdict.defect { background: rgba(244,63,94,0.08);  border-color: rgba(244,63,94,0.28); }
.verdict.defect .t { color: var(--bad); }

/* ---------- Buttons ---------- */
.stButton > button {
  background: linear-gradient(135deg, var(--brand), var(--brand-2)) !important;
  color: #fff !important; border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-size: 0.98rem !important;
  padding: 0.6rem 1.1rem !important;
  box-shadow: 0 12px 30px -12px rgba(99,102,241,0.9) !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
.stButton > button:hover:enabled { transform: translateY(-2px); box-shadow: 0 18px 38px -12px rgba(99,102,241,1) !important; }
.stButton > button:disabled { opacity: 0.45 !important; }

/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(180deg, var(--panel-2), var(--panel)) !important;
  border: 1.5px dashed rgba(99,102,241,0.45) !important;
  border-radius: 16px !important; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent) !important;
  background: rgba(99,102,241,0.06) !important;
}

/* ---------- Images ---------- */
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--border); }
[data-testid="stImage"] { text-align: center; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1322, #0a0e18) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* ---------- Tables, expander, misc ---------- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: 14px !important;
  background: var(--panel) !important;
}
code, .stCode, pre { font-family: 'JetBrains Mono', monospace !important; }
hr { border-color: var(--border); }
.footer-note {
  margin-top: 26px; padding: 14px 18px; border-radius: 13px;
  background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.25);
  color: #fcd9a6; font-size: 0.86rem; line-height: 1.5;
}
.pill-tag {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
  background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--muted);
  margin-right: 6px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-badge">● Live inference demo</div>
  <h1>Industrial Surface Defect Inspector</h1>
  <p>Upload a steel surface image to detect and classify manufacturing defects in
     real time, powered by a deep-learning pipeline served over a FastAPI backend.</p>
  <div class="hero-chips">
    <span class="chip">Dataset <b>NEU-DET steel</b></span>
    <span class="chip">Classifier <b>EfficientNet-B0</b></span>
    <span class="chip">Detector <b>YOLO26n</b></span>
    <span class="chip">Runtime <b>ONNX</b></span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, unit: str = "", sub: str = "") -> str:
    unit_html = f"<small>{html.escape(unit)}</small>" if unit else ""
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)} {unit_html}</div>{sub_html}</div>'
    )


# --------------------------------------------------------------------------- #
# Backend helpers (unchanged logic)
# --------------------------------------------------------------------------- #
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
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    labels: dict[str, str] = {}
    for det in detections:
        label = det["label"]
        color = labels.setdefault(label, BOX_COLORS[len(labels) % len(BOX_COLORS)])
        x0, y0, x1, y1 = det["xyxy"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f'{label} {det["confidence"]:.2f}'
        tw = d.textlength(tag, font=font) if hasattr(d, "textlength") else 8 * len(tag)
        ty = max(0, y0 - 18)
        d.rectangle([x0, ty, x0 + tw + 8, ty + 17], fill=color)
        d.text((x0 + 4, ty + 1), tag, fill="white", font=font)
    return shown


inject_css()
hero()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
health = api_health()
with st.sidebar:
    st.markdown('<div class="section-title"><span class="dot"></span>Service status</div>',
                unsafe_allow_html=True)
    if health:
        st.markdown(
            f'<div class="status ok"><span class="pulse"></span>API online · {html.escape(str(health["status"]))}</div>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            f'<div class="model-row"><span class="k">{html.escape(str(k))}</span>'
            f'<span class="v {"yes" if v else "no"}">{"ready" if v else "offline"}</span></div>'
            for k, v in health["models_ready"].items()
        )
        st.markdown(f'<div class="card" style="margin-top:12px">{rows}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status bad"><span class="pulse"></span>API unreachable</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Endpoint: {API}")
        st.code("uvicorn api.main:app --port 8000", language="bash")

    st.markdown('<div class="section-title" style="margin-top:22px">'
                '<span class="dot"></span>Inspection settings</div>',
                unsafe_allow_html=True)
    task = st.radio("Task", ["detection", "classification"],
                    format_func=lambda t: f"🎯 Detection" if t == "detection"
                    else "🏷️ Classification")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     disabled=(task != "detection"),
                     help="Minimum confidence for a detection to be shown.")

# --------------------------------------------------------------------------- #
# Main — upload & inspect
# --------------------------------------------------------------------------- #
st.markdown('<div class="section-title"><span class="dot"></span>Upload surface image</div>',
            unsafe_allow_html=True)
upload = st.file_uploader("Drag & drop a steel surface image",
                          type=["png", "jpg", "jpeg", "bmp"],
                          label_visibility="collapsed")

if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        st.error("Maximum upload size is 10 MB.")
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        run = st.button("⚡ Run inspection", type="primary", disabled=health is None,
                        use_container_width=True)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-title"><span class="dot"></span>Input</div>',
                        unsafe_allow_html=True)
            st.image(img, use_container_width=True)

        if run:
            with st.spinner("Running inference…"):
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
                    payload = None

            if payload is not None:
                if task == "detection":
                    dets = payload.get("detections", [])
                    with right:
                        st.markdown('<div class="section-title"><span class="dot"></span>'
                                    f'Annotated · {len(dets)} detection(s)</div>',
                                    unsafe_allow_html=True)
                        st.image(draw_boxes(img, dets), use_container_width=True)

                    if dets:
                        st.markdown(
                            '<div class="verdict defect"><span class="icon">⚠️</span>'
                            f'<div><div class="t">{len(dets)} defect(s) found</div>'
                            '<div class="s">Regions flagged above the confidence threshold.</div>'
                            '</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            [{"label": d["label"],
                              "confidence": round(d["confidence"], 3),
                              "xyxy (px)": [round(v, 1) for v in d["xyxy"]]} for d in dets],
                            use_container_width=True, hide_index=True)
                    else:
                        st.markdown(
                            '<div class="verdict clean"><span class="icon">✅</span>'
                            '<div><div class="t">No defects detected</div>'
                            '<div class="s">Nothing above the confidence threshold.</div>'
                            '</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    with right:
                        st.markdown('<div class="section-title"><span class="dot"></span>'
                                    'Classification</div>', unsafe_allow_html=True)
                        st.markdown(
                            metric_tile("Predicted class", str(payload["label"]),
                                        sub=f'{payload["confidence"]*100:.1f}% confidence'),
                            unsafe_allow_html=True)
                    probs = payload.get("probabilities", {})
                    if probs:
                        st.markdown('<div class="section-title" style="margin-top:18px">'
                                    '<span class="dot"></span>Class probabilities</div>',
                                    unsafe_allow_html=True)
                        st.bar_chart(probs, color="#6366f1")

                st.markdown('<div class="section-title" style="margin-top:20px">'
                            '<span class="dot"></span>Performance</div>',
                            unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.markdown(metric_tile("Model inference", str(payload["inference_ms"]), "ms"),
                            unsafe_allow_html=True)
                m2.markdown(metric_tile("End-to-end", str(payload["total_ms"]), "ms"),
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div style="margin-top:14px">'
                    f'<span class="pill-tag">request · {html.escape(str(payload["request_id"]))}</span>'
                    f'<span class="pill-tag">model · {html.escape(str(payload["model"]))}</span></div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Experiment metrics panel (read-only)
# --------------------------------------------------------------------------- #
with st.expander("📊  Experiment metrics  ·  from artifacts/reports"):
    det = REPORTS / "detection_metrics.json"
    onnx = REPORTS / "benchmark.json"
    cls = REPORTS / "classification_comparison.json"
    if cls.exists():
        st.subheader("Classification — top configs (val)")
        rows = json.loads(cls.read_text())[:5]
        st.dataframe([{k: row[k] for k in ("run", "val_macro_f1", "val_auroc_ovr",
                      "params_M", "ckpt_size_MB")} for row in rows],
                     use_container_width=True, hide_index=True)
    if det.exists():
        st.subheader("Detection — YOLO26n (test split)")
        st.json(json.loads(det.read_text())["results"]["test"])
    if onnx.exists():
        st.subheader("Latency (batch 1, p50)")
        b = json.loads(onnx.read_text())
        st.write(f"Device: {b['device']} · providers: {b['onnx_providers']}")
        st.json({"classifier": b["classifier"], "detector": b["detector"]})

st.markdown(
    '<div class="footer-note">⚠️ <b>Research / portfolio demo</b> on NEU-DET. '
    'Not validated for production inspection; a missed defect (false negative) is '
    'costlier than a false alarm.</div>',
    unsafe_allow_html=True,
)
