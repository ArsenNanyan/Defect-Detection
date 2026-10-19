"""STEP 9 — FastAPI inference service for the Surface Defect Inspector.

Endpoints
---------
GET  /health   model readiness
GET  /models   available models / tasks metadata
POST /predict  multipart image + task(classification|detection) + conf -> typed JSON

Hardening: content-type sniff, 10 MB body cap, decompression-bomb pixel guard, explicit
error handling, process-wide model caching (loaded once at startup), request IDs, and no
persistence of uploads. Coordinate convention is explicit pixel-space xyxy.

Run (from repo root, with src on PYTHONPATH):
    set PYTHONPATH=src
    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import io
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from inspection.inference import OnnxClassifier, OnnxDetector

MAX_BYTES = 10_000_000
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load models once; record failures instead of crashing so /health can report them
    errors = {}
    try:
        STATE["classifier"] = OnnxClassifier()
    except Exception as e:  # noqa: BLE001
        errors["classifier"] = str(e)
    try:
        STATE["detector"] = OnnxDetector()
    except Exception as e:  # noqa: BLE001
        errors["detector"] = str(e)
    STATE["errors"] = errors
    yield
    STATE.clear()


app = FastAPI(title="Surface Defect Inspector", version="0.1.0", lifespan=lifespan)


class Task(str, Enum):
    classification = "classification"
    detection = "detection"


class Health(BaseModel):
    status: str
    models_ready: dict[str, bool]
    errors: dict[str, str]


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    xyxy: list[float]  # pixel-space [x_min, y_min, x_max, y_max]


class PredictResponse(BaseModel):
    request_id: str
    width: int
    height: int
    model: str
    task: Task
    label: str | None = None
    class_id: int | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    detections: list[Detection] | None = None
    conf_threshold: float | None = None
    inference_ms: float
    total_ms: float


@app.get("/health", response_model=Health)
def health() -> Health:
    ready = {"classifier": "classifier" in STATE, "detector": "detector" in STATE}
    status = "ok" if all(ready.values()) else ("degraded" if any(ready.values()) else "down")
    return Health(status=status, models_ready=ready, errors=STATE.get("errors", {}))


@app.get("/models")
def models() -> dict:
    out = {}
    if "classifier" in STATE:
        c: OnnxClassifier = STATE["classifier"]  # type: ignore[assignment]
        out["classification"] = {"model": c.model_id, "image_size": c.image_size,
                                 "classes": c.class_names}
    if "detector" in STATE:
        d: OnnxDetector = STATE["detector"]  # type: ignore[assignment]
        out["detection"] = {"model": d.model_id, "imgsz": d.imgsz,
                            "classes": d.class_names, "coord": "pixel xyxy"}
    return out


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    task: Task = Form(Task.detection),
    conf: float = Form(0.25),
) -> PredictResponse:
    req_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content-type: {file.content_type}")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")
    if not raw:
        raise HTTPException(400, "Empty upload")
    try:
        im = Image.open(io.BytesIO(raw))
        if (im.width * im.height) > MAX_PIXELS:
            raise ValueError("too many pixels")
        im = im.convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Invalid or corrupt image") from None

    if not (0.0 <= conf <= 1.0):
        raise HTTPException(422, "conf must be in [0,1]")

    key = task.value if task != Task.classification else "classifier"
    key = "detector" if task == Task.detection else "classifier"
    if key not in STATE:
        raise HTTPException(503, f"{task.value} model unavailable")

    if task == Task.classification:
        r = STATE["classifier"].predict(im)  # type: ignore[attr-defined]
    else:
        r = STATE["detector"].predict(im, conf=conf)  # type: ignore[attr-defined]

    total = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        request_id=req_id, width=im.width, height=im.height,
        model=r["model"], task=task,
        label=r.get("label"), class_id=r.get("class_id"),
        confidence=r.get("confidence"), probabilities=r.get("probabilities"),
        detections=r.get("detections"), conf_threshold=r.get("conf_threshold"),
        inference_ms=r["inference_ms"], total_ms=round(total, 2),
    )
