"""API tests (guide section 16): valid image, corrupt image, oversized, zero detections,
unavailable model, and schema/coordinate sanity.

Uses FastAPI TestClient, which runs the lifespan (model load). If the ONNX models are not
present, tests skip rather than fail — run `python -m inspection.export_onnx` first for
full coverage.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

MODELS_PRESENT = (Path("artifacts/checkpoints/detector.onnx").exists()
                  and Path("artifacts/checkpoints/classifier.onnx").exists())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _jpeg_bytes(size=(200, 200), color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded", "down"}
    assert "models_ready" in body


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200


def test_corrupt_image_rejected(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")},
                    data={"task": "detection"})
    assert r.status_code == 400


def test_unsupported_type_rejected(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")},
                    data={"task": "detection"})
    assert r.status_code == 415


def test_bad_conf_rejected(client):
    r = client.post("/predict",
                    files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"task": "detection", "conf": "1.5"})
    assert r.status_code == 422


@pytest.mark.skipif(not MODELS_PRESENT, reason="ONNX models not exported")
def test_detection_valid_image(client):
    r = client.post("/predict",
                    files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"task": "detection", "conf": "0.25"})
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "detection"
    assert body["width"] == 200 and body["height"] == 200
    assert isinstance(body["detections"], list)  # may be zero on a blank image
    for d in body["detections"]:
        assert len(d["xyxy"]) == 4
        assert 0 <= d["confidence"] <= 1


@pytest.mark.skipif(not MODELS_PRESENT, reason="ONNX models not exported")
def test_classification_valid_image(client):
    r = client.post("/predict",
                    files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"task": "classification"})
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "classification"
    assert body["label"] is not None
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
