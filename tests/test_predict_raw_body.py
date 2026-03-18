import io

from fastapi.testclient import TestClient
from PIL import Image

import main


class _FakeResult:
    def __init__(self):
        self.boxes = []
        self.names = {}


class _FakeModel:
    def __call__(self, img, conf=0.3):
        return [_FakeResult()]


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), color=(64, 128, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_accepts_raw_jpeg_and_falls_back_to_rice(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_model", lambda crop: _FakeModel())
    monkeypatch.setattr(main, "DEBUG_DIR", tmp_path / "debug")
    client = TestClient(main.app)

    response = client.post(
        "/predict?crop=invalid&save_to_firebase=true",
        content=_jpeg_bytes(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["crop"] == "rice"
    assert response.json()["crop_requested"] == "invalid"
    assert response.json()["save_to_firebase_requested"] is True
    assert response.json()["debug_saved"] is True
    assert response.json()["debug_latest_image_url"] == "/debug/latest.jpg"
    assert response.json()["firebase_saved"] is False


def test_predict_rejects_empty_body(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda crop: _FakeModel())
    client = TestClient(main.app)

    response = client.post(
        "/predict?crop=rice&save_to_firebase=true",
        content=b"",
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body is empty"


def test_predict_form_reuses_same_processing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_model", lambda crop: _FakeModel())
    monkeypatch.setattr(main, "DEBUG_DIR", tmp_path / "debug-form")
    client = TestClient(main.app)

    response = client.post(
        "/predict_form?crop=wheat&save_to_firebase=true",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["crop"] == "wheat"
    assert response.json()["request_content_type"] == "image/jpeg"
