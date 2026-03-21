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


class _FakeFirebase:
    def __init__(self):
        self.missions = {}

    def firebase_is_configured(self):
        return True

    def create_mission(self, mission_id, payload):
        self.missions[mission_id] = payload

    def get_mission(self, mission_id):
        return self.missions.get(mission_id)

    def set_mission_status(self, mission_id, status, extra_fields=None):
        self.missions.setdefault(mission_id, {})
        self.missions[mission_id]["status"] = status
        if extra_fields:
            self.missions[mission_id].update(extra_fields)

    def list_mission_images(self, mission_id):
        return self.missions[mission_id].get("_images", [])

    def write_image_result(self, mission_id, image_id, yolo_result_json):
        self.missions[mission_id].setdefault("_results", {})[image_id] = yolo_result_json

    def write_mission_report(self, mission_id, report_json):
        self.missions[mission_id]["report"] = report_json

    def write_mission_summary(self, mission_id, summary_json):
        self.missions[mission_id]["summary"] = summary_json


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), color=(64, 128, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_accepts_raw_jpeg_and_falls_back_to_rice(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "process_prediction", main.process_prediction)
    monkeypatch.setattr("inference.get_model", lambda crop: _FakeModel())
    monkeypatch.setattr("inference.DEBUG_DIR", tmp_path / "debug")
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
    monkeypatch.setattr("inference.get_model", lambda crop: _FakeModel())
    client = TestClient(main.app)

    response = client.post(
        "/predict?crop=rice&save_to_firebase=true",
        content=b"",
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body is empty"


def test_predict_upload_alias_reuses_same_processing_path(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.get_model", lambda crop: _FakeModel())
    monkeypatch.setattr("inference.DEBUG_DIR", tmp_path / "debug-form")
    client = TestClient(main.app)

    response = client.post(
        "/predict_upload?crop=wheat&save_to_firebase=true",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["crop"] == "wheat"
    assert response.json()["request_content_type"] == "image/jpeg"


def test_mission_create_and_analyze(monkeypatch):
    fake_firebase = _FakeFirebase()
    client = TestClient(main.app)

    monkeypatch.setattr(main, "firebase_client", fake_firebase)
    monkeypatch.setattr(main, "_download_image_bytes", lambda url: (_jpeg_bytes(), "image/jpeg"))
    monkeypatch.setattr(main, "run_inference", lambda img, crop, confidence: ([{"disease": "blast", "confidence": 0.91, "bbox": [1, 2, 3, 4], "severity": "severe"}], {"disease": "blast", "confidence": 0.91, "bbox": [1, 2, 3, 4], "severity": "severe"}))

    create_response = client.post("/missions", json={"crop": "rice", "capture_interval_ms": 3000})
    assert create_response.status_code == 200
    mission_id = create_response.json()["missionId"]

    fake_firebase.missions[mission_id]["_images"] = [
        {
            "imageId": "img-1",
            "url": "https://example.com/test.jpg",
            "timestamp": "2026-03-21T00:00:00Z",
            "meta": {"gps": {"lat": 27.7, "lng": 85.3}},
            "uploaded": True,
            "raw": {},
        }
    ]

    analyze_response = client.post(f"/missions/{mission_id}/analyze")
    assert analyze_response.status_code == 200
    payload = analyze_response.json()
    assert payload["status"] == "done"
    assert payload["report"]["severity_assessment"] == "high"

    mission_response = client.get(f"/missions/{mission_id}")
    assert mission_response.status_code == 200
    assert mission_response.json()["status"] == "done"
