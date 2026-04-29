from types import SimpleNamespace

import flight_batch


def test_create_flight_record_normalizes_crop_and_persists(monkeypatch):
    captured = {}

    def fake_create_flight_doc(flight_id, payload):
        captured["flight_id"] = flight_id
        captured["payload"] = payload

    monkeypatch.setattr(flight_batch.production_store, "create_flight_doc", fake_create_flight_doc)
    monkeypatch.setattr(flight_batch.uuid, "uuid4", lambda: SimpleNamespace(hex="flight123"))

    response = flight_batch.create_flight_record(
        device_id="drone-01",
        field_id="field-alpha",
        crop_type="banana",
        operator_notes="test flight",
    )

    assert response["flight_id"] == "flight123"
    assert response["crop_type"] == "rice"
    assert "Unsupported crop" in response["crop_warning"]
    assert response["status"] == "awaiting_upload"
    assert captured["payload"]["field_id"] == "field-alpha"


def test_complete_flight_upload_processes_synchronously_without_queue(monkeypatch):
    monkeypatch.setattr(
        flight_batch,
        "_require_flight",
        lambda flight_id: {"flight_id": flight_id, "status": "awaiting_upload"},
    )
    monkeypatch.setattr(flight_batch.production_store, "list_flight_images", lambda flight_id: [{"image_id": "patch-00001"}])
    monkeypatch.setattr(flight_batch.queue_client, "queue_provider_name", lambda: "none")
    monkeypatch.setattr(flight_batch, "process_flight_job", lambda flight_id: {"flight_id": flight_id, "status": "completed"})
    monkeypatch.setattr(flight_batch, "get_flight_status", lambda flight_id: {"flight_id": flight_id, "status": "completed"})

    response = flight_batch.complete_flight_upload("flight123")

    assert response["status"] == "completed"
