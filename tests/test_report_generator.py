import report_generator


def test_generate_report_fallback_for_flight_summary(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    summary = {
        "flight_id": "flight123",
        "crop_type": "rice",
        "processed_patches": 8,
        "failed_patches": 1,
        "patches_with_detections": 3,
        "disease_counts": {"blast": 2, "brown_spot": 1},
        "severity_distribution": {"moderate": 2, "mild": 1},
        "average_confidence_per_disease": {"blast": 0.81, "brown_spot": 0.67},
        "top_disease": "blast",
        "highest_severity": "moderate",
        "hotspot_candidates": [
            {
                "patch_index": 4,
                "gps": {"lat": 27.7, "lon": 85.3},
                "disease": "blast",
                "severity": "moderate",
                "confidence": 0.85,
            }
        ],
    }

    report = report_generator.generate_report(
        summary,
        {"flight_id": "flight123", "crop_type": "rice"},
        sensor_stats={"soil_moisture": 37},
    )

    assert report["entity_id"] == "flight123"
    assert "8 analyzed patches" in report["executive_summary"]
    assert any("Most common disease: blast" == item for item in report["key_indicators"])
    assert report["sensor_stats"]["soil_moisture"] == 37
    assert report["priority_zones"]
