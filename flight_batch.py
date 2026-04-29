import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException

import production_store
import queue_client
from inference import decode_image_bytes, is_healthy_disease, normalize_crop, run_inference
from report_generator import generate_report

logger = logging.getLogger("agridrone.flight_batch")

FLIGHT_INFERENCE_CONFIDENCE = float(os.getenv("FLIGHT_INFERENCE_CONFIDENCE", "0.3"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_image_id(patch_index: int) -> str:
    return f"patch-{patch_index:05d}"


def build_mission_folder() -> str:
    """Build a per-mission Supabase Storage folder name.

    Format: mission_YYYYMMDD_HHMMSS_<short>
    Example: mission_20260429_143022_a3f9
    """
    now = datetime.now(timezone.utc)
    short = uuid.uuid4().hex[:4]
    return f"mission_{now.strftime('%Y%m%d_%H%M%S')}_{short}"


def _require_flight(flight_id: str) -> dict[str, Any]:
    flight = production_store.get_flight_doc(flight_id)
    if flight is None:
        raise HTTPException(status_code=404, detail=f"Flight '{flight_id}' not found")
    return flight


def _gps_payload(
    *,
    lat: Optional[float],
    lon: Optional[float],
    gps_fix: bool,
    altitude_m: Optional[float],
    heading_deg: Optional[float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"gps_fix": gps_fix}
    if lat is not None:
        payload["lat"] = lat
    if lon is not None:
        payload["lon"] = lon
    if altitude_m is not None:
        payload["altitude_m"] = altitude_m
    if heading_deg is not None:
        payload["heading_deg"] = heading_deg
    return payload


def _counts_from_patches(patches: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "uploaded": 0,
        "pending_analysis": 0,
        "processed": 0,
        "failed": 0,
        "with_detections": 0,
    }

    for patch in patches:
        upload_status = patch.get("upload_status")
        analysis_status = patch.get("analysis_status")
        if upload_status == "uploaded":
            counts["uploaded"] += 1
        if analysis_status == "pending":
            counts["pending_analysis"] += 1
        elif analysis_status == "completed":
            counts["processed"] += 1
            primary = patch.get("primary_detection")
            if primary and not is_healthy_disease(primary.get("disease")):
                counts["with_detections"] += 1
        elif analysis_status == "error":
            counts["failed"] += 1

    return counts


def _patch_summary(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": patch.get("image_id"),
        "patch_index": patch.get("patch_index"),
        "captured_at": patch.get("captured_at"),
        "gps": patch.get("gps"),
        "upload_status": patch.get("upload_status"),
        "analysis_status": patch.get("analysis_status"),
        "primary_detection": patch.get("primary_detection"),
        "storage_path": patch.get("storage_path"),
    }


def _hotspot_candidates_from_patches(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for patch in patches:
        primary = patch.get("primary_detection")
        gps = patch.get("gps")
        if not primary or not gps:
            continue
        if is_healthy_disease(primary.get("disease")):
            continue
        candidates.append(
            {
                "image_id": patch.get("image_id"),
                "patch_index": patch.get("patch_index"),
                "captured_at": patch.get("captured_at"),
                "gps": gps,
                "disease": primary.get("disease"),
                "confidence": primary.get("confidence"),
                "severity": primary.get("severity"),
            }
        )

    severity_rank = {"severe": 3, "moderate": 2, "mild": 1, "trace": 0, "none": -1}
    candidates.sort(
        key=lambda item: (
            severity_rank.get(item.get("severity", "none"), -1),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return candidates[:10]


def _build_flight_summary(
    flight_id: str,
    flight: dict[str, Any],
    patches: list[dict[str, Any]],
    sensor_stats: Optional[dict[str, Any]],
) -> dict[str, Any]:
    disease_counts: dict[str, int] = {}
    severity_distribution: dict[str, int] = {}
    average_confidence_sums: dict[str, float] = {}
    average_confidence_counts: dict[str, int] = {}

    processed = 0
    failed = 0
    with_detections = 0

    for patch in patches:
        analysis_status = patch.get("analysis_status")
        if analysis_status == "completed":
            processed += 1
            primary = patch.get("primary_detection")
            if primary and not is_healthy_disease(primary.get("disease")):
                with_detections += 1
                disease = primary["disease"]
                severity = primary["severity"]
                confidence = float(primary["confidence"])
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                average_confidence_sums[disease] = average_confidence_sums.get(disease, 0.0) + confidence
                average_confidence_counts[disease] = average_confidence_counts.get(disease, 0) + 1
        elif analysis_status == "error":
            failed += 1

    average_confidence_per_disease = {
        disease: round(average_confidence_sums[disease] / average_confidence_counts[disease], 4)
        for disease in average_confidence_sums
    }
    hotspot_candidates = _hotspot_candidates_from_patches(patches)
    top_disease = max(disease_counts, key=disease_counts.get) if disease_counts else None
    highest_severity = "none"
    if severity_distribution.get("severe"):
        highest_severity = "severe"
    elif severity_distribution.get("moderate"):
        highest_severity = "moderate"
    elif severity_distribution.get("mild"):
        highest_severity = "mild"
    elif severity_distribution.get("trace"):
        highest_severity = "trace"

    return {
        "flight_id": flight_id,
        "field_id": flight.get("field_id"),
        "device_id": flight.get("device_id"),
        "crop_type": flight.get("crop_type"),
        "processed_patches": processed,
        "failed_patches": failed,
        "patches_with_detections": with_detections,
        "disease_counts": disease_counts,
        "severity_distribution": severity_distribution,
        "average_confidence_per_disease": average_confidence_per_disease,
        "top_disease": top_disease,
        "highest_severity": highest_severity,
        "hotspot_candidates": hotspot_candidates,
        "generated_at": _utc_now(),
        "sensor_snapshot_used": sensor_stats,
    }


def create_flight_record(
    *,
    device_id: str,
    field_id: str,
    crop_type: str,
    operator_notes: Optional[str],
    capture_interval_ms: Optional[int] = None,
    model_version_id: Optional[str] = None,
) -> dict[str, Any]:
    normalized_crop, crop_warning = normalize_crop(crop_type)
    flight_id = uuid.uuid4().hex
    now = _utc_now()
    storage_folder = build_mission_folder()
    payload = {
        "flight_id": flight_id,
        "device_id": device_id.strip(),
        "field_id": field_id.strip(),
        "crop_type": normalized_crop,
        "crop_requested": crop_type,
        "crop_warning": crop_warning,
        "operator_notes": operator_notes,
        "status": "awaiting_upload",
        "upload_status": "pending",
        "processing_status": "not_started",
        "queue_job_id": None,
        "capture_interval_ms": capture_interval_ms,
        "model_version_id": model_version_id,
        "storage_folder": storage_folder,
        "created_at": now,
        "updated_at": now,
    }
    production_store.create_flight_doc(flight_id, payload)
    logger.info(
        "flight.created",
        extra={
            "flight_id": flight_id,
            "device_id": device_id,
            "field_id": field_id,
            "storage_folder": storage_folder,
        },
    )
    return payload


def upload_flight_image(
    *,
    flight_id: str,
    raw_bytes: bytes,
    content_type: str,
    patch_index: int,
    captured_at: str,
    lat: Optional[float],
    lon: Optional[float],
    gps_fix: bool,
    altitude_m: Optional[float],
    heading_deg: Optional[float],
    device_id: Optional[str] = None,
    field_id: Optional[str] = None,
    crop_type: Optional[str] = None,
) -> dict[str, Any]:
    flight = _require_flight(flight_id)
    if flight.get("status") in {"queued", "processing", "completed"}:
        raise HTTPException(status_code=409, detail="Flight upload is closed for new images")
    if patch_index < 0:
        raise HTTPException(status_code=400, detail="patch_index must be >= 0")

    image_id = build_image_id(patch_index)
    decode_target = decode_image_bytes(raw_bytes)
    gps = _gps_payload(
        lat=lat,
        lon=lon,
        gps_fix=gps_fix,
        altitude_m=altitude_m,
        heading_deg=heading_deg,
    )

    storage_folder = flight.get("storage_folder")
    storage_path = production_store.upload_flight_image_bytes(
        flight_id=flight_id,
        image_id=image_id,
        raw_bytes=raw_bytes,
        content_type=content_type or "image/jpeg",
        storage_folder=storage_folder,
        patch_index=patch_index,
        metadata={
            "flight_id": flight_id,
            "device_id": device_id or flight.get("device_id") or "",
            "field_id": field_id or flight.get("field_id") or "",
            "crop_type": crop_type or flight.get("crop_type") or "",
            "image_id": image_id,
            "patch_index": str(patch_index),
            "captured_at": captured_at,
            "storage_folder": storage_folder or "",
        },
    )

    existing = production_store.get_flight_image_doc(flight_id, image_id) or {}
    patch_doc = {
        "image_id": image_id,
        "patch_index": patch_index,
        "captured_at": captured_at,
        "gps": gps,
        "upload_status": "uploaded",
        "analysis_status": existing.get("analysis_status", "pending"),
        "storage_path": storage_path,
        "storage_folder": storage_folder,
        "content_type": content_type or "image/jpeg",
        "image_size": {"width": decode_target.width, "height": decode_target.height},
        "crop_type": crop_type or flight.get("crop_type"),
        "uploaded_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    production_store.upsert_flight_image_doc(flight_id, image_id, patch_doc)

    patches = production_store.list_flight_images(flight_id)
    image_counts = _counts_from_patches(patches)
    production_store.update_flight_doc(
        flight_id,
        {
            "status": "uploading",
            "upload_status": "uploading",
            "processing_status": "not_started",
            "updated_at": _utc_now(),
            "image_counts": image_counts,
        },
    )
    return {
        "flight_id": flight_id,
        "image_id": image_id,
        "patch_index": patch_index,
        "status": "uploading",
        "upload_status": "uploaded",
        "processing_status": "not_started",
        "image_counts": image_counts,
    }


def complete_flight_upload(flight_id: str) -> dict[str, Any]:
    flight = _require_flight(flight_id)
    patches = production_store.list_flight_images(flight_id)
    if not patches:
        raise HTTPException(status_code=400, detail="Cannot queue processing before at least one image is uploaded")

    if flight.get("status") in {"queued", "processing", "completed"}:
        return get_flight_status(flight_id)

    if queue_client.queue_provider_name() == "none":
        logger.info("flight.processing_sync", extra={"flight_id": flight_id})
        process_flight_job(flight_id)
        return get_flight_status(flight_id)

    queue_job = queue_client.enqueue_flight_processing(flight_id)
    image_counts = _counts_from_patches(patches)
    production_store.update_flight_doc(
        flight_id,
        {
            "status": "queued",
            "upload_status": "complete",
            "processing_status": "queued",
            "queue_job_id": queue_job.id,
            "queued_at": _utc_now(),
            "updated_at": _utc_now(),
            "image_counts": image_counts,
        },
    )
    return {
        "flight_id": flight_id,
        "status": "queued",
        "upload_status": "complete",
        "processing_status": "queued",
        "queue_job_id": queue_job.id,
        "image_counts": image_counts,
    }


def get_flight_status(flight_id: str) -> dict[str, Any]:
    flight = _require_flight(flight_id)
    patches = production_store.list_flight_images(flight_id)
    image_counts = _counts_from_patches(patches)
    response = dict(flight)
    response["image_counts"] = image_counts
    response["patches"] = [_patch_summary(patch) for patch in patches]
    return response


def _build_patch_error(
    *,
    flight_id: str,
    image_id: str,
    patch: dict[str, Any],
    error_code: str,
    message: str,
) -> None:
    production_store.upsert_flight_image_doc(
        flight_id,
        image_id,
        {
            "analysis_status": "error",
            "error_code": error_code,
            "error_message": message,
            "updated_at": _utc_now(),
            "processed_at": _utc_now(),
        },
    )
    logger.warning(
        "flight.patch_error",
        extra={"flight_id": flight_id, "image_id": image_id, "error_code": error_code, "message": message},
    )


def process_flight_job(flight_id: str) -> dict[str, Any]:
    flight = _require_flight(flight_id)
    production_store.update_flight_doc(
        flight_id,
        {
            "status": "processing",
            "processing_status": "processing",
            "processing_started_at": _utc_now(),
            "updated_at": _utc_now(),
        },
    )

    patches = production_store.list_flight_images(flight_id)
    if not patches:
        production_store.update_flight_doc(
            flight_id,
            {
                "status": "error",
                "processing_status": "error",
                "error_code": "no_images",
                "error_message": "Flight has no uploaded images",
                "updated_at": _utc_now(),
            },
        )
        raise RuntimeError(f"Flight '{flight_id}' has no uploaded images")

    crop_type = flight.get("crop_type", "rice")
    normalized_crop, crop_warning = normalize_crop(crop_type)

    for patch in patches:
        image_id = patch["image_id"]
        if patch.get("analysis_status") == "completed":
            continue
        storage_path = patch.get("storage_path")
        if not storage_path:
            _build_patch_error(
                flight_id=flight_id,
                image_id=image_id,
                patch=patch,
                error_code="missing_storage_path",
                message="Patch is missing storage_path",
            )
            continue

        try:
            raw_bytes, content_type = production_store.download_storage_bytes(storage_path)
            image = decode_image_bytes(raw_bytes)
            detections, primary = run_inference(image, normalized_crop, FLIGHT_INFERENCE_CONFIDENCE)
            production_store.upsert_flight_image_doc(
                flight_id,
                image_id,
                {
                    "analysis_status": "completed",
                    "crop_type": normalized_crop,
                    "crop_warning": crop_warning,
                    "detections": detections,
                    "primary_detection": primary,
                    "detection_count": len(detections),
                    "content_type": content_type,
                    "processed_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "image_size": {"width": image.width, "height": image.height},
                    "highest_severity": primary["severity"] if primary else "none",
                },
            )
        except Exception as exc:
            _build_patch_error(
                flight_id=flight_id,
                image_id=image_id,
                patch=patch,
                error_code="patch_processing_failed",
                message=str(exc),
            )

    patches = production_store.list_flight_images(flight_id)
    image_counts = _counts_from_patches(patches)
    sensor_stats = production_store.get_latest_field_sensor_snapshot(flight.get("field_id"))
    summary = _build_flight_summary(flight_id, flight, patches, sensor_stats)
    report = generate_report(summary, flight, sensor_stats=sensor_stats)
    final_status = "completed" if image_counts["processed"] > 0 else "error"
    final_processing_status = "completed" if final_status == "completed" else "error"

    production_store.write_flight_summary(flight_id, summary)
    production_store.write_flight_report(flight_id, report)
    production_store.update_flight_doc(
        flight_id,
        {
            "status": final_status,
            "upload_status": "complete",
            "processing_status": final_processing_status,
            "image_counts": image_counts,
            "latest_sensor_snapshot": sensor_stats,
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
            "error_code": None if final_status == "completed" else "flight_processing_failed",
            "error_message": None if final_status == "completed" else "No patches were processed successfully",
        },
    )
    logger.info(
        "flight.processed",
        extra={
            "flight_id": flight_id,
            "status": final_status,
            "processed": image_counts["processed"],
            "failed": image_counts["failed"],
        },
    )
    return {
        "flight_id": flight_id,
        "status": final_status,
        "summary": summary,
        "report": report,
        "image_counts": image_counts,
    }
