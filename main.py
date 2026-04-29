import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import requests

import firebase_client
import production_store
import queue_client
from firebase_client import FirebaseConfigError
from flight_batch import complete_flight_upload, create_flight_record, get_flight_status, upload_flight_image
from inference import (
    DEBUG_LATEST_DECODED_IMAGE_URL,
    DEBUG_LATEST_IMAGE_URL,
    VALID_CROPS,
    debug_path,
    decode_image_bytes,
    ensure_default_model,
    get_model,
    is_healthy_disease,
    list_model_details,
    normalize_crop,
    process_prediction,
    run_inference,
)
from report_generator import generate_report
from schemas import (
    DeviceRegisterRequest,
    DeviceResponse,
    DeviceStatusResponse,
    Esp32CameraControlRequest,
    Esp32FlightCaptureRequest,
    Esp32FlightCaptureResponse,
    Esp32SnapshotInferenceRequest,
    FieldFlightsResponse,
    FlightCompleteResponse,
    FlightCreateRequest,
    FlightCreateResponse,
    FlightImageUploadResponse,
    FlightStatusResponse,
    InferenceRecordResponse,
    PhotoRecordResponse,
)
from storage_helper import StorageDownloadError, download_and_validate_image

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agridrone.api")

app = FastAPI(
    title="AgriDrone Guardian API",
    description="Post-flight crop surveillance backend for AgriDrone Guardian",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONCURRENT_INFER = max(1, int(os.getenv("MAX_CONCURRENT_INFER", "1")))
MISSION_ANALYZE_TIMEOUT = float(os.getenv("MISSION_ANALYZE_TIMEOUT_SEC", "0.1"))
ANALYZE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFER)

# ── Device registry (ESP32 self-registration) ────────────────────────────────
# In-memory store keyed by device_id.  Persisted to Firebase RTDB so the
# registry survives server restarts.  Loaded from RTDB on startup.
_device_registry: dict[str, dict] = {}


def _load_device_registry_from_firebase() -> None:
    """Populate in-memory registry from RTDB on startup (best-effort)."""
    if not firebase_client.firebase_is_configured():
        return
    try:
        from firebase_admin import db as rtdb
        firebase_client.get_firebase_app()
        data = rtdb.reference("/devices").get() or {}
        loaded = {k: v for k, v in data.items() if isinstance(v, dict)}
        _device_registry.update(loaded)
        logger.info("device_registry.loaded_from_firebase", extra={"count": len(loaded)})
    except Exception as exc:
        logger.warning("device_registry.firebase_load_failed: %s", exc)


def _persist_device_to_firebase(device_id: str, record: dict) -> None:
    """Write a single device record to RTDB (best-effort)."""
    if not firebase_client.firebase_is_configured():
        return
    try:
        from firebase_admin import db as rtdb
        firebase_client.get_firebase_app()
        rtdb.reference(f"/devices/{device_id}").set(record)
    except Exception as exc:
        logger.warning("device_registry.firebase_write_failed device_id=%s: %s", device_id, exc)


def _sync_drone_ip_to_firebase(ip: str) -> None:
    """Mirror the ESP32 IP into drone/ip in RTDB so the Flutter live-stream
    screen (which watches the 'drone' node) auto-updates without any manual
    config."""
    if not firebase_client.firebase_is_configured():
        return
    try:
        from firebase_admin import db as rtdb
        firebase_client.get_firebase_app()
        rtdb.reference("/drone").update({"ip": ip, "status": "online"})
    except Exception as exc:
        logger.warning("drone_ip_sync.firebase_failed: %s", exc)


def _resolve_esp32_ip(esp32_ip: Optional[str], device_id: Optional[str]) -> str:
    """Return a usable ESP32 IP, resolved from the registry if only device_id given."""
    if esp32_ip:
        return esp32_ip
    if device_id:
        record = _device_registry.get(device_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{device_id}' is not registered. "
                       "Ensure the ESP32 calls POST /v1/devices/register on boot.",
            )
        return record["ip"]
    raise HTTPException(status_code=400, detail="Provide esp32_ip or device_id")


def _normalize_esp32_ip(raw: str) -> str:
    candidate = raw.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="ESP32 IP address is required")
    if "://" in candidate:
        parsed = urlparse(candidate)
        candidate = parsed.hostname or ""
    candidate = candidate.strip().strip("/").split("/")[0]
    if not candidate:
        raise HTTPException(status_code=400, detail="ESP32 IP address is invalid")
    return candidate


def _esp32_capture_base_candidates(raw: str) -> list[str]:
    candidate = raw.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="ESP32 IP address is required")

    if "://" in candidate:
        parsed = urlparse(candidate)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or ""
        if not host:
            raise HTTPException(status_code=400, detail="ESP32 IP address is invalid")
        bases = [f"{scheme}://{host}:{parsed.port}"] if parsed.port else [f"{scheme}://{host}"]
        if parsed.port != 81:
            bases.append(f"{scheme}://{host}:81")
        return list(dict.fromkeys(bases))

    host = candidate.strip().strip("/").split("/")[0]
    if not host:
        raise HTTPException(status_code=400, detail="ESP32 IP address is invalid")
    if ":" in host:
        return [f"http://{host}"]
    return [f"http://{host}", f"http://{host}:81"]


def _fetch_esp32_snapshot_bytes(esp32_ip: str) -> tuple[bytes, str]:
    attempts: list[str] = []
    last_error = "no capture endpoint attempted"
    for base_url in _esp32_capture_base_candidates(esp32_ip):
        url = f"{base_url}/capture"
        attempts.append(url)
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException as exc:
            last_error = f"{url}: {exc}"
            continue

        content_type = response.headers.get("content-type") or "image/jpeg"
        is_jpeg = response.content.startswith(b"\xff\xd8") if response.content else False
        is_image = content_type.lower().startswith("image/")
        if response.status_code == 200 and response.content and (is_image or is_jpeg):
            return response.content, content_type
        last_error = (
            f"{url}: HTTP {response.status_code}, "
            f"content-type={content_type}, bytes={len(response.content or b'')}"
        )

    raise HTTPException(
        status_code=502,
        detail=(
            "Could not fetch an ESP32 snapshot. Tried "
            f"{', '.join(attempts)}. Last error: {last_error}"
        ),
    )


def _storage_preview_url(request: Request, storage_path: Optional[str]) -> Optional[str]:
    if not storage_path:
        return None
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/v1/storage/image?path={quote(storage_path, safe='')}"


def _photo_response_from_patch(request: Request, patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(patch.get("id") or f"{patch.get('flight_id')}:{patch.get('image_id')}"),
        "flight_id": str(patch.get("flight_id") or ""),
        "image_id": str(patch.get("image_id") or ""),
        "patch_index": int(patch.get("patch_index") or 0),
        "captured_at": patch.get("captured_at"),
        "storage_path": patch.get("storage_path"),
        "storage_folder": patch.get("storage_folder"),
        "preview_url": _storage_preview_url(request, patch.get("storage_path")),
        "upload_status": patch.get("upload_status") or "unknown",
        "analysis_status": patch.get("analysis_status") or "unknown",
        "primary_detection": patch.get("primary_detection"),
        "detection_count": int(patch.get("detection_count") or 0),
        "highest_severity": patch.get("highest_severity"),
        "crop_type": patch.get("crop_type"),
        "content_type": patch.get("content_type"),
        "uploaded_at": patch.get("uploaded_at"),
        "processed_at": patch.get("processed_at"),
    }


@app.on_event("startup")
def startup() -> None:
    production_store.initialize()
    ensure_default_model()
    _load_device_registry_from_firebase()
    for crop in VALID_CROPS:
        try:
            model = get_model(crop)
            if model is not None:
                logger.info("startup.model_preloaded", extra={"crop": crop})
        except Exception as exc:
            logger.warning("startup.model_preload_failed: %s", exc, extra={"crop": crop})
    model_details = list_model_details()
    loaded_crops = sorted(detail["crop"] for detail in model_details if detail.get("loaded"))
    logger.info(
        "startup.models",
        extra={"models_loaded": loaded_crops, "max_concurrent_infer": MAX_CONCURRENT_INFER},
    )
    if firebase_client.firebase_is_configured():
        try:
            firebase_client.get_firebase_app()
            firebase_client.get_firestore_client()
            logger.info("startup.firebase_init", extra={"firebase_configured": True})
        except Exception as exc:
            logger.warning("startup.firebase_init_failed: %s", exc)
    else:
        logger.info("startup.firebase_init_skipped", extra={"firebase_configured": False})
    # Seed device registry from Firebase so it survives server restarts.
    seeded = _registry_from_firebase()
    _device_registry.update(seeded)
    logger.info("startup.device_registry_loaded", extra={"count": len(seeded)})
    logger.info(
        "startup.complete",
        extra={
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "production_backend_provider": production_store.provider_name(),
            "queue_provider": queue_client.queue_provider_name(),
        },
    )


@app.exception_handler(FirebaseConfigError)
async def firebase_config_exception_handler(_: Request, exc: FirebaseConfigError):
    return JSONResponse(
        status_code=503,
        content={"error_code": "firebase_not_configured", "message": str(exc)},
    )


@app.get("/")
@app.head("/")
def root():
    return {
        "status": "AgriDrone Guardian API is running",
        "version": "2.0.0",
        "crops_supported": VALID_CROPS,
    }


@app.get("/health")
def health():
    model_details = list_model_details()
    models_loaded = sorted(detail["crop"] for detail in model_details if detail.get("loaded"))
    available_models = sorted(detail["crop"] for detail in model_details if detail.get("ready"))
    production_health = production_store.health_summary()
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "models_available": available_models,
        "model_details": model_details,
        "production_backend_provider": production_health["provider"],
        "storage_mode": production_health["storage_mode"],
        "firebase_configured": production_health["firebase_configured"],
        "firestore_configured": production_health["firestore_configured"],
        "rtdb_configured": production_health["rtdb_configured"],
        "supabase_configured": production_health["supabase_configured"],
        "supabase_service_role_configured": production_health["supabase_service_role_configured"],
        "supabase_connection_ok": production_health["supabase_connection_ok"],
        "supabase_connection_error": production_health["supabase_connection_error"],
        "redis_configured": queue_client.redis_is_configured(),
        "sqs_configured": queue_client.sqs_is_configured(),
        "queue_provider": queue_client.queue_provider_name(),
        "iot_shadow_enabled": production_health["iot_shadow_enabled"],
        "aws_s3_bucket": production_health["aws_s3_bucket"],
        "max_concurrent_infer": MAX_CONCURRENT_INFER,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _supabase_bucket_name() -> str:
    return os.getenv("SUPABASE_STORAGE_BUCKET", "Crop-photos").strip() or "Crop-photos"


def _normalize_supabase_url(raw_url: str) -> str:
    return raw_url.rstrip("/").removesuffix("/rest/v1").rstrip("/")


@lru_cache(maxsize=1)
def _supabase_py_client():
    url = _normalize_supabase_url(os.getenv("SUPABASE_URL", "").strip())
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not service_role_key:
        raise HTTPException(
            status_code=503,
            detail="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for Supabase webhook processing",
        )

    try:
        from supabase import create_client
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="supabase-py is not installed. Install backend requirements before using the webhook.",
        ) from exc

    return create_client(url, service_role_key)


def _normalize_supabase_object_path(storage_path: str, bucket: str) -> str:
    normalized = storage_path.strip()
    if normalized.startswith("supabase://"):
        normalized = normalized[len("supabase://") :]
        bucket_prefix = f"{bucket}/"
        if normalized.startswith(bucket_prefix):
            normalized = normalized[len(bucket_prefix) :]
    return normalized.lstrip("/")


def _mission_error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": error_code, "message": message})


def _get_existing_mission_or_404(mission_id: str) -> dict[str, Any]:
    mission = firebase_client.get_mission(mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found")
    return mission


def _extract_gps(meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    gps = meta.get("gps")
    return gps if isinstance(gps, dict) else None


def _summarize_results(
    mission_id: str,
    mission: dict[str, Any],
    analyzed_images: list[dict[str, Any]],
    failed_images: list[dict[str, Any]],
) -> dict[str, Any]:
    counts_per_disease: dict[str, int] = {}
    severity_distribution: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}
    confidence_counts: dict[str, int] = {}
    images_with_detections = 0
    top_severe_detections: list[dict[str, Any]] = []

    for image in analyzed_images:
        primary = image.get("primary_detection")
        if not primary:
            continue
        disease = primary["disease"]
        if is_healthy_disease(disease):
            continue
        images_with_detections += 1
        severity = primary["severity"]
        confidence = float(primary["confidence"])
        counts_per_disease[disease] = counts_per_disease.get(disease, 0) + 1
        severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        confidence_sums[disease] = confidence_sums.get(disease, 0.0) + confidence
        confidence_counts[disease] = confidence_counts.get(disease, 0) + 1
        top_severe_detections.append(
            {
                "imageId": image["imageId"],
                "disease": disease,
                "confidence": round(confidence, 4),
                "severity": severity,
                "gps": image.get("gps"),
                "timestamp": image.get("timestamp"),
            }
        )

    top_severe_detections.sort(
        key=lambda item: (
            ["trace", "mild", "moderate", "severe"].index(item["severity"]),
            item["confidence"],
        ),
        reverse=True,
    )
    average_confidence_per_disease = {
        disease: round(confidence_sums[disease] / confidence_counts[disease], 4)
        for disease in confidence_sums
    }

    hotspot_candidates = [item for item in top_severe_detections if item.get("gps")][:5]

    return {
        "mission_id": mission_id,
        "crop": mission.get("crop", "rice"),
        "processed_images": len(analyzed_images),
        "failed_images": len(failed_images),
        "images_with_detections": images_with_detections,
        "counts_per_disease": counts_per_disease,
        "severity_distribution": severity_distribution,
        "average_confidence_per_disease": average_confidence_per_disease,
        "top_severe_detections": top_severe_detections[:5],
        "hotspot_candidates": hotspot_candidates,
        "generated_at": _utc_now(),
    }


def _analyze_mission_sync(mission_id: str) -> dict[str, Any]:
    mission = _get_existing_mission_or_404(mission_id)
    crop = mission.get("crop", "rice")
    normalized_crop, crop_warning = normalize_crop(crop)
    images = firebase_client.list_mission_images(mission_id)

    logger.info(
        "mission.analysis_begin",
        extra={"mission_id": mission_id, "crop": normalized_crop, "image_count": len(images)},
    )

    if not images:
        message = "Mission has no images to analyze. Add RTDB image records with storage_url first."
        firebase_client.set_mission_status(
            mission_id,
            "error",
            {
                "error_code": "no_images",
                "error_message": message,
                "error_reason": message,
                "updated_at": _utc_now(),
            },
        )
        raise HTTPException(status_code=400, detail=message)

    analyzed_images: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []

    for image in images:
        image_id = image["imageId"]
        if not image.get("url"):
            message = "Mission image is missing storage_url"
            error_payload = {
                "status": "error",
                "error_code": "missing_storage_url",
                "message": message,
                "processed_at": _utc_now(),
            }
            firebase_client.write_image_result(mission_id, image_id, error_payload)
            failed_images.append({"imageId": image_id, "error_code": "missing_storage_url", "message": message})
            continue
        try:
            raw_bytes, content_type = download_and_validate_image(image["url"])
            img = decode_image_bytes(raw_bytes)
            detections, primary = run_inference(img, normalized_crop, confidence=0.3)
            yolo_result = {
                "status": "success",
                "crop": normalized_crop,
                "crop_warning": crop_warning,
                "timestamp": image.get("timestamp"),
                "content_type": content_type,
                "image_size": {"width": img.width, "height": img.height},
                "detections": detections,
                "primary_detection": primary,
                "processed_at": _utc_now(),
            }
            firebase_client.write_image_result(mission_id, image_id, yolo_result)
            analyzed_images.append(
                {
                    "imageId": image_id,
                    "timestamp": image.get("timestamp"),
                    "gps": _extract_gps(image.get("meta", {})),
                    "detections": detections,
                    "primary_detection": primary,
                }
            )
        except HTTPException:
            raise
        except StorageDownloadError as exc:
            logger.exception(
                "mission.image_failed",
                extra={"mission_id": mission_id, "image_id": image_id, "url": image.get("url")},
            )
            error_payload = {
                "status": "error",
                "error_code": exc.error_code,
                "message": exc.message,
                "processed_at": _utc_now(),
            }
            firebase_client.write_image_result(mission_id, image_id, error_payload)
            failed_images.append({"imageId": image_id, "error_code": exc.error_code, "message": exc.message})
        except Exception as exc:
            logger.exception(
                "mission.image_failed",
                extra={"mission_id": mission_id, "image_id": image_id, "url": image.get("url")},
            )
            error_payload = {
                "status": "error",
                "error_code": "image_processing_failed",
                "message": str(exc),
                "processed_at": _utc_now(),
            }
            firebase_client.write_image_result(mission_id, image_id, error_payload)
            failed_images.append({"imageId": image_id, "error_code": "image_processing_failed", "message": str(exc)})

    failure_ratio = len(failed_images) / max(len(images), 1)
    summary = _summarize_results(mission_id, mission, analyzed_images, failed_images)
    summary["failure_ratio"] = round(failure_ratio, 4)
    summary["failed_image_details"] = failed_images

    report = generate_report(summary, {"missionId": mission_id, **mission})
    firebase_client.write_mission_summary(mission_id, summary)
    firebase_client.write_mission_report(mission_id, report)

    if failure_ratio > 0.3:
        error_reason = "More than 30% of mission images failed during analysis."
        firebase_client.set_mission_status(
            mission_id,
            "error",
            {
                "error_code": "too_many_failed_images",
                "error_message": error_reason,
                "error_reason": error_reason,
                "updated_at": _utc_now(),
                "processed_images": summary["processed_images"],
                "failed_images": summary["failed_images"],
            },
        )
        logger.error(
            "mission.analysis_error_threshold",
            extra={"mission_id": mission_id, "failure_ratio": failure_ratio},
        )
        return {
            "missionId": mission_id,
            "status": "error",
            "error_reason": error_reason,
            "report": report,
            "summary": summary,
        }

    firebase_client.set_mission_status(
        mission_id,
        "done",
        {
            "updated_at": _utc_now(),
            "processed_images": summary["processed_images"],
            "failed_images": summary["failed_images"],
        },
    )
    logger.info("mission.analysis_complete", extra={"mission_id": mission_id, "processed": len(analyzed_images)})
    return {"missionId": mission_id, "status": "done", "report": report, "summary": summary}


async def _run_mission_analysis(mission_id: str) -> dict[str, Any]:
    try:
        await asyncio.wait_for(ANALYZE_SEMAPHORE.acquire(), timeout=MISSION_ANALYZE_TIMEOUT)
    except TimeoutError:
        return {"busy": True}

    try:
        return await asyncio.to_thread(_analyze_mission_sync, mission_id)
    finally:
        ANALYZE_SEMAPHORE.release()


@app.get(DEBUG_LATEST_IMAGE_URL)
def debug_latest_image():
    latest_path = debug_path("latest.jpg")
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail="No debug image has been saved yet.")
    return FileResponse(latest_path, media_type="image/jpeg")


@app.get(DEBUG_LATEST_DECODED_IMAGE_URL)
def debug_latest_decoded_image():
    latest_decoded_path = debug_path("latest_decoded.jpg")
    if not latest_decoded_path.exists():
        raise HTTPException(status_code=404, detail="No decoded debug image has been saved yet.")
    return FileResponse(latest_decoded_path, media_type="image/jpeg")


@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Query(default="rice", description="Crop type: rice, wheat, maize, potato, tomato, pepper"),
    confidence: float = Query(default=0.3, description="Minimum confidence threshold"),
    save_to_backend: Optional[bool] = Query(default=None, description="Whether to persist results to the configured backend."),
    save_to_firebase: bool = Query(default=True, description="Legacy alias for save_to_backend."),
):
    body = await request.body()
    return process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_backend if save_to_backend is not None else save_to_firebase,
        request_content_type=request.headers.get("content-type", ""),
        source_kind="raw_request",
        source_label="predict",
    )


@app.post("/predict_form")
@app.post("/predict_upload")
async def predict_form(
    image: UploadFile = File(...),
    crop: str = Query(default="rice", description="Crop type: rice, wheat, maize, potato, tomato, pepper"),
    confidence: float = Query(default=0.3, description="Minimum confidence threshold"),
    save_to_backend: Optional[bool] = Query(default=None, description="Whether to persist results to the configured backend."),
    save_to_firebase: bool = Query(default=True, description="Legacy alias for save_to_backend."),
):
    body = await image.read()
    return process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_backend if save_to_backend is not None else save_to_firebase,
        request_content_type=image.content_type or "",
        source_kind="multipart_upload",
        source_label=image.filename or "upload",
    )

    detections, primary = await run_inference(img_np, crop, confidence)

@app.post("/v1/inferences/esp32-snapshot", response_model=InferenceRecordResponse)
async def infer_esp32_snapshot(payload: Esp32SnapshotInferenceRequest):
    raw_bytes, content_type = await asyncio.to_thread(
        _fetch_esp32_snapshot_bytes,
        payload.esp32_ip,
    )
    result = process_prediction(
        raw_bytes=raw_bytes,
        crop=payload.crop,
        confidence=payload.confidence,
        save_to_firebase=True,
        request_content_type=content_type,
        source_kind="esp32_snapshot",
        source_label=_normalize_esp32_ip(payload.esp32_ip),
    )
    return InferenceRecordResponse(**result)


@app.get("/v1/inferences", response_model=list[InferenceRecordResponse])
async def list_inferences(limit: int = Query(default=20, ge=1, le=100)):
    records = production_store.list_inference_docs(limit=limit)
    return [InferenceRecordResponse(**record) for record in records]


@app.get("/v1/inferences/{inference_id}", response_model=InferenceRecordResponse)
async def fetch_inference(inference_id: str):
    record = production_store.get_inference_doc(inference_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inference '{inference_id}' not found")
    return InferenceRecordResponse(**record)


@app.get("/v1/storage/image")
def fetch_storage_image(path: str = Query(..., min_length=1)):
    normalized_path = path.strip()
    if not normalized_path:
        raise HTTPException(status_code=400, detail="Storage path is required")

    try:
        raw_bytes, content_type = production_store.download_storage_bytes(normalized_path)
    except Exception as exc:
        logger.warning("storage.image_fetch_failed: %s", exc, extra={"storage_path": normalized_path})
        raise HTTPException(status_code=404, detail="Stored image could not be retrieved") from exc

    if not content_type.startswith("image/"):
        content_type = "image/jpeg"

    return Response(
        content=raw_bytes,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=60"},
    )


@app.post("/webhook/new_patch")
async def webhook_new_patch(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Webhook payload must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook payload must be a JSON object")

    record = payload.get("record")
    if not isinstance(record, dict):
        raise HTTPException(status_code=400, detail="Supabase webhook payload is missing record")

    patch_id = str(record.get("id") or "").strip()
    flight_id = str(record.get("flight_id") or "").strip()
    storage_path = str(record.get("storage_path") or record.get("image_path") or "").strip()

    if not patch_id:
        raise HTTPException(status_code=400, detail="Webhook record is missing id")
    if not storage_path:
        raise HTTPException(status_code=400, detail="Webhook record is missing storage_path")

    bucket = _supabase_bucket_name()
    object_path = _normalize_supabase_object_path(storage_path, bucket)
    client = _supabase_py_client()

    try:
        image_bytes = await asyncio.to_thread(
            client.storage.from_(bucket).download,
            object_path,
        )
    except Exception as exc:
        now = _utc_now()
        try:
            await asyncio.to_thread(
                client.table("flight_patches")
                .update(
                    {
                        "analysis_status": "error",
                        "error_code": "storage_download_failed",
                        "error_message": str(exc),
                        "updated_at": now,
                    }
                )
                .eq("id", patch_id)
                .execute
            )
        except Exception:
            logger.exception("webhook.patch_error_update_failed", extra={"patch_id": patch_id})
        raise HTTPException(status_code=502, detail=f"Could not download Supabase image: {exc}") from exc

    # TODO: Run YOLOv8 inference on image_bytes.
    disease_detected = "Mock Brown Spot"
    confidence = 0.87
    severity = "moderate"
    primary_detection = {
        "disease_detected": disease_detected,
        "disease": disease_detected,
        "confidence": confidence,
        "severity": severity,
        "prediction_type": "mock_webhook",
        "source": "supabase_webhook",
    }
    now = _utc_now()
    update_payload = {
        "analysis_status": "completed",
        "primary_detection": primary_detection,
        "detections": [primary_detection],
        "detection_count": 1,
        "highest_severity": severity,
        "processed_at": now,
        "updated_at": now,
        "error_code": None,
        "error_message": None,
    }

    try:
        update_response = await asyncio.to_thread(
            client.table("flight_patches").update(update_payload).eq("id", patch_id).execute
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not update flight_patches row: {exc}") from exc

    updated_rows = getattr(update_response, "data", None) or []
    logger.info(
        "webhook.new_patch_completed",
        extra={
            "patch_id": patch_id,
            "flight_id": flight_id,
            "storage_path": storage_path,
            "downloaded_bytes": len(image_bytes),
        },
    )
    return {
        "ok": True,
        "patch_id": patch_id,
        "flight_id": flight_id,
        "bucket": bucket,
        "storage_path": storage_path,
        "downloaded_bytes": len(image_bytes),
        "result": primary_detection,
        "updated_rows": len(updated_rows),
    }


@app.post("/v1/flights", response_model=FlightCreateResponse)
async def create_flight(payload: FlightCreateRequest):
    flight = create_flight_record(
        device_id=payload.device_id,
        field_id=payload.field_id,
        crop_type=payload.crop_type,
        operator_notes=payload.operator_notes,
        capture_interval_ms=payload.capture_interval_ms,
        model_version_id=payload.model_version_id,
    )
    return FlightCreateResponse(
        flight_id=flight["flight_id"],
        status=flight["status"],
        upload_status=flight["upload_status"],
        processing_status=flight["processing_status"],
        crop_type=flight["crop_type"],
        crop_warning=flight.get("crop_warning"),
        storage_folder=flight.get("storage_folder"),
    )


@app.post("/v1/flights/{flight_id}/images", response_model=FlightImageUploadResponse)
async def upload_flight_patch(
    flight_id: str,
    image: UploadFile = File(...),
    patch_index: int = Form(...),
    captured_at: str = Form(...),
    device_id: Optional[str] = Form(default=None),
    field_id: Optional[str] = Form(default=None),
    crop_type: Optional[str] = Form(default=None),
    lat: Optional[float] = Form(default=None),
    lon: Optional[float] = Form(default=None),
    gps_fix: bool = Form(...),
    altitude_m: Optional[float] = Form(default=None),
    heading_deg: Optional[float] = Form(default=None),
):
    raw_bytes = await image.read()
    response = upload_flight_image(
        flight_id=flight_id,
        raw_bytes=raw_bytes,
        content_type=image.content_type or "image/jpeg",
        patch_index=patch_index,
        captured_at=captured_at,
        lat=lat,
        lon=lon,
        gps_fix=gps_fix,
        altitude_m=altitude_m,
        heading_deg=heading_deg,
        device_id=device_id,
        field_id=field_id,
        crop_type=crop_type,
    )
    return FlightImageUploadResponse(**response)


@app.post("/v1/flights/{flight_id}/esp32-capture", response_model=Esp32FlightCaptureResponse)
async def capture_esp32_flight_patch(flight_id: str, payload: Esp32FlightCaptureRequest):
    raw_bytes, content_type = await asyncio.to_thread(
        _fetch_esp32_snapshot_bytes,
        payload.esp32_ip,
    )
    captured_at = _utc_now()
    response = upload_flight_image(
        flight_id=flight_id,
        raw_bytes=raw_bytes,
        content_type=content_type,
        patch_index=payload.patch_index,
        captured_at=captured_at,
        lat=payload.lat,
        lon=payload.lon,
        gps_fix=payload.gps_fix,
        altitude_m=payload.altitude_m,
        heading_deg=payload.heading_deg,
        crop_type=payload.crop_type,
    )
    patch = production_store.get_flight_image_doc(flight_id, response["image_id"]) or {}
    return Esp32FlightCaptureResponse(
        flight_id=flight_id,
        image_id=response["image_id"],
        patch_index=response["patch_index"],
        storage_path=patch.get("storage_path"),
        storage_folder=patch.get("storage_folder"),
        upload_status=response.get("upload_status", patch.get("upload_status", "uploaded")),
        analysis_status=patch.get("analysis_status", "pending"),
    )


# ── Device Registration ───────────────────────────────────────────────────────

@app.post("/v1/devices/register", response_model=DeviceResponse)
async def register_device(body: DeviceRegisterRequest):
    """Called by the ESP32 on every boot to announce its current LAN IP.

    The server stores the record in memory and persists it to Firebase RTDB so
    the Flutter app can discover the ESP32 IP automatically without any manual
    configuration.
    """
    now = datetime.now(timezone.utc).isoformat()
    existing = _device_registry.get(body.device_id, {})
    record: dict = {
        "device_id": body.device_id,
        "ip": body.ip,
        "firmware_version": body.firmware_version,
        "registered_at": existing.get("registered_at", now),
        "last_seen": now,
    }
    _device_registry[body.device_id] = record
    _persist_device_to_firebase(body.device_id, record)
    # Mirror the IP into drone/ip so the Flutter live-stream screen (which
    # reads Firebase RTDB `drone` node) auto-picks up the new address.
    _sync_drone_ip_to_firebase(body.ip)
    logger.info(
        "device.registered",
        extra={"device_id": body.device_id, "ip": body.ip, "firmware": body.firmware_version},
    )
    return DeviceResponse(**record)


@app.get("/v1/devices", response_model=list[DeviceResponse])
async def list_devices():
    """Return all registered devices with their current IPs."""
    return [DeviceResponse(**r) for r in _device_registry.values()]


@app.get("/v1/devices/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str):
    """Return the current IP for a specific device_id."""
    record = _device_registry.get(device_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not registered. "
                   "Ensure the ESP32 calls POST /v1/devices/register on boot.",
        )
    return DeviceResponse(**record)


@app.get("/v1/esp32/probe")
async def probe_esp32_camera(esp32_ip: Optional[str] = None, device_id: Optional[str] = None):
    ip = _resolve_esp32_ip(esp32_ip, device_id)
    raw_bytes, content_type = await asyncio.to_thread(_fetch_esp32_snapshot_bytes, ip)
    return {
        "ok": True,
        "esp32_ip": _normalize_esp32_ip(ip),
        "capture_bytes": len(raw_bytes),
        "content_type": content_type,
    }


@app.get("/v1/esp32/snapshot")
async def esp32_snapshot_proxy(esp32_ip: Optional[str] = None, device_id: Optional[str] = None):
    ip = _resolve_esp32_ip(esp32_ip, device_id)
    raw_bytes, content_type = await asyncio.to_thread(_fetch_esp32_snapshot_bytes, ip)

    return Response(
        content=raw_bytes,
        media_type=content_type,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _esp32_origin(esp32_ip: str) -> str:
    """Return the first plausible origin for the ESP32 (no port, then :81 fallback).

    This reuses _esp32_capture_base_candidates to honour any explicit scheme/port
    in the input.
    """
    candidates = _esp32_capture_base_candidates(esp32_ip)
    if not candidates:
        raise HTTPException(status_code=400, detail="ESP32 IP address is required")
    return candidates[0]


def _fetch_esp32_status(esp32_ip: str) -> dict[str, Any]:
    last_error = "no status endpoint attempted"
    attempts: list[str] = []
    for base_url in _esp32_capture_base_candidates(esp32_ip):
        url = f"{base_url}/status"
        attempts.append(url)
        try:
            response = requests.get(url, timeout=5)
        except requests.RequestException as exc:
            last_error = f"{url}: {exc}"
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                last_error = f"{url}: invalid JSON ({exc})"
                continue
        last_error = f"{url}: HTTP {response.status_code}"
    raise HTTPException(
        status_code=502,
        detail=(
            "Could not fetch ESP32 camera status. Tried "
            f"{', '.join(attempts)}. Last error: {last_error}"
        ),
    )


def _send_esp32_control(esp32_ip: str, var: str, val: int) -> None:
    last_error = "no control endpoint attempted"
    attempts: list[str] = []
    for base_url in _esp32_capture_base_candidates(esp32_ip):
        url = f"{base_url}/control"
        attempts.append(url)
        try:
            response = requests.get(url, params={"var": var, "val": val}, timeout=5)
        except requests.RequestException as exc:
            last_error = f"{url}: {exc}"
            continue
        if response.status_code == 200:
            return
        last_error = f"{url}: HTTP {response.status_code}"
    raise HTTPException(
        status_code=502,
        detail=(
            "Could not push ESP32 camera control. Tried "
            f"{', '.join(attempts)}. Last error: {last_error}"
        ),
    )


@app.get("/v1/esp32/camera-status")
async def esp32_camera_status(esp32_ip: Optional[str] = None, device_id: Optional[str] = None):
    """Proxy GET <esp32>/status — returns the camera's full settings JSON."""
    ip = _resolve_esp32_ip(esp32_ip, device_id)
    payload = await asyncio.to_thread(_fetch_esp32_status, ip)
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/v1/esp32/camera-control")
async def esp32_camera_control(body: Esp32CameraControlRequest, esp32_ip: Optional[str] = None, device_id: Optional[str] = None):
    """Proxy GET <esp32>/control?var=<var>&val=<val> — applies a single setting."""
    ip = _resolve_esp32_ip(esp32_ip, device_id)
    await asyncio.to_thread(_send_esp32_control, ip, body.var, body.val)
    return {"success": True, "var": body.var, "val": body.val}


@app.post("/v1/flights/{flight_id}/complete", response_model=FlightCompleteResponse)
async def complete_flight(flight_id: str):
    response = complete_flight_upload(flight_id)
    return FlightCompleteResponse(**response)


@app.get("/v1/photos", response_model=list[PhotoRecordResponse])
async def list_photos(
    request: Request,
    flight_id: Optional[str] = Query(default=None, min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    patches = production_store.list_recent_photos(flight_id=flight_id, limit=limit)
    return [PhotoRecordResponse(**_photo_response_from_patch(request, patch)) for patch in patches]


@app.get("/v1/flights/{flight_id}", response_model=FlightStatusResponse)
async def fetch_flight_status(flight_id: str):
    response = get_flight_status(flight_id)
    return FlightStatusResponse(**response)


@app.get("/v1/devices/{device_id}", response_model=DeviceStatusResponse)
async def fetch_device_status(device_id: str):
    response = production_store.get_device(device_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
    return DeviceStatusResponse(**response)


@app.get("/v1/fields/{field_id}/flights", response_model=FieldFlightsResponse)
async def fetch_field_flights(field_id: str, limit: int = Query(default=20, ge=1, le=100)):
    flights = production_store.list_field_flights(field_id, limit=limit)
    return FieldFlightsResponse(field_id=field_id, flights=flights)


@app.post("/missions")
async def create_mission(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    crop = payload.get("crop", "rice")
    normalized_crop, crop_warning = normalize_crop(crop)
    mission_id = str(uuid.uuid4())
    mission_payload = {
        "missionId": mission_id,
        "status": "capturing",
        "crop": normalized_crop,
        "crop_requested": crop,
        "crop_warning": crop_warning,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "capture_interval_ms": payload.get("capture_interval_ms"),
        "notes": payload.get("notes"),
        "analyze_now": False,
    }
    firebase_client.create_mission(mission_id, mission_payload)
    logger.info("mission.created", extra={"mission_id": mission_id, "crop": normalized_crop})
    return {"missionId": mission_id, "status": "capturing", "crop": normalized_crop}


@app.get("/missions/{mission_id}")
async def get_mission(mission_id: str):
    mission = _get_existing_mission_or_404(mission_id)
    return mission


@app.post("/missions/{mission_id}/analyze")
async def analyze_mission(mission_id: str):
    mission = _get_existing_mission_or_404(mission_id)
    current_status = mission.get("status")
    if current_status == "processing":
        return _mission_error_response(409, "analysis_in_progress", "analysis in progress")

    firebase_client.set_mission_status(
        mission_id,
        "processing",
        {"analyze_now": True, "updated_at": _utc_now()},
    )

    try:
        result = await _run_mission_analysis(mission_id)
    except HTTPException as exc:
        firebase_client.set_mission_status(
            mission_id,
            "error",
            {
                "error_code": "analysis_failed",
                "error_message": str(exc.detail),
                "error_reason": str(exc.detail),
                "updated_at": _utc_now(),
            },
        )
        raise
    except Exception as exc:
        firebase_client.set_mission_status(
            mission_id,
            "error",
            {"error_code": "analysis_failed", "error_message": str(exc), "error_reason": str(exc), "updated_at": _utc_now()},
        )
        raise
    if result.get("busy"):
        firebase_client.set_mission_status(
            mission_id,
            current_status or "uploaded",
            {"updated_at": _utc_now()},
        )
        return _mission_error_response(409, "analysis_in_progress", "analysis in progress")
    return result


@app.middleware("http")
async def add_default_json_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except FirebaseConfigError:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request.unhandled_exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"error_code": "internal_error", "message": str(exc)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True, workers=1)
