import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import firebase_client
from firebase_client import FirebaseConfigError
from inference import (
    DEBUG_LATEST_DECODED_IMAGE_URL,
    DEBUG_LATEST_IMAGE_URL,
    VALID_CROPS,
    debug_path,
    decode_image_bytes,
    ensure_default_model,
    get_model,
    loaded_models,
    normalize_crop,
    process_prediction,
    run_inference,
)
from report_generator import generate_report
from storage_helper import StorageDownloadError, download_and_validate_image

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agridrone.api")

app = FastAPI(
    title="AgriDrone Guardian API",
    description="AI-powered crop disease detection for Nepali farmers",
    version="1.1.0",
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


@app.on_event("startup")
def startup() -> None:
    ensure_default_model()
    get_model("rice")
    logger.info(
        "startup.models",
        extra={"models_loaded": sorted(loaded_models.keys()), "max_concurrent_infer": MAX_CONCURRENT_INFER},
    )
    if firebase_client.firebase_is_configured():
        try:
            firebase_client.get_firebase_app()
            logger.info("startup.firebase_init", extra={"firebase_configured": True})
        except Exception as exc:
            logger.warning("startup.firebase_init_failed: %s", exc)
    else:
        logger.info("startup.firebase_init_skipped", extra={"firebase_configured": False})
    logger.info("startup.complete", extra={"max_concurrent_infer": MAX_CONCURRENT_INFER})


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
        "version": "1.1.0",
        "crops_supported": VALID_CROPS,
    }


@app.get("/health")
def health():
    from inference import MODELS_DIR, loaded_models

    models_loaded = list(loaded_models.keys())
    available_models = []
    for crop in VALID_CROPS:
        onnx = MODELS_DIR / f"{crop}_disease_best.onnx"
        pt = MODELS_DIR / f"{crop}_disease_best.pt"
        if onnx.exists() or pt.exists():
            available_models.append(crop)
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "models_available": available_models,
        "firebase_configured": firebase_client.firebase_is_configured(),
        "max_concurrent_infer": MAX_CONCURRENT_INFER,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        detections = image.get("detections", [])
        if detections:
            images_with_detections += 1
        for detection in detections:
            disease = detection["disease"]
            severity = detection["severity"]
            confidence = float(detection["confidence"])
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
    save_to_firebase: bool = Query(default=True, description="Whether to save results to Firebase if configured."),
):
    body = await request.body()
    return process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_firebase,
        request_content_type=request.headers.get("content-type", ""),
    )


@app.post("/predict_form")
@app.post("/predict_upload")
async def predict_form(
    image: UploadFile = File(...),
    crop: str = Query(default="rice", description="Crop type: rice, wheat, maize, potato, tomato, pepper"),
    confidence: float = Query(default=0.3, description="Minimum confidence threshold"),
    save_to_firebase: bool = Query(default=True, description="Whether to save results to Firebase if configured."),
):
    body = await image.read()
    return process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_firebase,
        request_content_type=image.content_type or "",
    )


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
