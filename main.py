import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AgriDrone Guardian API",
    description="AI-powered crop disease detection for Nepali farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "").strip()
DEBUG_DIR = Path(os.getenv("DEBUG_IMAGE_DIR", "./debug"))
DEBUG_LATEST_IMAGE_URL = "/debug/latest.jpg"
DEBUG_LATEST_DECODED_IMAGE_URL = "/debug/latest_decoded.jpg"
VALID_CROPS = ["rice", "wheat", "maize", "potato", "tomato", "pepper"]

def _download_model_from_gdrive(model_path: Path) -> bool:
    """Downloads the model file from Google Drive if MODEL_GDRIVE_ID is set."""
    if not MODEL_GDRIVE_ID:
        print("⚠️ MODEL_GDRIVE_ID not set; skipping model download.")
        return False
        
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Standard Google Drive download URL format for gdown
    url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
    
    try:
        import gdown
        print(f"⬇️ Downloading model from Google Drive (ID: {MODEL_GDRIVE_ID}) to {model_path} ...")
        # Ensure we download to the exact path
        output = gdown.download(url, str(model_path), quiet=False)
        
        if output and Path(output).exists():
            print(f"✅ Model downloaded successfully to {model_path}")
            return True
        else:
            print(f"❌ Download failed: Output file not found.")
            return False
    except Exception as exc:
        print(f"❌ Failed to download model: {exc}")
        return False

@app.on_event("startup")
def _startup_download_model():
    rice_model = MODELS_DIR / "rice_disease_best.onnx"
    if not rice_model.exists():
        _download_model_from_gdrive(rice_model)

# Load models into memory at startup — one per crop
loaded_models = {}

def get_model(crop: str):
    if crop in loaded_models:
        return loaded_models[crop]
    
    # Try ONNX first (faster on CPU), fall back to .pt
    onnx_path = MODELS_DIR / f"{crop}_disease_best.onnx"
    pt_path   = MODELS_DIR / f"{crop}_disease_best.pt"
    
    if onnx_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(onnx_path), task='detect')
        loaded_models[crop] = model
        print(f"✅ Loaded {crop} model from {onnx_path}")
        return model
    elif pt_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        loaded_models[crop] = model
        print(f"✅ Loaded {crop} model from {pt_path}")
        return model
    else:
        return None


@app.get("/")
def root():
    return {
        "status": "AgriDrone Guardian API is running",
        "version": "1.0.0",
        "crops_supported": ["rice", "wheat", "maize", "potato", "tomato", "pepper"]
    }


@app.get("/health")
def health():
    models_loaded = list(loaded_models.keys())
    available_models = []
    for crop in VALID_CROPS:
        onnx = MODELS_DIR / f"{crop}_disease_best.onnx"
        pt   = MODELS_DIR / f"{crop}_disease_best.pt"
        if onnx.exists() or pt.exists():
            available_models.append(crop)
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "models_available": available_models
    }


def _normalize_crop(crop: str) -> tuple[str, Optional[str]]:
    normalized_crop = (crop or "rice").strip().lower()
    if normalized_crop in VALID_CROPS:
        return normalized_crop, None
    return "rice", f"Unsupported crop '{crop}'. Falling back to 'rice'."


def _decode_raw_image(body: bytes) -> Image.Image:
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty")

    np_buffer = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode JPEG image from request body")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def _debug_path(name: str) -> Path:
    return DEBUG_DIR / name


def _save_debug_images(raw_bytes: bytes, img: Image.Image) -> bool:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

        _debug_path(f"{stamp}.jpg").write_bytes(raw_bytes)
        _debug_path("latest.jpg").write_bytes(raw_bytes)

        decoded_path = _debug_path(f"{stamp}_decoded.jpg")
        latest_decoded_path = _debug_path("latest_decoded.jpg")
        img.save(decoded_path, format="JPEG")
        img.save(latest_decoded_path, format="JPEG")
        return True
    except Exception as exc:
        print(f"⚠️ Failed to save debug images: {exc}")
        return False


def _run_inference(img: Image.Image, crop: str, confidence: float) -> tuple[list[dict], Optional[dict]]:
    model = get_model(crop)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{crop}' not available yet. Training in progress."
        )

    results = model(img, conf=confidence)[0]

    detections = []
    if results.boxes and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = results.names[class_id]
            bbox = box.xyxy[0].tolist()

            detections.append({
                "disease": class_name,
                "confidence": round(conf_score, 4),
                "bbox": [round(x, 2) for x in bbox],
                "severity": _get_severity(conf_score)
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    primary = detections[0] if detections else None
    return detections, primary


def _build_prediction_response(
    *,
    crop: str,
    crop_requested: str,
    crop_warning: Optional[str],
    img: Image.Image,
    detections: list[dict],
    primary: Optional[dict],
    request_content_type: str,
    save_to_firebase: bool,
    debug_saved: bool,
) -> dict:
    return {
        "status": "success",
        "crop": crop,
        "crop_requested": crop_requested,
        "crop_warning": crop_warning,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": img.width, "height": img.height},
        "model": f"{crop}_disease",
        "save_to_firebase_requested": save_to_firebase,
        "firebase_saved": False,
        "debug_saved": debug_saved,
        "debug_latest_image_url": DEBUG_LATEST_IMAGE_URL,
        "debug_latest_decoded_image_url": DEBUG_LATEST_DECODED_IMAGE_URL,
        "request_content_type": request_content_type,
    }


def _process_prediction(
    *,
    raw_bytes: bytes,
    crop: str,
    confidence: float,
    save_to_firebase: bool,
    request_content_type: str,
) -> dict:
    normalized_crop, crop_warning = _normalize_crop(crop)
    img = _decode_raw_image(raw_bytes)
    detections, primary = _run_inference(img, normalized_crop, confidence)
    debug_saved = _save_debug_images(raw_bytes, img)
    return _build_prediction_response(
        crop=normalized_crop,
        crop_requested=crop,
        crop_warning=crop_warning,
        img=img,
        detections=detections,
        primary=primary,
        request_content_type=request_content_type,
        save_to_firebase=save_to_firebase,
        debug_saved=debug_saved,
    )


@app.get(DEBUG_LATEST_IMAGE_URL)
def debug_latest_image():
    latest_path = _debug_path("latest.jpg")
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail="No debug image has been saved yet.")
    return FileResponse(latest_path, media_type="image/jpeg")


@app.get(DEBUG_LATEST_DECODED_IMAGE_URL)
def debug_latest_decoded_image():
    latest_decoded_path = _debug_path("latest_decoded.jpg")
    if not latest_decoded_path.exists():
        raise HTTPException(status_code=404, detail="No decoded debug image has been saved yet.")
    return FileResponse(latest_decoded_path, media_type="image/jpeg")


@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Query(default="rice", description="Crop type: rice, wheat, maize, potato, tomato, pepper"),
    confidence: float = Query(default=0.3, description="Minimum confidence threshold"),
    save_to_firebase: bool = Query(default=True, description="Whether to save results to Firebase if configured.")
):
    body = await request.body()
    return _process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_firebase,
        request_content_type=request.headers.get("content-type", ""),
    )


@app.post("/predict_form")
async def predict_form(
    image: UploadFile = File(...),
    crop: str = Query(default="rice", description="Crop type: rice, wheat, maize, potato, tomato, pepper"),
    confidence: float = Query(default=0.3, description="Minimum confidence threshold"),
    save_to_firebase: bool = Query(default=True, description="Whether to save results to Firebase if configured."),
):
    body = await image.read()
    return _process_prediction(
        raw_bytes=body,
        crop=crop,
        confidence=confidence,
        save_to_firebase=save_to_firebase,
        request_content_type=image.content_type or "",
    )


def _get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "severe"
    elif confidence >= 0.60:
        return "moderate"
    elif confidence >= 0.30:
        return "mild"
    else:
        return "trace"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
