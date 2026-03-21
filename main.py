import os
import io
import asyncio
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "").strip()

# Keep crop list, but only load models that actually exist in /models
VALID_CROPS = ["rice", "wheat", "maize", "potato", "tomato", "pepper"]

# Prevent memory spikes from concurrent inference
INFER_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFER", "1")))

app = FastAPI(
    title="AgriDrone Guardian API",
    version="2.1.0",
    description="Low-memory, production-safe crop disease detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_models: Dict[str, object] = {}


def download_model_if_needed(model_path: Path) -> None:
    """Download model from Google Drive if configured (used for rice bootstrap)."""
    if not MODEL_GDRIVE_ID:
        print("⚠️ MODEL_GDRIVE_ID not set; skipping model download.")
        return
    try:
        import gdown
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
        print(f"⬇️ Downloading model to {model_path} ...")
        gdown.download(url, str(model_path), quiet=False)
    except Exception as e:
        print("❌ Model download failed:", e)


def _find_available_models() -> Dict[str, Path]:
    """Find all crop models present on disk (onnx preferred)."""
    available: Dict[str, Path] = {}
    for crop in VALID_CROPS:
        onnx_path = MODELS_DIR / f"{crop}_disease_best.onnx"
        pt_path = MODELS_DIR / f"{crop}_disease_best.pt"
        if onnx_path.exists():
            available[crop] = onnx_path
        elif pt_path.exists():
            available[crop] = pt_path
    return available


@app.on_event("startup")
def startup():
    print("🚀 Starting AgriDrone API...")

    # Ensure rice exists (optional bootstrap)
    rice_onnx = MODELS_DIR / "rice_disease_best.onnx"
    if not rice_onnx.exists():
        download_model_if_needed(rice_onnx)

    available = _find_available_models()
    if not available:
        print("⚠️ No models found in MODELS_DIR:", MODELS_DIR.resolve())

    from ultralytics import YOLO

    for crop, path in available.items():
        try:
            if path.suffix.lower() == ".onnx":
                print(f"Loading {crop} ONNX model: {path}")
                loaded_models[crop] = YOLO(str(path), task="detect")
            else:
                print(f"Loading {crop} PT model: {path}")
                loaded_models[crop] = YOLO(str(path))
        except Exception as e:
            print(f"❌ Failed loading model {crop}:", e)

    print("✅ Models loaded:", list(loaded_models.keys()))


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "agridrone-api",
        "models_loaded": list(loaded_models.keys()),
        "models_available": list(_find_available_models().keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "models_available": list(_find_available_models().keys()),
    }


def decode_image(raw_bytes: bytes):
    """Decode JPEG bytes to RGB numpy array using Pillow (lighter than cv2)."""
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image body")

    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JPEG image")

    img_np = np.array(img)
    w, h = img.size
    return img_np, (w, h)


def get_severity(conf: float) -> str:
    if conf >= 0.85:
        return "severe"
    if conf >= 0.60:
        return "moderate"
    if conf >= 0.30:
        return "mild"
    return "trace"


async def run_inference(img_np: np.ndarray, crop: str, confidence: float):
    if crop not in loaded_models:
        raise HTTPException(status_code=503, detail=f"Model '{crop}' not loaded")

    model = loaded_models[crop]

    async with INFER_SEMAPHORE:
        try:
            results = await run_in_threadpool(model, img_np, conf=confidence)
            results = results[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = results.names[class_id]
            bbox = box.xyxy[0].tolist()
            detections.append({
                "disease": class_name,
                "confidence": round(conf_score, 4),
                "bbox": [round(x, 2) for x in bbox],
                "severity": get_severity(conf_score),
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    primary = detections[0] if detections else None
    return detections, primary


@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Query(default="rice"),
    confidence: float = Query(default=0.3),
):
    raw_bytes = await request.body()
    img_np, (w, h) = decode_image(raw_bytes)
    crop = (crop or "rice").strip().lower()

    detections, primary = await run_inference(img_np, crop, confidence)

    return {
        "status": "success",
        "crop": crop,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": w, "height": h},
        "model": f"{crop}_disease",
    }


@app.post("/predict_upload")
async def predict_upload(
    image: UploadFile = File(...),
    crop: str = Query(default="rice"),
    confidence: float = Query(default=0.3),
):
    raw_bytes = await image.read()
    img_np, (w, h) = decode_image(raw_bytes)
    crop = (crop or "rice").strip().lower()

    detections, primary = await run_inference(img_np, crop, confidence)

    return {
        "status": "success",
        "crop": crop,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": w, "height": h},
        "model": f"{crop}_disease",
    }
