import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from dotenv import load_dotenv

# ===============================
# ENVIRONMENT SETUP
# ===============================

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "").strip()
DEBUG_DIR = Path(os.getenv("DEBUG_IMAGE_DIR", "./debug"))
VALID_CROPS = ["rice", "wheat", "maize", "potato", "tomato", "pepper"]

app = FastAPI(
    title="AgriDrone Guardian API",
    version="2.0.0",
    description="Production-ready AI crop disease detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODEL LOADING (SAFE STARTUP)
# ===============================

loaded_models = {}

def download_model_if_needed(model_path: Path):
    if not MODEL_GDRIVE_ID:
        print("⚠️ MODEL_GDRIVE_ID not set. Skipping download.")
        return

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(model_path), quiet=False)
    except Exception as e:
        print("❌ Model download failed:", e)

@app.on_event("startup")
def startup():
    print("🚀 Starting AgriDrone API")

    from ultralytics import YOLO

    for crop in VALID_CROPS:
        onnx_path = MODELS_DIR / f"{crop}_disease_best.onnx"
        pt_path = MODELS_DIR / f"{crop}_disease_best.pt"

        try:
            if not onnx_path.exists() and not pt_path.exists():
                if crop == "rice":
                    download_model_if_needed(onnx_path)

            if onnx_path.exists():
                print(f"Loading {crop} ONNX model...")
                loaded_models[crop] = YOLO(str(onnx_path), task="detect")
            elif pt_path.exists():
                print(f"Loading {crop} PT model...")
                loaded_models[crop] = YOLO(str(pt_path))

        except Exception as e:
            print(f"❌ Failed loading model {crop}:", e)

    print("✅ Models loaded:", list(loaded_models.keys()))

# ===============================
# HEALTH ENDPOINT
# ===============================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "models_available": [
            crop for crop in VALID_CROPS
            if (MODELS_DIR / f"{crop}_disease_best.onnx").exists()
            or (MODELS_DIR / f"{crop}_disease_best.pt").exists()
        ]
    }

# ===============================
# IMAGE DECODING (SAFE)
# ===============================

def decode_image(raw_bytes: bytes) -> Image.Image:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image body")

    np_buffer = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG image")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ===============================
# SEVERITY LOGIC
# ===============================

def get_severity(conf: float) -> str:
    if conf >= 0.85:
        return "severe"
    elif conf >= 0.60:
        return "moderate"
    elif conf >= 0.30:
        return "mild"
    return "trace"

# ===============================
# SAFE INFERENCE (THREADPOOL)
# ===============================

async def run_inference(img: Image.Image, crop: str, confidence: float):

    if crop not in loaded_models:
        raise HTTPException(status_code=503, detail=f"Model '{crop}' not loaded")

    model = loaded_models[crop]

    try:
        results = await run_in_threadpool(model, img, conf=confidence)
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
                "severity": get_severity(conf_score)
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    primary = detections[0] if detections else None

    return detections, primary

# ===============================
# RAW JPEG ENDPOINT (ESP32)
# ===============================

@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Query(default="rice"),
    confidence: float = Query(default=0.3),
):
    raw_bytes = await request.body()
    img = decode_image(raw_bytes)

    detections, primary = await run_inference(img, crop.lower(), confidence)

    return {
        "status": "success",
        "crop": crop,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": img.width, "height": img.height},
        "model": f"{crop}_disease"
    }

# ===============================
# SWAGGER FILE UPLOAD ENDPOINT
# ===============================

@app.post("/predict_upload")
async def predict_upload(
    image: UploadFile = File(...),
    crop: str = Query(default="rice"),
    confidence: float = Query(default=0.3),
):
    raw_bytes = await image.read()
    img = decode_image(raw_bytes)

    detections, primary = await run_inference(img, crop.lower(), confidence)

    return {
        "status": "success",
        "crop": crop,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": img.width, "height": img.height},
        "model": f"{crop}_disease"
    }

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
