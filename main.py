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
VALID_CROPS = ["rice", "wheat", "maize", "potato", "tomato", "pepper"]

# ===============================
# MODEL LOADING AT STARTUP
# ===============================

loaded_models = {}

def _download_model_from_gdrive(model_path: Path):
    if not MODEL_GDRIVE_ID:
        return
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(model_path), quiet=False)
    except Exception as e:
        print(f"Model download failed: {e}")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting AgriDrone API...")
    from ultralytics import YOLO

    for crop in VALID_CROPS:
        onnx_path = MODELS_DIR / f"{crop}_disease_best.onnx"
        pt_path   = MODELS_DIR / f"{crop}_disease_best.pt"

        if not onnx_path.exists() and not pt_path.exists():
            if crop == "rice":
                _download_model_from_gdrive(onnx_path)

        if onnx_path.exists():
            print(f"Loading {crop} ONNX model...")
            loaded_models[crop] = YOLO(str(onnx_path), task="detect")
        elif pt_path.exists():
            print(f"Loading {crop} PT model...")
            loaded_models[crop] = YOLO(str(pt_path))

    print("✅ Models loaded:", list(loaded_models.keys()))

# ===============================
# HEALTH CHECK
# ===============================

@app.get("/health")
def health():
    available_models = []
    for crop in VALID_CROPS:
        if (MODELS_DIR / f"{crop}_disease_best.onnx").exists() or \
           (MODELS_DIR / f"{crop}_disease_best.pt").exists():
            available_models.append(crop)

    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "models_available": available_models
    }

# ===============================
# IMAGE PROCESSING
# ===============================

def _decode_raw_image(body: bytes) -> Image.Image:
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    np_buffer = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG image")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def _get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "severe"
    elif confidence >= 0.60:
        return "moderate"
    elif confidence >= 0.30:
        return "mild"
    return "trace"

def _run_inference(img: Image.Image, crop: str, confidence: float):
    if crop not in loaded_models:
        raise HTTPException(status_code=503, detail=f"Model '{crop}' not loaded")

    model = loaded_models[crop]
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

# ===============================
# PREDICT ENDPOINT
# ===============================

@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Query(default="rice"),
    confidence: float = Query(default=0.3),
    save_to_firebase: bool = Query(default=True),
):
    body = await request.body()

    img = _decode_raw_image(body)
    detections, primary = _run_inference(img, crop.lower(), confidence)

    return {
        "status": "success",
        "crop": crop,
        "disease": primary["disease"] if primary else "Healthy",
        "confidence": primary["confidence"] if primary else 1.0,
        "severity": primary["severity"] if primary else "none",
        "all_detections": detections,
        "image_size": {"width": img.width, "height": img.height},
        "model": f"{crop}_disease",
        "firebase_saved": False
    }

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
