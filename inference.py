import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger("agridrone.inference")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "").strip()
DEBUG_DIR = Path(os.getenv("DEBUG_IMAGE_DIR", "./debug"))
DEBUG_LATEST_IMAGE_URL = "/debug/latest.jpg"
DEBUG_LATEST_DECODED_IMAGE_URL = "/debug/latest_decoded.jpg"
VALID_CROPS = ["rice", "wheat", "maize", "potato", "tomato", "pepper"]

loaded_models: dict[str, object] = {}
_onnx_runtime = None


def _download_model_from_gdrive(model_path: Path) -> bool:
    if not MODEL_GDRIVE_ID:
        logger.warning("startup.model_download_skipped", extra={"reason": "missing_model_gdrive_id"})
        return False

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"

    try:
        import gdown

        logger.info("startup.model_download_begin", extra={"model_path": str(model_path)})
        output = gdown.download(url, str(model_path), quiet=False)
        if output and Path(output).exists():
            logger.info("startup.model_download_complete", extra={"model_path": str(model_path)})
            return True
        logger.error("startup.model_download_failed", extra={"model_path": str(model_path)})
        return False
    except Exception:
        logger.exception("startup.model_download_exception", extra={"model_path": str(model_path)})
        return False


def ensure_default_model() -> None:
    rice_model = MODELS_DIR / "rice_disease_best.onnx"
    if not rice_model.exists():
        _download_model_from_gdrive(rice_model)


def _load_onnxruntime():
    global _onnx_runtime
    if _onnx_runtime is not None:
        return _onnx_runtime
    try:
        import onnxruntime as ort

        _onnx_runtime = ort
        logger.info("startup.onnxruntime_available")
        return ort
    except Exception:
        _onnx_runtime = False
        logger.info("startup.onnxruntime_unavailable")
        return None


class OnnxRuntimeDetector:
    def __init__(self, model_path: Path):
        ort = _load_onnxruntime()
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]
        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        self.names = self._parse_names(metadata.get("names"))
        logger.info("model.loaded_onnxruntime", extra={"model_path": str(model_path)})

    @staticmethod
    def _parse_names(raw_names: Optional[str]) -> dict[int, str]:
        if not raw_names:
            return {}
        stripped = raw_names.strip().strip("{}")
        names: dict[int, str] = {}
        for chunk in stripped.split(","):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            try:
                idx = int(key.strip().strip("'\""))
            except ValueError:
                continue
            names[idx] = value.strip().strip("'\"")
        return names

    def __call__(self, img: Image.Image, conf: float = 0.3):
        image = img.convert("RGB").resize((640, 640))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        batch = np.expand_dims(array, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: batch})
        predictions = outputs[0]
        return [OnnxRuntimeResult(predictions, self.names, conf)]


class _ScalarWrapper:
    def __init__(self, value: float):
        self.value = value

    def __getitem__(self, index):
        return self.value

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)


class _ArrayWrapper:
    def __init__(self, values: list[float]):
        self.values = values

    def __getitem__(self, index):
        return self

    def tolist(self):
        return self.values


class OnnxRuntimeBox:
    def __init__(self, class_id: int, conf: float, bbox: list[float]):
        self.cls = _ScalarWrapper(class_id)
        self.conf = _ScalarWrapper(conf)
        self.xyxy = _ArrayWrapper(bbox)


class OnnxRuntimeResult:
    def __init__(self, predictions, names: dict[int, str], threshold: float):
        self.names = names or {}
        self.boxes = self._parse_boxes(predictions, threshold)

    def _parse_boxes(self, predictions, threshold: float):
        boxes = []
        array = np.array(predictions)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            return boxes

        # Best-effort parser for YOLO-style ONNX output.
        if array.shape[0] > array.shape[1]:
            for row in array:
                if row.shape[0] < 6:
                    continue
                objectness = float(row[4])
                class_scores = row[5:]
                if class_scores.size == 0:
                    continue
                class_id = int(np.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
                if confidence < threshold:
                    continue
                x_center, y_center, width, height = [float(v) for v in row[:4]]
                bbox = [
                    x_center - width / 2.0,
                    y_center - height / 2.0,
                    x_center + width / 2.0,
                    y_center + height / 2.0,
                ]
                boxes.append(OnnxRuntimeBox(class_id, confidence, bbox))
        return boxes


def get_model(crop: str):
    if crop in loaded_models:
        return loaded_models[crop]

    onnx_path = MODELS_DIR / f"{crop}_disease_best.onnx"
    pt_path = MODELS_DIR / f"{crop}_disease_best.pt"

    if onnx_path.exists():
        try:
            model = OnnxRuntimeDetector(onnx_path)
        except Exception:
            from ultralytics import YOLO

            model = YOLO(str(onnx_path), task="detect")
            logger.info("model.loaded_ultralytics_onnx", extra={"crop": crop, "path": str(onnx_path)})
        loaded_models[crop] = model
        return model

    if pt_path.exists():
        from ultralytics import YOLO

        model = YOLO(str(pt_path))
        loaded_models[crop] = model
        logger.info("model.loaded_ultralytics_pt", extra={"crop": crop, "path": str(pt_path)})
        return model

    return None


def normalize_crop(crop: str) -> tuple[str, Optional[str]]:
    normalized_crop = (crop or "rice").strip().lower()
    if normalized_crop in VALID_CROPS:
        return normalized_crop, None
    return "rice", f"Unsupported crop '{crop}'. Falling back to 'rice'."


def decode_raw_image(body: bytes) -> Image.Image:
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty")

    np_buffer = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode JPEG image from request body")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def decode_image_bytes(body: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(body))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Failed to decode image bytes") from exc


def debug_path(name: str) -> Path:
    return DEBUG_DIR / name


def save_debug_images(raw_bytes: bytes, img: Image.Image) -> bool:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

        debug_path(f"{stamp}.jpg").write_bytes(raw_bytes)
        debug_path("latest.jpg").write_bytes(raw_bytes)

        decoded_path = debug_path(f"{stamp}_decoded.jpg")
        latest_decoded_path = debug_path("latest_decoded.jpg")
        img.save(decoded_path, format="JPEG")
        img.save(latest_decoded_path, format="JPEG")
        return True
    except Exception:
        logger.exception("debug.save_failed")
        return False


def get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "severe"
    if confidence >= 0.60:
        return "moderate"
    if confidence >= 0.30:
        return "mild"
    return "trace"


def run_inference(img: Image.Image, crop: str, confidence: float) -> tuple[list[dict], Optional[dict]]:
    model = get_model(crop)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{crop}' not available yet. Training in progress.",
        )

    results = model(img, conf=confidence)[0]
    detections: list[dict] = []
    if getattr(results, "boxes", None):
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = getattr(results, "names", {}).get(class_id, str(class_id))
            bbox = box.xyxy[0].tolist()
            detections.append(
                {
                    "disease": class_name,
                    "confidence": round(conf_score, 4),
                    "bbox": [round(float(x), 2) for x in bbox],
                    "severity": get_severity(conf_score),
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    primary = detections[0] if detections else None
    return detections, primary


def build_prediction_response(
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


def process_prediction(
    *,
    raw_bytes: bytes,
    crop: str,
    confidence: float,
    save_to_firebase: bool,
    request_content_type: str,
) -> dict:
    normalized_crop, crop_warning = normalize_crop(crop)
    img = decode_raw_image(raw_bytes)
    detections, primary = run_inference(img, normalized_crop, confidence)
    debug_saved = save_debug_images(raw_bytes, img)
    return build_prediction_response(
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
