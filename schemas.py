from typing import Any, Optional

from pydantic import BaseModel, Field


class FlightCreateRequest(BaseModel):
    device_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)
    crop_type: str = Field(default="rice", min_length=1)
    operator_notes: Optional[str] = None
    capture_interval_ms: Optional[int] = Field(default=None, ge=1000)
    model_version_id: Optional[str] = None


class FlightCreateResponse(BaseModel):
    flight_id: str
    status: str
    upload_status: str
    processing_status: str
    crop_type: str
    crop_warning: Optional[str] = None
    storage_folder: Optional[str] = None


class FlightImageUploadResponse(BaseModel):
    flight_id: str
    image_id: str
    patch_index: int
    status: str
    upload_status: str
    processing_status: str
    image_counts: dict[str, int]


class FlightCompleteResponse(BaseModel):
    flight_id: str
    status: str
    upload_status: str
    processing_status: str
    queue_job_id: Optional[str] = None
    image_counts: dict[str, int]


class FlightStatusResponse(BaseModel):
    flight_id: str
    status: str
    upload_status: str
    processing_status: str
    device_id: str
    field_id: str
    crop_type: str
    crop_requested: str
    crop_warning: Optional[str] = None
    operator_notes: Optional[str] = None
    queue_job_id: Optional[str] = None
    capture_interval_ms: Optional[int] = None
    model_version_id: Optional[str] = None
    storage_folder: Optional[str] = None
    created_at: str
    updated_at: str
    image_counts: dict[str, int]
    summary: Optional[dict[str, Any]] = None
    report: Optional[dict[str, Any]] = None
    latest_sensor_snapshot: Optional[dict[str, Any]] = None
    patches: list[dict[str, Any]] = Field(default_factory=list)


class DeviceStatusResponse(BaseModel):
    id: str
    thing_name: Optional[str] = None
    hardware_type: Optional[str] = None
    camera_type: Optional[str] = None
    firmware_version: Optional[str] = None
    field_id: Optional[str] = None
    status: Optional[str] = None
    last_seen_at: Optional[str] = None
    last_reported_ip: Optional[str] = None
    last_rssi: Optional[int] = None


class FieldFlightsResponse(BaseModel):
    field_id: str
    flights: list[dict[str, Any]] = Field(default_factory=list)


class Esp32SnapshotInferenceRequest(BaseModel):
    esp32_ip: str = Field(min_length=1)
    crop: str = "rice"
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class Esp32CameraControlRequest(BaseModel):
    var: str = Field(min_length=1, description="Camera setting name (framesize, quality, brightness, etc.)")
    val: int = Field(description="Integer value for the setting")


class Esp32FlightCaptureRequest(BaseModel):
    esp32_ip: str = Field(min_length=1)
    patch_index: int = Field(ge=0)
    crop_type: str = "rice"
    gps_fix: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None


class Esp32FlightCaptureResponse(BaseModel):
    flight_id: str
    image_id: str
    patch_index: int
    storage_path: Optional[str] = None
    storage_folder: Optional[str] = None
    upload_status: str
    analysis_status: str


class PhotoRecordResponse(BaseModel):
    id: str
    flight_id: str
    image_id: str
    patch_index: int
    captured_at: Optional[str] = None
    storage_path: Optional[str] = None
    storage_folder: Optional[str] = None
    preview_url: Optional[str] = None
    upload_status: str = "unknown"
    analysis_status: str = "unknown"
    primary_detection: Optional[dict[str, Any]] = None
    detection_count: int = 0
    highest_severity: Optional[str] = None
    crop_type: Optional[str] = None
    content_type: Optional[str] = None
    uploaded_at: Optional[str] = None
    processed_at: Optional[str] = None


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(min_length=1, description="Stable device identifier, e.g. 'esp32-drone-01'")
    ip: str = Field(min_length=1, description="Current LAN IP address of the ESP32")
    firmware_version: Optional[str] = None


class DeviceResponse(BaseModel):
    device_id: str
    ip: str
    firmware_version: Optional[str] = None
    registered_at: str
    last_seen: str


class InferenceRecordResponse(BaseModel):
    inference_id: str
    source_kind: str = "upload"
    source_label: Optional[str] = None
    crop: str = "rice"
    crop_requested: str = "rice"
    crop_warning: Optional[str] = None
    status: str = "processing"
    request_content_type: Optional[str] = None
    storage_path: Optional[str] = None
    image_size: dict[str, Any] = Field(default_factory=dict)
    disease: str = "Unknown"
    confidence: float = 0.0
    severity: str = "unknown"
    prediction_type: str = "unknown"
    model: Optional[str] = None
    model_kind: Optional[str] = None
    artifact_label: Optional[str] = None
    primary_detection: Optional[dict[str, Any]] = None
    all_detections: list[dict[str, Any]] = Field(default_factory=list)
    top_predictions: list[dict[str, Any]] = Field(default_factory=list)
    firebase_saved: bool = False
    supabase_saved: bool = False
    latest_detection_saved: bool = False
    record_saved: bool = True
    backend_provider: Optional[str] = None
    storage_mode: Optional[str] = None
    debug_latest_image_url: Optional[str] = None
    debug_latest_decoded_image_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
