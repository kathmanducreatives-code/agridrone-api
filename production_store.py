import os
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

import firebase_client
import supabase_client
from sqlalchemy import JSON, Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column, sessionmaker

try:
    import boto3
except ImportError:  # pragma: no cover - optional in some local environments
    boto3 = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_provider() -> str:
    explicit = os.getenv("PRODUCTION_BACKEND_PROVIDER", "").strip().lower()
    if explicit in {"aws", "firebase", "supabase"}:
        return explicit
    return "aws"


def provider_name() -> str:
    return _normalized_provider()


def using_aws_backend() -> bool:
    return provider_name() == "aws"


def using_firebase_backend() -> bool:
    return provider_name() == "firebase"


def using_supabase_backend() -> bool:
    return provider_name() == "supabase"


def _database_url() -> str:
    explicit = os.getenv("DATABASE_URL", "").strip() or os.getenv("PRODUCTION_DATABASE_URL", "").strip()
    if explicit:
        return explicit

    default_path = Path(__file__).resolve().parent / "data" / "agridrone_production.db"
    try:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        if os.access(default_path.parent, os.W_OK):
            return f"sqlite:///{default_path}"
    except OSError:
        pass

    fallback_path = Path("/tmp/agridrone_production.db")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback_path}"


def _local_storage_root() -> Path:
    return Path(
        os.getenv("LOCAL_OBJECT_STORAGE_DIR", "").strip()
        or (Path(__file__).resolve().parent / "data" / "object_storage")
    )


def _s3_bucket_name() -> str:
    return os.getenv("AWS_S3_BUCKET", "").strip()


def storage_mode() -> str:
    if using_firebase_backend():
        return "firebase"
    if using_supabase_backend():
        return "supabase-storage" if supabase_client.supabase_is_configured() else "supabase-unconfigured"
    return "s3" if _s3_bucket_name() else "local-files"


class Base(DeclarativeBase):
    pass


class DeviceRecord(Base):
    __tablename__ = "devices"

    id = mapped_column(String(128), primary_key=True)
    thing_name = mapped_column(String(255))
    hardware_type = mapped_column(String(128))
    camera_type = mapped_column(String(128))
    firmware_version = mapped_column(String(128))
    field_id = mapped_column(String(128))
    status = mapped_column(String(64))
    last_seen_at = mapped_column(String(64))
    last_reported_ip = mapped_column(String(128))
    last_rssi = mapped_column(Integer)


class FieldRecord(Base):
    __tablename__ = "fields"

    id = mapped_column(String(128), primary_key=True)
    name = mapped_column(String(255))
    farm_name = mapped_column(String(255))
    crop_type_default = mapped_column(String(128))
    boundary_geojson = mapped_column(JSON)
    notes = mapped_column(Text)


class FlightRecord(Base):
    __tablename__ = "flights"

    id = mapped_column(String(64), primary_key=True)
    device_id = mapped_column(String(128), index=True)
    field_id = mapped_column(String(128), index=True)
    crop_type = mapped_column(String(128))
    crop_requested = mapped_column(String(128))
    crop_warning = mapped_column(Text)
    operator_notes = mapped_column(Text)
    status = mapped_column(String(64))
    upload_status = mapped_column(String(64))
    processing_status = mapped_column(String(64))
    queue_job_id = mapped_column(String(128))
    capture_interval_ms = mapped_column(Integer)
    model_version_id = mapped_column(String(128))
    storage_folder = mapped_column(String(255))
    summary = mapped_column(JSON)
    report = mapped_column(JSON)
    latest_sensor_snapshot = mapped_column(JSON)
    image_counts = mapped_column(JSON)
    created_at = mapped_column(String(64))
    updated_at = mapped_column(String(64))
    uploaded_at = mapped_column(String(64))
    completed_at = mapped_column(String(64))
    queued_at = mapped_column(String(64))
    processing_started_at = mapped_column(String(64))
    error_code = mapped_column(String(128))
    error_message = mapped_column(Text)


class FlightPatchRecord(Base):
    __tablename__ = "flight_patches"

    id = mapped_column(String(128), primary_key=True)
    flight_id = mapped_column(String(64), index=True)
    image_id = mapped_column(String(128), index=True)
    patch_index = mapped_column(Integer)
    captured_at = mapped_column(String(64))
    storage_path = mapped_column(Text)
    storage_folder = mapped_column(String(255))
    upload_status = mapped_column(String(64))
    analysis_status = mapped_column(String(64))
    gps = mapped_column(JSON)
    detections = mapped_column(JSON)
    primary_detection = mapped_column(JSON)
    content_type = mapped_column(String(128))
    image_size = mapped_column(JSON)
    detection_count = mapped_column(Integer)
    highest_severity = mapped_column(String(64))
    crop_type = mapped_column(String(128))
    crop_warning = mapped_column(Text)
    uploaded = mapped_column(Boolean)
    uploaded_at = mapped_column(String(64))
    processed_at = mapped_column(String(64))
    updated_at = mapped_column(String(64))
    error_code = mapped_column(String(128))
    error_message = mapped_column(Text)


class FlightDetectionRecord(Base):
    __tablename__ = "detections"

    id = mapped_column(String(128), primary_key=True)
    flight_patch_id = mapped_column(String(128), index=True)
    disease = mapped_column(String(255))
    confidence = mapped_column(Float)
    severity = mapped_column(String(64))
    bbox_x1 = mapped_column(Float, nullable=True)
    bbox_y1 = mapped_column(Float, nullable=True)
    bbox_x2 = mapped_column(Float, nullable=True)
    bbox_y2 = mapped_column(Float, nullable=True)
    class_index = mapped_column(Integer)


class FlightReportRecord(Base):
    __tablename__ = "flight_reports"

    id = mapped_column(String(128), primary_key=True)
    flight_id = mapped_column(String(64), index=True)
    top_disease = mapped_column(String(255))
    highest_severity = mapped_column(String(64))
    summary_json = mapped_column(JSON)
    report_json = mapped_column(JSON)
    generated_at = mapped_column(String(64))


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id = mapped_column(String(128), primary_key=True)
    crop_type = mapped_column(String(128))
    model_name = mapped_column(String(255))
    model_format = mapped_column(String(64))
    storage_key = mapped_column(Text)
    version_label = mapped_column(String(128))
    trained_at = mapped_column(String(64))
    deployed_at = mapped_column(String(64))
    is_active = mapped_column(Boolean, default=False)


class InferenceRecord(Base):
    __tablename__ = "inferences"

    id = mapped_column(String(128), primary_key=True)
    source_kind = mapped_column(String(64))
    source_label = mapped_column(String(255))
    crop_type = mapped_column(String(128))
    crop_requested = mapped_column(String(128))
    crop_warning = mapped_column(Text)
    status = mapped_column(String(64))
    request_content_type = mapped_column(String(128))
    storage_path = mapped_column(Text)
    image_size = mapped_column(JSON)
    disease = mapped_column(String(255))
    confidence = mapped_column(Float)
    severity = mapped_column(String(64))
    prediction_type = mapped_column(String(64))
    model_name = mapped_column(String(255))
    model_kind = mapped_column(String(64))
    artifact_label = mapped_column(String(255))
    primary_detection = mapped_column(JSON)
    all_detections = mapped_column(JSON)
    top_predictions = mapped_column(JSON)
    firebase_saved = mapped_column(Boolean, default=False)
    backend_provider = mapped_column(String(64))
    storage_mode_name = mapped_column(String(64))
    created_at = mapped_column(String(64))
    updated_at = mapped_column(String(64))
    error_message = mapped_column(Text)


@lru_cache(maxsize=1)
def _engine():
    url = _database_url()
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


@lru_cache(maxsize=1)
def _session_factory():
    return sessionmaker(bind=_engine(), expire_on_commit=False, future=True)


def initialize() -> None:
    if using_aws_backend():
        Base.metadata.create_all(_engine())
        if storage_mode() == "local-files":
            _local_storage_root().mkdir(parents=True, exist_ok=True)


@contextmanager
def _session_scope() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _flight_to_dict(flight: FlightRecord) -> dict[str, Any]:
    return {
        "flight_id": flight.id,
        "device_id": flight.device_id,
        "field_id": flight.field_id,
        "crop_type": flight.crop_type,
        "crop_requested": flight.crop_requested,
        "crop_warning": flight.crop_warning,
        "operator_notes": flight.operator_notes,
        "status": flight.status,
        "upload_status": flight.upload_status,
        "processing_status": flight.processing_status,
        "queue_job_id": flight.queue_job_id,
        "capture_interval_ms": flight.capture_interval_ms,
        "model_version_id": flight.model_version_id,
        "storage_folder": flight.storage_folder,
        "summary": flight.summary,
        "report": flight.report,
        "latest_sensor_snapshot": flight.latest_sensor_snapshot,
        "image_counts": flight.image_counts,
        "created_at": flight.created_at,
        "updated_at": flight.updated_at,
        "uploaded_at": flight.uploaded_at,
        "completed_at": flight.completed_at,
        "queued_at": flight.queued_at,
        "processing_started_at": flight.processing_started_at,
        "error_code": flight.error_code,
        "error_message": flight.error_message,
    }


def _patch_to_dict(patch: FlightPatchRecord) -> dict[str, Any]:
    return {
        "id": patch.id,
        "flight_id": patch.flight_id,
        "image_id": patch.image_id,
        "patch_index": patch.patch_index,
        "captured_at": patch.captured_at,
        "storage_path": patch.storage_path,
        "storage_folder": patch.storage_folder,
        "upload_status": patch.upload_status,
        "analysis_status": patch.analysis_status,
        "gps": patch.gps,
        "detections": patch.detections or [],
        "primary_detection": patch.primary_detection,
        "content_type": patch.content_type,
        "image_size": patch.image_size,
        "detection_count": patch.detection_count,
        "highest_severity": patch.highest_severity,
        "crop_type": patch.crop_type,
        "crop_warning": patch.crop_warning,
        "uploaded": patch.uploaded,
        "uploaded_at": patch.uploaded_at,
        "processed_at": patch.processed_at,
        "updated_at": patch.updated_at,
        "error_code": patch.error_code,
        "error_message": patch.error_message,
    }


def _inference_to_dict(record: InferenceRecord) -> dict[str, Any]:
    return {
        "inference_id": record.id,
        "source_kind": record.source_kind,
        "source_label": record.source_label,
        "crop": record.crop_type,
        "crop_requested": record.crop_requested,
        "crop_warning": record.crop_warning,
        "status": record.status,
        "request_content_type": record.request_content_type,
        "storage_path": record.storage_path,
        "image_size": record.image_size or {},
        "disease": record.disease or "Unknown",
        "confidence": round(float(record.confidence or 0.0), 4),
        "severity": record.severity or "unknown",
        "prediction_type": record.prediction_type or "unknown",
        "model": record.model_name,
        "model_kind": record.model_kind,
        "artifact_label": record.artifact_label,
        "primary_detection": record.primary_detection,
        "all_detections": record.all_detections or [],
        "top_predictions": record.top_predictions or [],
        "firebase_saved": bool(record.firebase_saved),
        "record_saved": True,
        "backend_provider": record.backend_provider,
        "storage_mode": record.storage_mode_name,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "error_message": record.error_message,
    }


def _ensure_device_and_field(session: Session, *, device_id: str, field_id: str, crop_type: str) -> None:
    device = session.get(DeviceRecord, device_id) or DeviceRecord(id=device_id)
    device.field_id = field_id
    device.status = device.status or "awaiting_upload"
    device.last_seen_at = _utc_now()
    session.add(device)

    field = session.get(FieldRecord, field_id) or FieldRecord(id=field_id)
    field.crop_type_default = field.crop_type_default or crop_type
    session.add(field)


def _s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3-backed production storage")
    region = os.getenv("AWS_REGION", "").strip() or os.getenv("AWS_DEFAULT_REGION", "").strip()
    kwargs = {"region_name": region} if region else {}
    return boto3.client("s3", **kwargs)


def _normalize_storage_path(storage_path: str) -> tuple[str, str]:
    bucket = _s3_bucket_name()
    if storage_path.startswith("s3://"):
        trimmed = storage_path[5:]
        bucket_name, _, key = trimmed.partition("/")
        return bucket_name, key
    if bucket:
        return bucket, storage_path.lstrip("/")
    return "", storage_path.lstrip("/")


def _upload_local_file(raw_bytes: bytes, key: str) -> str:
    target = _local_storage_root() / key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw_bytes)
    return key


def _download_local_file(key: str) -> bytes:
    target = _local_storage_root() / key
    if not target.exists():
        raise FileNotFoundError(f"Storage object '{key}' not found")
    return target.read_bytes()


_SUPABASE_COLUMNS: dict[str, set[str]] = {
    "devices": {
        "id",
        "thing_name",
        "hardware_type",
        "camera_type",
        "firmware_version",
        "field_id",
        "status",
        "last_seen_at",
        "last_reported_ip",
        "last_rssi",
    },
    "flights": {
        "flight_id",
        "device_id",
        "field_id",
        "crop_type",
        "crop_requested",
        "crop_warning",
        "operator_notes",
        "status",
        "upload_status",
        "processing_status",
        "queue_job_id",
        "capture_interval_ms",
        "model_version_id",
        "storage_folder",
        "summary",
        "report",
        "latest_sensor_snapshot",
        "image_counts",
        "created_at",
        "updated_at",
        "uploaded_at",
        "completed_at",
        "queued_at",
        "processing_started_at",
        "error_code",
        "error_message",
    },
    "flight_patches": {
        "id",
        "flight_id",
        "image_id",
        "patch_index",
        "captured_at",
        "storage_path",
        "storage_folder",
        "upload_status",
        "analysis_status",
        "gps",
        "detections",
        "primary_detection",
        "content_type",
        "image_size",
        "detection_count",
        "highest_severity",
        "crop_type",
        "crop_warning",
        "uploaded",
        "uploaded_at",
        "processed_at",
        "updated_at",
        "error_code",
        "error_message",
    },
    "flight_reports": {
        "id",
        "flight_id",
        "top_disease",
        "highest_severity",
        "summary_json",
        "report_json",
        "generated_at",
    },
    "inferences": {
        "inference_id",
        "source_kind",
        "source_label",
        "crop",
        "crop_requested",
        "crop_warning",
        "status",
        "request_content_type",
        "storage_path",
        "image_size",
        "disease",
        "confidence",
        "severity",
        "prediction_type",
        "model",
        "model_kind",
        "artifact_label",
        "primary_detection",
        "all_detections",
        "top_predictions",
        "firebase_saved",
        "supabase_saved",
        "latest_detection_saved",
        "record_saved",
        "backend_provider",
        "storage_mode",
        "created_at",
        "updated_at",
        "error_message",
    },
    "latest_detections": {
        "id",
        "inference_id",
        "source_kind",
        "source_label",
        "crop",
        "disease",
        "confidence",
        "severity",
        "prediction_type",
        "model",
        "model_kind",
        "artifact_label",
        "storage_path",
        "image_size",
        "timestamp",
        "primary_detection",
        "all_detections",
        "top_predictions",
        "created_at",
    },
    "detection_history": {
        "inference_id",
        "source_kind",
        "source_label",
        "crop",
        "disease",
        "confidence",
        "severity",
        "prediction_type",
        "model",
        "model_kind",
        "artifact_label",
        "storage_path",
        "image_size",
        "timestamp",
        "primary_detection",
        "all_detections",
        "top_predictions",
        "created_at",
    },
}


def _supabase_payload(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    columns = _SUPABASE_COLUMNS.get(table)
    if not columns:
        return payload
    return {key: value for key, value in payload.items() if key in columns}


def firebase_is_configured() -> bool:
    return firebase_client.firebase_is_configured()


def firestore_is_configured() -> bool:
    return firebase_client.firestore_is_configured()


def database_is_configured() -> bool:
    return firebase_client.database_is_configured()


def create_flight_doc(flight_id: str, payload: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.create_flight_doc(flight_id, payload)
        return
    if using_supabase_backend():
        supabase_client.upsert_row("flights", _supabase_payload("flights", payload))
        return

    initialize()
    with _session_scope() as session:
        _ensure_device_and_field(
            session,
            device_id=payload["device_id"],
            field_id=payload["field_id"],
            crop_type=payload["crop_type"],
        )
        flight = FlightRecord(
            id=flight_id,
            device_id=payload["device_id"],
            field_id=payload["field_id"],
            crop_type=payload["crop_type"],
            crop_requested=payload.get("crop_requested", payload["crop_type"]),
            crop_warning=payload.get("crop_warning"),
            operator_notes=payload.get("operator_notes"),
            status=payload.get("status", "awaiting_upload"),
            upload_status=payload.get("upload_status", "pending"),
            processing_status=payload.get("processing_status", "not_started"),
            queue_job_id=payload.get("queue_job_id"),
            capture_interval_ms=payload.get("capture_interval_ms"),
            model_version_id=payload.get("model_version_id"),
            storage_folder=payload.get("storage_folder"),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
        )
        session.add(flight)


def get_flight_doc(flight_id: str) -> Optional[dict[str, Any]]:
    if using_firebase_backend():
        return firebase_client.get_flight_doc(flight_id)
    if using_supabase_backend():
        return supabase_client.fetch_row("flights", "flight_id", flight_id)

    initialize()
    with _session_scope() as session:
        flight = session.get(FlightRecord, flight_id)
        return None if flight is None else _flight_to_dict(flight)


def update_flight_doc(flight_id: str, payload: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.update_flight_doc(flight_id, payload)
        return
    if using_supabase_backend():
        supabase_client.patch_row("flights", "flight_id", flight_id, _supabase_payload("flights", payload))
        return

    initialize()
    with _session_scope() as session:
        flight = session.get(FlightRecord, flight_id)
        if flight is None:
            raise KeyError(f"Flight '{flight_id}' not found")
        for key, value in payload.items():
            if hasattr(flight, key):
                setattr(flight, key, value)
        if payload.get("device_id") and payload.get("field_id"):
            _ensure_device_and_field(
                session,
                device_id=payload["device_id"],
                field_id=payload["field_id"],
                crop_type=payload.get("crop_type", flight.crop_type),
            )


def get_flight_image_doc(flight_id: str, image_id: str) -> Optional[dict[str, Any]]:
    if using_firebase_backend():
        return firebase_client.get_flight_image_doc(flight_id, image_id)
    if using_supabase_backend():
        return supabase_client.fetch_row("flight_patches", "id", f"{flight_id}:{image_id}")

    initialize()
    with _session_scope() as session:
        patch = session.get(FlightPatchRecord, f"{flight_id}:{image_id}")
        return None if patch is None else _patch_to_dict(patch)


def upsert_flight_image_doc(flight_id: str, image_id: str, payload: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.upsert_flight_image_doc(flight_id, image_id, payload)
        return
    if using_supabase_backend():
        supabase_client.upsert_row(
            "flight_patches",
            _supabase_payload(
                "flight_patches",
                {
                    "id": f"{flight_id}:{image_id}",
                    "flight_id": flight_id,
                    **payload,
                },
            ),
        )
        return

    initialize()
    with _session_scope() as session:
        patch_id = f"{flight_id}:{image_id}"
        patch = session.get(FlightPatchRecord, patch_id)
        if patch is None:
            patch = FlightPatchRecord(
                id=patch_id,
                flight_id=flight_id,
                image_id=image_id,
                patch_index=int(payload.get("patch_index", 0)),
                captured_at=payload.get("captured_at", _utc_now()),
                upload_status=payload.get("upload_status", "pending"),
                analysis_status=payload.get("analysis_status", "pending"),
            )
            session.add(patch)

        for key, value in payload.items():
            if hasattr(patch, key):
                setattr(patch, key, value)

        if "detections" in payload:
            session.query(FlightDetectionRecord).filter(
                FlightDetectionRecord.flight_patch_id == patch_id
            ).delete()
            detections = payload.get("detections") or []
            for index, detection in enumerate(detections):
                bbox = detection.get("bbox") or [None, None, None, None]
                session.add(
                    FlightDetectionRecord(
                        id=f"{patch_id}:{index}",
                        flight_patch_id=patch_id,
                        disease=str(detection.get("disease", "unknown")),
                        confidence=float(detection.get("confidence") or 0.0),
                        severity=str(detection.get("severity", "unknown")),
                        bbox_x1=bbox[0] if len(bbox) > 0 else None,
                        bbox_y1=bbox[1] if len(bbox) > 1 else None,
                        bbox_x2=bbox[2] if len(bbox) > 2 else None,
                        bbox_y2=bbox[3] if len(bbox) > 3 else None,
                        class_index=detection.get("class_index"),
                    )
                )


def list_flight_images(flight_id: str) -> list[dict[str, Any]]:
    if using_firebase_backend():
        return firebase_client.list_flight_images(flight_id)
    if using_supabase_backend():
        return supabase_client.list_rows_eq(
            "flight_patches",
            key="flight_id",
            value=flight_id,
            limit=500,
            order="patch_index.asc",
        )

    initialize()
    with _session_scope() as session:
        rows = session.scalars(
            select(FlightPatchRecord)
            .where(FlightPatchRecord.flight_id == flight_id)
            .order_by(FlightPatchRecord.patch_index.asc(), FlightPatchRecord.image_id.asc())
        ).all()
        return [_patch_to_dict(row) for row in rows]


def list_recent_photos(*, flight_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 200))
    if flight_id:
        rows = list_flight_images(flight_id)
        return sorted(
            rows,
            key=lambda row: (
                str(row.get("uploaded_at") or row.get("captured_at") or ""),
                int(row.get("patch_index") or 0),
            ),
            reverse=True,
        )[:safe_limit]

    if using_firebase_backend():
        return []
    if using_supabase_backend():
        return supabase_client.list_rows(
            "flight_patches",
            limit=safe_limit,
            order="uploaded_at.desc",
        )

    initialize()
    with _session_scope() as session:
        rows = session.scalars(
            select(FlightPatchRecord)
            .order_by(FlightPatchRecord.uploaded_at.desc(), FlightPatchRecord.patch_index.desc())
            .limit(safe_limit)
        ).all()
        return [_patch_to_dict(row) for row in rows]


def _content_type_extension(content_type: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get((content_type or "").lower(), "bin")


def _flight_object_key(
    *,
    flight_id: str,
    image_id: str,
    extension: str,
    storage_folder: Optional[str] = None,
    patch_index: Optional[int] = None,
) -> str:
    """Build the storage key for a flight image.

    When a per-mission `storage_folder` is supplied (e.g. mission_20260429_143022_a3f9)
    we group the file under `drone-images/<folder>/frame_NNN.<ext>` so each mission
    owns a self-contained directory. Otherwise we fall back to the legacy
    `flights/<flight_id>/originals/<image_id>.<ext>` layout for backwards compat.
    """
    if storage_folder:
        clean_folder = storage_folder.strip("/")
        if patch_index is not None:
            filename = f"frame_{int(patch_index) + 1:03d}.{extension}"
        else:
            filename = f"{image_id}.{extension}"
        return f"drone-images/{clean_folder}/{filename}"
    return f"flights/{flight_id}/originals/{image_id}.{extension}"


def upload_flight_image_bytes(
    *,
    flight_id: str,
    image_id: str,
    raw_bytes: bytes,
    content_type: str,
    metadata: Optional[dict[str, str]] = None,
    storage_folder: Optional[str] = None,
    patch_index: Optional[int] = None,
) -> str:
    if using_firebase_backend():
        return firebase_client.upload_flight_image_bytes(
            flight_id=flight_id,
            image_id=image_id,
            raw_bytes=raw_bytes,
            content_type=content_type,
            metadata=metadata,
        )

    extension = _content_type_extension(content_type)
    key = _flight_object_key(
        flight_id=flight_id,
        image_id=image_id,
        extension=extension,
        storage_folder=storage_folder,
        patch_index=patch_index,
    )

    if using_supabase_backend():
        return supabase_client.upload_object(
            path=key,
            raw_bytes=raw_bytes,
            content_type=content_type,
            metadata=metadata,
        )

    if storage_mode() == "s3":
        client = _s3_client()
        extra_args: dict[str, Any] = {"ContentType": content_type or "application/octet-stream"}
        if metadata:
            extra_args["Metadata"] = metadata
        client.put_object(Bucket=_s3_bucket_name(), Key=key, Body=raw_bytes, **extra_args)
        return f"s3://{_s3_bucket_name()}/{key}"

    return _upload_local_file(raw_bytes, key)


def upload_inference_image_bytes(
    *,
    inference_id: str,
    raw_bytes: bytes,
    content_type: str,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    if using_firebase_backend():
        return firebase_client.upload_inference_image_bytes(
            inference_id=inference_id,
            raw_bytes=raw_bytes,
            content_type=content_type,
            metadata=metadata,
        )
    if using_supabase_backend():
        extension = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }.get((content_type or "").lower(), "bin")
        return supabase_client.upload_object(
            path=f"inferences/{inference_id}/source.{extension}",
            raw_bytes=raw_bytes,
            content_type=content_type,
            metadata=metadata,
        )

    extension = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get((content_type or "").lower(), "bin")
    key = f"inferences/{inference_id}/source.{extension}"

    if storage_mode() == "s3":
        client = _s3_client()
        extra_args: dict[str, Any] = {"ContentType": content_type or "application/octet-stream"}
        if metadata:
            extra_args["Metadata"] = metadata
        client.put_object(Bucket=_s3_bucket_name(), Key=key, Body=raw_bytes, **extra_args)
        return f"s3://{_s3_bucket_name()}/{key}"

    return _upload_local_file(raw_bytes, key)


def download_storage_bytes(storage_path: str) -> tuple[bytes, str]:
    if using_firebase_backend():
        return firebase_client.download_storage_bytes(storage_path)
    if using_supabase_backend():
        return supabase_client.download_object(storage_path)

    bucket, key = _normalize_storage_path(storage_path)
    if storage_mode() == "s3":
        client = _s3_client()
        response = client.get_object(Bucket=bucket or _s3_bucket_name(), Key=key)
        return response["Body"].read(), response.get("ContentType") or "application/octet-stream"

    return _download_local_file(key), "application/octet-stream"


def write_flight_summary(flight_id: str, summary_json: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.write_flight_summary(flight_id, summary_json)
        return
    if using_supabase_backend():
        supabase_client.patch_row(
            "flights",
            "flight_id",
            flight_id,
            _supabase_payload("flights", {"summary": summary_json}),
        )
        return

    update_flight_doc(flight_id, {"summary": summary_json})


def write_flight_report(flight_id: str, report_json: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.write_flight_report(flight_id, report_json)
        return
    if using_supabase_backend():
        supabase_client.upsert_row(
            "flight_reports",
            _supabase_payload(
                "flight_reports",
                {
                    "id": flight_id,
                    "flight_id": flight_id,
                    "top_disease": report_json.get("top_disease"),
                    "highest_severity": report_json.get("severity_assessment")
                    or report_json.get("highest_severity"),
                    "report_json": report_json,
                    "generated_at": _utc_now(),
                },
            ),
        )
        supabase_client.patch_row(
            "flights",
            "flight_id",
            flight_id,
            _supabase_payload("flights", {"report": report_json, "updated_at": _utc_now()}),
        )
        return

    top_disease = report_json.get("top_disease")
    highest_severity = report_json.get("severity_assessment") or report_json.get("highest_severity")
    with _session_scope() as session:
        report = session.get(FlightReportRecord, flight_id) or FlightReportRecord(
            id=flight_id,
            flight_id=flight_id,
            generated_at=_utc_now(),
        )
        report.top_disease = top_disease
        report.highest_severity = highest_severity
        report.report_json = report_json
        report.generated_at = _utc_now()
        session.add(report)

        flight = session.get(FlightRecord, flight_id)
        if flight is not None:
            flight.report = report_json
            flight.updated_at = _utc_now()


def create_inference_doc(inference_id: str, payload: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.create_inference_doc(inference_id, payload)
        return
    if using_supabase_backend():
        supabase_client.upsert_row(
            "inferences",
            _supabase_payload("inferences", {"inference_id": inference_id, **payload}),
        )
        return

    initialize()
    with _session_scope() as session:
        record = InferenceRecord(
            id=inference_id,
            source_kind=payload.get("source_kind", "upload"),
            source_label=payload.get("source_label"),
            crop_type=payload.get("crop", "rice"),
            crop_requested=payload.get("crop_requested", payload.get("crop", "rice")),
            crop_warning=payload.get("crop_warning"),
            status=payload.get("status", "processing"),
            request_content_type=payload.get("request_content_type"),
            storage_path=payload.get("storage_path"),
            image_size=payload.get("image_size"),
            disease=payload.get("disease"),
            confidence=float(payload.get("confidence") or 0.0),
            severity=payload.get("severity"),
            prediction_type=payload.get("prediction_type"),
            model_name=payload.get("model"),
            model_kind=payload.get("model_kind"),
            artifact_label=payload.get("artifact_label"),
            primary_detection=payload.get("primary_detection"),
            all_detections=payload.get("all_detections"),
            top_predictions=payload.get("top_predictions"),
            firebase_saved=bool(payload.get("firebase_saved", False)),
            backend_provider=payload.get("backend_provider", provider_name()),
            storage_mode_name=payload.get("storage_mode", storage_mode()),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
            error_message=payload.get("error_message"),
        )
        session.add(record)


def get_inference_doc(inference_id: str) -> Optional[dict[str, Any]]:
    if using_firebase_backend():
        return firebase_client.get_inference_doc(inference_id)
    if using_supabase_backend():
        record = supabase_client.fetch_row("inferences", "inference_id", inference_id)
        if record:
            record.setdefault("record_saved", True)
        return record

    initialize()
    with _session_scope() as session:
        record = session.get(InferenceRecord, inference_id)
        return None if record is None else _inference_to_dict(record)


def update_inference_doc(inference_id: str, payload: dict[str, Any]) -> None:
    if using_firebase_backend():
        firebase_client.update_inference_doc(inference_id, payload)
        return
    if using_supabase_backend():
        supabase_client.patch_row(
            "inferences",
            "inference_id",
            inference_id,
            _supabase_payload("inferences", payload),
        )
        return

    initialize()
    with _session_scope() as session:
        record = session.get(InferenceRecord, inference_id)
        if record is None:
            raise KeyError(f"Inference '{inference_id}' not found")
        for key, value in payload.items():
            if key == "storage_mode" and hasattr(record, "storage_mode_name"):
                setattr(record, "storage_mode_name", value)
            elif key == "crop" and hasattr(record, "crop_type"):
                setattr(record, "crop_type", value)
            elif hasattr(record, key):
                setattr(record, key, value)


def list_inference_docs(limit: int = 20) -> list[dict[str, Any]]:
    if using_firebase_backend():
        return firebase_client.list_inference_docs(limit=limit)
    if using_supabase_backend():
        records = supabase_client.list_rows("inferences", limit=limit)
        for record in records:
            record.setdefault("record_saved", True)
        return records

    initialize()
    with _session_scope() as session:
        rows = session.scalars(
            select(InferenceRecord)
            .order_by(InferenceRecord.created_at.desc())
            .limit(limit)
        ).all()
        return [_inference_to_dict(row) for row in rows]


def write_latest_detection(payload: dict[str, Any]) -> bool:
    if using_supabase_backend():
        supabase_client.upsert_row(
            "latest_detections",
            _supabase_payload("latest_detections", {"id": "latest", **payload}),
        )
        supabase_client.insert_row("detection_history", _supabase_payload("detection_history", payload))
        return True
    if not firebase_client.database_is_configured():
        return False
    firebase_client.write_detection_latest(payload)
    firebase_client.append_detection_history(payload)
    return True


def get_latest_field_sensor_snapshot(field_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not field_id:
        return None
    if using_firebase_backend():
        return firebase_client.get_latest_field_sensor_snapshot(field_id)
    return None


def get_device(device_id: str) -> Optional[dict[str, Any]]:
    if using_firebase_backend():
        if not firebase_client.firestore_is_configured():
            return None
        firestore = firebase_client.get_firestore_client()
        query = firestore.collection("flights").where("device_id", "==", device_id).limit(1).stream()
        for doc in query:
            payload = doc.to_dict() or {}
            return {
                "id": device_id,
                "field_id": payload.get("field_id"),
                "status": payload.get("status"),
                "last_seen_at": payload.get("updated_at"),
            }
        return None
    if using_supabase_backend():
        return supabase_client.fetch_row("devices", "id", device_id)

    initialize()
    with _session_scope() as session:
        device = session.get(DeviceRecord, device_id)
        if device is None:
            return None
        return {
            "id": device.id,
            "thing_name": device.thing_name,
            "hardware_type": device.hardware_type,
            "camera_type": device.camera_type,
            "firmware_version": device.firmware_version,
            "field_id": device.field_id,
            "status": device.status,
            "last_seen_at": device.last_seen_at,
            "last_reported_ip": device.last_reported_ip,
            "last_rssi": device.last_rssi,
        }


def list_field_flights(field_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if using_firebase_backend():
        if not firebase_client.firestore_is_configured():
            return []
        firestore = firebase_client.get_firestore_client()
        query = (
            firestore.collection("flights")
            .where("field_id", "==", field_id)
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() or {} for doc in query]
    if using_supabase_backend():
        return supabase_client.list_rows_eq(
            "flights",
            key="field_id",
            value=field_id,
            limit=limit,
            order="created_at.desc",
        )

    initialize()
    with _session_scope() as session:
        rows = session.scalars(
            select(FlightRecord)
            .where(FlightRecord.field_id == field_id)
            .order_by(FlightRecord.created_at.desc())
            .limit(limit)
        ).all()
        return [_flight_to_dict(row) for row in rows]


def health_summary() -> dict[str, Any]:
    supabase_status = (
        supabase_client.supabase_connection_status()
        if using_supabase_backend() or supabase_client.supabase_is_configured()
        else {
            "supabase_connection_ok": False,
            "supabase_connection_error": None,
        }
    )
    return {
        "provider": provider_name(),
        "storage_mode": storage_mode(),
        "database_url": _database_url() if using_aws_backend() else None,
        "firebase_configured": firebase_client.firebase_is_configured(),
        "firestore_configured": firebase_client.firestore_is_configured(),
        "rtdb_configured": firebase_client.database_is_configured(),
        "supabase_configured": supabase_client.supabase_is_configured(),
        "supabase_service_role_configured": supabase_client.supabase_service_role_is_configured(),
        **supabase_status,
        "aws_s3_bucket": _s3_bucket_name() or None,
        "iot_shadow_enabled": bool(os.getenv("AWS_IOT_SHADOW_UPDATE_TOPIC", "").strip()),
    }
