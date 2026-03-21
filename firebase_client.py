import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("agridrone.firebase")


class FirebaseConfigError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise FirebaseConfigError(f"Missing required Firebase environment variable: {name}")
    return value


def _load_credentials():
    from firebase_admin import credentials

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if service_account_json:
        try:
            return credentials.Certificate(json.loads(service_account_json))
        except json.JSONDecodeError as exc:
            raise FirebaseConfigError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc

    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if project_id:
        logger.info("firebase.using_application_default_credentials", extra={"project_id": project_id})
        return credentials.ApplicationDefault()

    raise FirebaseConfigError(
        "Firebase credentials are not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_PROJECT_ID."
    )


@lru_cache(maxsize=1)
def get_firebase_app():
    import firebase_admin

    if firebase_admin._apps:
        return firebase_admin.get_app()

    database_url = _required_env("FIREBASE_DATABASE_URL")
    storage_bucket = _required_env("FIREBASE_STORAGE_BUCKET")
    cred = _load_credentials()

    return firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": database_url,
            "storageBucket": storage_bucket,
        },
    )


def firebase_is_configured() -> bool:
    needed = ["FIREBASE_DATABASE_URL", "FIREBASE_STORAGE_BUCKET"]
    has_backend = bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip() or os.getenv("FIREBASE_PROJECT_ID", "").strip())
    return has_backend and all(os.getenv(name, "").strip() for name in needed)


def _db_ref(path: str):
    from firebase_admin import db

    app = get_firebase_app()
    return db.reference(path, app=app)


def get_mission(mission_id: str) -> Optional[dict[str, Any]]:
    data = _db_ref(f"/missions/{mission_id}").get()
    return data if isinstance(data, dict) else None


def create_mission(mission_id: str, payload: dict[str, Any]) -> None:
    _db_ref(f"/missions/{mission_id}").set(payload)


def set_mission_status(mission_id: str, status: str, extra_fields: Optional[dict[str, Any]] = None) -> None:
    payload: dict[str, Any] = {"status": status}
    if extra_fields:
        payload.update(extra_fields)
    _db_ref(f"/missions/{mission_id}").update(payload)


def list_mission_images(mission_id: str) -> list[dict[str, Any]]:
    images = _db_ref(f"/missions/{mission_id}/images").get() or {}
    results = []
    for image_id, payload in images.items():
        if not isinstance(payload, dict):
            continue
        url = payload.get("storage_url") or payload.get("url")
        results.append(
            {
                "imageId": image_id,
                "url": url,
                "timestamp": payload.get("timestamp"),
                "meta": payload.get("meta") or {},
                "uploaded": payload.get("uploaded", False),
                "raw": payload,
            }
        )
    return results


def write_image_result(mission_id: str, image_id: str, yolo_result_json: dict[str, Any]) -> None:
    _db_ref(f"/missions/{mission_id}/images/{image_id}/yolo").set(yolo_result_json)


def write_mission_report(mission_id: str, report_json: dict[str, Any]) -> None:
    _db_ref(f"/missions/{mission_id}/report").set(report_json)


def write_mission_summary(mission_id: str, summary_json: dict[str, Any]) -> None:
    _db_ref(f"/missions/{mission_id}/summary").set(summary_json)
