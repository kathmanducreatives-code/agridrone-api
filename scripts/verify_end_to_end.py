import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import firebase_client


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AgriDrone mission flow end-to-end with Firebase Storage and RTDB")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--crop", default="rice")
    parser.add_argument("--image", action="append", dest="images", required=True, help="Path to a local JPG file. Repeat 1-3 times.")
    parser.add_argument("--poll-timeout-sec", type=int, default=120)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    return parser.parse_args()


def _validate_images(image_paths: list[str]) -> list[Path]:
    if not 1 <= len(image_paths) <= 3:
        raise RuntimeError("Pass between 1 and 3 --image arguments.")
    paths: list[Path] = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Image path does not exist: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise RuntimeError(f"Image must be a JPG/JPEG file: {path}")
        paths.append(path)
    return paths


def _signed_url_for_blob(blob) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    try:
        return blob.generate_signed_url(version="v4", expiration=expires_at, method="GET")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate signed URL for {blob.name}: {exc}") from exc


def _create_mission(base_url: str, crop: str) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/missions",
        json={"crop": crop, "capture_interval_ms": 3000, "notes": "verify_end_to_end"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Mission creation failed: {response.status_code} {response.text}")
    payload = response.json()
    mission_id = payload.get("missionId")
    if not mission_id:
        raise RuntimeError(f"Mission creation did not return missionId: {payload}")
    return mission_id


def _upload_images_and_seed_rtdb(mission_id: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    bucket = firebase_client.get_storage_bucket()
    uploaded: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image_id = f"img-{index}-{uuid.uuid4().hex[:8]}"
        blob = bucket.blob(f"missions/{mission_id}/{image_path.name}")
        try:
            blob.upload_from_filename(str(image_path), content_type="image/jpeg")
        except Exception as exc:
            raise RuntimeError(f"Failed to upload {image_path} to Firebase Storage: {exc}") from exc

        download_url = _signed_url_for_blob(blob)
        payload = {
            "storage_url": download_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uploaded": True,
            "meta": {
                "source": "verify_end_to_end",
                "local_filename": image_path.name,
            },
        }
        try:
            firebase_client.write_mission_image_record(mission_id, image_id, payload)
        except Exception as exc:
            raise RuntimeError(f"Failed to write RTDB image record for {image_id}: {exc}") from exc
        uploaded.append({"imageId": image_id, "storage_path": blob.name, "storage_url": download_url})
    return uploaded


def _trigger_analyze(base_url: str, mission_id: str) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/missions/{mission_id}/analyze", timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Mission analyze failed: {response.status_code} {response.text}")
    return response.json()


def _poll_mission(base_url: str, mission_id: str, timeout_sec: int, interval_sec: float) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = requests.get(f"{base_url.rstrip('/')}/missions/{mission_id}", timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"Mission poll failed: {response.status_code} {response.text}")
        payload = response.json()
        status = payload.get("status")
        if status in {"done", "error"}:
            return payload
        time.sleep(interval_sec)
    raise RuntimeError(f"Polling timed out after {timeout_sec} seconds for mission {mission_id}")


def _count_detections(mission_state: dict[str, Any]) -> int:
    summary = mission_state.get("summary") or {}
    return int(sum((summary.get("counts_per_disease") or {}).values()))


def _find_image_failures(mission_state: dict[str, Any]) -> list[dict[str, Any]]:
    images = mission_state.get("images") or {}
    failures: list[dict[str, Any]] = []
    for image_id, payload in images.items():
        yolo = (payload or {}).get("yolo") or {}
        if yolo.get("status") == "error":
            failures.append(
                {
                    "imageId": image_id,
                    "error_code": yolo.get("error_code"),
                    "message": yolo.get("message"),
                }
            )
    return failures


def _print_final_summary(mission_id: str, mission_state: dict[str, Any], uploaded_images: list[dict[str, Any]]) -> int:
    summary = mission_state.get("summary") or {}
    failures = _find_image_failures(mission_state)
    status = mission_state.get("status")
    outcome = "PASS" if status == "done" else "FAIL"

    print(outcome)
    print(f"mission_id={mission_id}")
    print(f"images_uploaded={len(uploaded_images)}")
    print(f"images_analyzed={summary.get('processed_images', 0)}")
    print(f"images_failed={summary.get('failed_images', len(failures))}")
    print(f"total_detections={_count_detections(mission_state)}")
    print(f"report_path=/missions/{mission_id}/report")
    print(f"summary_path=/missions/{mission_id}/summary")
    if failures:
        print("failures=" + json.dumps(failures, indent=2))
    if status == "error":
        print(f"error_reason={mission_state.get('error_reason') or mission_state.get('error_message')}")
        return 1
    return 0


def main() -> int:
    args = _parse_args()
    _required_env("FIREBASE_SERVICE_ACCOUNT_JSON")
    _required_env("FIREBASE_DATABASE_URL")
    _required_env("FIREBASE_STORAGE_BUCKET")

    image_paths = _validate_images(args.images)

    try:
        firebase_client.get_firebase_app()
    except Exception as exc:
        print(f"FAIL\nfirebase_init_error={exc}", file=sys.stderr)
        return 1

    try:
        mission_id = _create_mission(args.base_url, args.crop)
        uploaded_images = _upload_images_and_seed_rtdb(mission_id, image_paths)
        _trigger_analyze(args.base_url, mission_id)
        mission_state = _poll_mission(args.base_url, mission_id, args.poll_timeout_sec, args.poll_interval_sec)
        return _print_final_summary(mission_id, mission_state, uploaded_images)
    except Exception as exc:
        print(f"FAIL\nerror={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
