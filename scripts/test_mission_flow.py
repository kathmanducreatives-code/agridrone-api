import argparse
import json
import os
import sys
import uuid

import requests


def _mission_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _firebase_headers():
    token = os.getenv("FIREBASE_DB_SECRET", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _insert_image_records(mission_id: str, image_urls: list[str]) -> None:
    database_url = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
    if not database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for the mission harness")

    headers = {"Content-Type": "application/json", **_firebase_headers()}
    for index, image_url in enumerate(image_urls, start=1):
        image_id = f"img-{index}-{uuid.uuid4().hex[:8]}"
        payload = {
            "storage_url": image_url,
            "timestamp": f"2026-03-21T00:00:0{index}Z",
            "uploaded": True,
            "meta": {
                "gps": {"lat": 27.7172 + (index * 0.0001), "lng": 85.324 + (index * 0.0001)},
                "source": "test_mission_flow",
            },
        }
        response = requests.put(
            f"{database_url}/missions/{mission_id}/images/{image_id}.json",
            data=json.dumps(payload),
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local harness for the AgriDrone mission flow")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="Public Firebase Storage or other reachable image URL. Pass 1-2 times.",
    )
    args = parser.parse_args()

    if not args.image_urls:
        print("Provide at least one --image-url for a reachable JPEG/PNG file.", file=sys.stderr)
        return 1

    base_url = _mission_base_url(args.base_url)
    create_response = requests.post(
        f"{base_url}/missions",
        json={"crop": "rice", "capture_interval_ms": 3000, "notes": "local harness"},
        timeout=20,
    )
    create_response.raise_for_status()
    mission = create_response.json()
    mission_id = mission["missionId"]
    print(f"Created mission: {mission_id}")

    _insert_image_records(mission_id, args.image_urls)
    print(f"Inserted {len(args.image_urls)} RTDB image records")

    analyze_response = requests.post(f"{base_url}/missions/{mission_id}/analyze", timeout=120)
    analyze_response.raise_for_status()
    analysis = analyze_response.json()
    print(json.dumps(analysis, indent=2))

    status_response = requests.get(f"{base_url}/missions/{mission_id}", timeout=20)
    status_response.raise_for_status()
    mission_state = status_response.json()
    print("Mission status:", mission_state.get("status"))
    if mission_state.get("status") not in {"done", "error"}:
        raise RuntimeError(f"Unexpected mission status: {mission_state.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
