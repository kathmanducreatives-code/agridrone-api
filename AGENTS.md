# AgriDrone API Agent Guide

This repo hosts the FastAPI backend for AgriDrone Guardian. Keep changes backward compatible with the existing raw-image inference flow while adding mission-based processing on top.

## Local Run

1. Create a virtualenv and install dependencies:
```bash
cd /Users/prasidha/screeningpilot/screeningpilot/agridrone-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Set environment variables:
```bash
export FIREBASE_DATABASE_URL='https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app'
export FIREBASE_STORAGE_BUCKET='agridrone-guardian.appspot.com'
export FIREBASE_SERVICE_ACCOUNT_JSON='<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>'
export FIREBASE_PROJECT_ID='agridrone-guardian'
export MAX_CONCURRENT_INFER=1
```
3. Start the API:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Required Environment Variables

- `FIREBASE_DATABASE_URL`: Firebase Realtime Database base URL.
- `FIREBASE_DATABASE_URL`: `https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app`
- `FIREBASE_STORAGE_BUCKET`: `agridrone-guardian.appspot.com`
- `FIREBASE_SERVICE_ACCOUNT_JSON`: Full single-line Firebase Admin service account JSON for `firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com`
- `FIREBASE_PROJECT_ID`: `agridrone-guardian`
- `MAX_CONCURRENT_INFER`: Defaults to `1`; keep it at `1` on Render.
- `MODEL_GDRIVE_ID`: Optional bootstrap path for downloading the rice ONNX model on startup.
- `MODELS_DIR`: Optional model directory, defaults to `./models`.
- `DEBUG_IMAGE_DIR`: Optional local debug image directory, defaults to `./debug`.
- `OPENAI_API_KEY`: Optional placeholder only. Report generation currently stays local and deterministic.

## Render Deploy

- Use a single web worker only. Do not increase `uvicorn` workers because model memory is loaded in-process.
- Recommended start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```
- Keep the Render instance on one worker and `MAX_CONCURRENT_INFER=1` to avoid free-tier memory spikes.
- Set these Render environment variables:
  - `FIREBASE_DATABASE_URL=https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app`
  - `FIREBASE_STORAGE_BUCKET=agridrone-guardian.appspot.com`
  - `FIREBASE_SERVICE_ACCOUNT_JSON=<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>`
  - `FIREBASE_PROJECT_ID=agridrone-guardian`
  - `MAX_CONCURRENT_INFER=1`
- Deploy checklist:
  - Confirm start command is `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
  - Confirm only one Render web worker is running
  - Confirm all Firebase env vars are present in the Render dashboard
  - Deploy and wait for `/health` to return `200`

## Smoke Tests

Health:
```bash
curl -s http://localhost:8000/health
```

Raw JPEG predict:
```bash
curl -X POST "http://localhost:8000/predict?crop=rice&save_to_firebase=true" \
  -H "Content-Type: image/jpeg" \
  --data-binary @leaf.jpg
```

Multipart predict:
```bash
curl -X POST "http://localhost:8000/predict_upload?crop=rice" \
  -F "image=@leaf.jpg"
```

Mission create:
```bash
curl -X POST "http://localhost:8000/missions" \
  -H "Content-Type: application/json" \
  -d '{"crop":"rice","capture_interval_ms":3000,"notes":"manual assisted-grid test"}'
```

Mission analyze:
```bash
curl -X POST "http://localhost:8000/missions/<missionId>/analyze"
```

Mission fetch:
```bash
curl -s "http://localhost:8000/missions/<missionId>"
```

Verification script:
```bash
python scripts/verify_end_to_end.py \
  --base-url http://localhost:8000 \
  --crop rice \
  --image /absolute/path/to/test1.jpg \
  --image /absolute/path/to/test2.jpg
```

The verification script uploads local JPG files to Firebase Storage under `missions/{missionId}/`, writes RTDB image records with `storage_url`, runs mission analysis, polls for completion, and prints a final `PASS` or `FAIL` summary.

## Verification Checklist

- `pip install -r requirements.txt`
- `pytest -q`
- `curl /health`
- `POST /missions (create mission)`
- `upload 1–3 images to Firebase Storage under missions/{missionId}/`
- `write RTDB: /missions/{missionId}/images/{imageId}/storage_url`
- `POST /missions/{missionId}/analyze`
- `confirm RTDB writes yolo, summary, report, status done`

## How To Verify

Local environment:
```bash
cd /Users/prasidha/screeningpilot/screeningpilot/agridrone-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FIREBASE_DATABASE_URL='https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app'
export FIREBASE_STORAGE_BUCKET='agridrone-guardian.appspot.com'
export FIREBASE_SERVICE_ACCOUNT_JSON='<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>'
export FIREBASE_PROJECT_ID='agridrone-guardian'
export MAX_CONCURRENT_INFER=1
pytest -q
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Local verification against localhost:
```bash
curl -s http://localhost:8000/health
python scripts/verify_end_to_end.py \
  --base-url http://localhost:8000 \
  --crop rice \
  --image ./test1.jpg \
  --image ./test2.jpg
```

Render deploy:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

Render environment values:
```bash
FIREBASE_DATABASE_URL=https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app
FIREBASE_STORAGE_BUCKET=agridrone-guardian.appspot.com
FIREBASE_SERVICE_ACCOUNT_JSON=<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>
FIREBASE_PROJECT_ID=agridrone-guardian
MAX_CONCURRENT_INFER=1
```

Render verification:
```bash
python scripts/verify_end_to_end.py \
  --base-url https://agridrone-api.onrender.com \
  --crop rice \
  --image ./test1.jpg \
  --image ./test2.jpg
```

## Development Rules

- Preserve `/predict`, `/predict_form`, `/predict_upload`, `/health`, and `/`.
- Return structured JSON errors for mission failures and persist mission `status=error` in Firebase where appropriate.
- Do not hardcode any secrets or service-account material in the codebase.
- Keep dependencies lean and prefer CPU-friendly ONNX inference paths.
