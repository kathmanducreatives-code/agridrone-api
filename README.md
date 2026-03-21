# AgriDrone API

Mission-based batch backend for AgriDrone Guardian, with backward-compatible raw-image inference endpoints.

## Docs

- Operator and developer workflow: [AGENTS.md](/Users/prasidha/screeningpilot/screeningpilot/agridrone-api/AGENTS.md)

## Backward-Compatible Endpoints

- `POST /predict`: raw JPEG request body
- `POST /predict_form`: multipart upload for browser/manual testing
- `POST /predict_upload`: alias for Swagger/manual multipart testing
- `GET /health`: model and runtime health
- `GET /`: root probe endpoint, also supports `HEAD`

## Mission Endpoints

- `POST /missions`
- `GET /missions/{missionId}`
- `POST /missions/{missionId}/analyze`

## Required Environment Variables

- `FIREBASE_DATABASE_URL=https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app`
- `FIREBASE_STORAGE_BUCKET=agridrone-guardian.appspot.com`
- `FIREBASE_SERVICE_ACCOUNT_JSON=<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>`
- `FIREBASE_PROJECT_ID=agridrone-guardian`
- `MAX_CONCURRENT_INFER=1`

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

Local setup:
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

Verify localhost:
```bash
curl -s http://localhost:8000/health
python scripts/verify_end_to_end.py \
  --base-url http://localhost:8000 \
  --crop rice \
  --image ./test1.jpg \
  --image ./test2.jpg
```

Render setup:
- start command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- set `FIREBASE_DATABASE_URL=https://agridrone-guardian-default-rtdb.asia-southeast1.firebasedatabase.app`
- set `FIREBASE_STORAGE_BUCKET=agridrone-guardian.appspot.com`
- set `FIREBASE_SERVICE_ACCOUNT_JSON=<paste the single-line service account JSON for firebase-adminsdk-fbsvc@agridrone-guardian.iam.gserviceaccount.com>`
- set `FIREBASE_PROJECT_ID=agridrone-guardian`
- set `MAX_CONCURRENT_INFER=1`
- deploy and verify `GET /health` returns `200`

Verify Render:
```bash
python scripts/verify_end_to_end.py \
  --base-url https://agridrone-api.onrender.com \
  --crop rice \
  --image ./test1.jpg \
  --image ./test2.jpg
```

Use [AGENTS.md](/Users/prasidha/screeningpilot/screeningpilot/agridrone-api/AGENTS.md) for the full local run, Render deployment, and smoke-test workflow.
