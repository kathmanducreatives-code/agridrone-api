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

- `FIREBASE_SERVICE_ACCOUNT_JSON`
- `FIREBASE_DATABASE_URL`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_PROJECT_ID` (optional)
- `MAX_CONCURRENT_INFER=1`

## Verification Checklist

- `pip install -r requirements.txt`
- `pytest -q`
- `curl /health`
- `Create mission with POST /missions`
- `Add RTDB image records with storage_url`
- `Run POST /missions/{missionId}/analyze`
- `Confirm RTDB writes images/*/yolo, summary, report, and final status`
- `Deploy to Render with --workers 1 and MAX_CONCURRENT_INFER=1`

## How To Verify

Local setup:
```bash
cd /Users/prasidha/screeningpilot/screeningpilot/agridrone-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
export FIREBASE_DATABASE_URL='https://<project>.firebaseio.com'
export FIREBASE_STORAGE_BUCKET='<project>.appspot.com'
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
  --image /absolute/path/to/test1.jpg
```

Render setup:
- deploy with `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- set `MAX_CONCURRENT_INFER=1`
- set Firebase env vars in the Render dashboard

Verify Render:
```bash
python scripts/verify_end_to_end.py \
  --base-url https://agridrone-api.onrender.com \
  --crop rice \
  --image /absolute/path/to/test1.jpg
```

Use [AGENTS.md](/Users/prasidha/screeningpilot/screeningpilot/agridrone-api/AGENTS.md) for the full local run, Render deployment, and smoke-test workflow.
